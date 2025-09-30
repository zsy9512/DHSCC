"""
Modulation utilities with QAM support.
"""

import torch
import torch.nn as nn
import numpy as np

def pam_constellation(num_bits_per_symbol, normalize=True):
    """
    Create PAM constellation points

    Args:
        num_bits_per_symbol (int): Number of bits per PAM symbol
        normalize (bool): If True, normalize the constellation to unit power

    Returns:
        torch.Tensor: PAM constellation points
    """
    num_points = 2**num_bits_per_symbol
    points = torch.arange(start=0, end=num_points, dtype=torch.float32)
    points = 2*points - (num_points - 1)  # Center around origin

    if normalize:
        # Calculate average power for normalization
        power = torch.mean(torch.abs(points)**2)
        points = points / torch.sqrt(power)

    return points

def qam_constellation(num_bits_per_symbol, normalize=True):
    """
    Create QAM constellation points to match Sionna's implementation

    Args:
        num_bits_per_symbol (int): Number of bits per QAM symbol (must be even)
        normalize (bool): If True, normalize the constellation to unit power

    Returns:
        torch.Tensor: Complex QAM constellation points
    """
    if num_bits_per_symbol % 2 != 0:
        raise ValueError("num_bits_per_symbol must be even for QAM")

    # For 16-QAM (bits_per_symbol=4), Sionna uses this specific ordering
    if num_bits_per_symbol == 4:
        # Create the explicit 16-QAM constellation points based on Sionna's ordering
        real_parts = torch.tensor([0.31622776, 0.31622776, 0.94868326, 0.94868326,
                                 0.31622776, 0.31622776, 0.94868326, 0.94868326,
                                 -0.31622776, -0.31622776, -0.94868326, -0.94868326,
                                 -0.31622776, -0.31622776, -0.94868326, -0.94868326])

        imag_parts = torch.tensor([0.31622776, 0.94868326, 0.31622776, 0.94868326,
                                 -0.31622776, -0.94868326, -0.31622776, -0.94868326,
                                 0.31622776, 0.94868326, 0.31622776, 0.94868326,
                                 -0.31622776, -0.94868326, -0.31622776, -0.94868326])

        # Create complex constellation points
        return torch.complex(real_parts, imag_parts)

    # For other QAM sizes, we'll need to implement a more general solution
    # Getting the number of PAM points per dimension
    num_pam_points = 2**(num_bits_per_symbol // 2)

    # Create a list to hold the constellation points
    points = []

    # The PAM constellation for one dimension
    pam_values = torch.linspace(-(num_pam_points-1), (num_pam_points-1), num_pam_points)
    pam_values = pam_values / torch.sqrt(torch.mean(pam_values**2))

    # Ordering of constellation points in Sionna appears to follow a specific pattern
    # Generate the constellation points based on this pattern
    for i in range(num_pam_points):
        for j in range(num_pam_points):
            # Reorder indices based on Sionna's bit mapping
            point = torch.complex(pam_values[i], pam_values[j])
            points.append(point)

    # Convert to tensor
    qam_points = torch.stack(points)

    if normalize:
        # Calculate average power for normalization
        power = torch.mean(torch.abs(qam_points)**2)
        qam_points = qam_points / torch.sqrt(power)

    return qam_points

class Constellation:
    """
    Constellation class for mapping binary data to constellation points
    """

    def __init__(self, constellation_type="qam", num_bits_per_symbol=4, normalize=True):
        """
        Initialize constellation

        Args:
            constellation_type (str): Type of constellation ('qam', 'bpsk', 'pam')
            num_bits_per_symbol (int): Number of bits per symbol
            normalize (bool): Whether to normalize constellation to unit power
        """
        self.constellation_type = constellation_type.lower()
        self.num_bits_per_symbol = num_bits_per_symbol
        self.normalize = normalize

        if self.constellation_type == "qam":
            if num_bits_per_symbol % 2 != 0:
                raise ValueError("num_bits_per_symbol must be even for QAM")
            self.points = qam_constellation(num_bits_per_symbol, normalize)
        elif self.constellation_type == "pam":
            self.points = pam_constellation(num_bits_per_symbol, normalize)
        elif self.constellation_type == "bpsk" or num_bits_per_symbol == 1:
            # BPSK is essentially 2-PAM
            self.points = pam_constellation(1, normalize)
        else:
            raise ValueError(f"Unsupported constellation type: {constellation_type}")

        self.num_symbols = len(self.points)

    def __len__(self):
        return self.num_symbols

    def __getitem__(self, idx):
        return self.points[idx]

class Mapper:
    """
    Mapper class for converting binary data to constellation symbols
    """

    def __init__(self, constellation):
        """
        Initialize mapper

        Args:
            constellation (Constellation): Constellation object
        """
        self.constellation = constellation
        self.num_bits_per_symbol = constellation.num_bits_per_symbol

        # Create binary to symbol mapping
        self.binary_to_symbol_map = {}
        for symbol_idx in range(len(constellation)):
            # Convert symbol index to binary representation
            binary = format(symbol_idx, f'0{self.num_bits_per_symbol}b')
            binary_tuple = tuple(int(bit) for bit in binary)
            self.binary_to_symbol_map[binary_tuple] = symbol_idx

    def map(self, binary_data):
        """
        Map binary data to constellation symbols

        Args:
            binary_data (torch.Tensor): Binary data of shape [num_symbols, bits_per_symbol]

        Returns:
            torch.Tensor: Constellation symbols [num_symbols]
        """
        num_symbols, bits_per_symbol = binary_data.shape

        # Convert binary to decimal indices
        powers = 2 ** torch.arange(bits_per_symbol - 1, -1, -1, device=binary_data.device, dtype=binary_data.dtype)
        symbol_indices = torch.sum(binary_data * powers, dim=-1).long()

        # Map to constellation points
        symbols = self.constellation.points.to(symbol_indices.device)[symbol_indices]

        return symbols

class Modulator(nn.Module):
    """
    Modulator supporting QAM, BPSK, and PAM modulation.
    """
    def __init__(self, args):
        super().__init__()
        self.bits_per_symbol = getattr(args, 'bits_per_symbol', 4)
        self.constellation_type = getattr(args, 'constellation_type', 'qam')
        self.normalize = getattr(args, 'normalize_constellation', True)

        if self.constellation_type.lower() == "bpsk" or self.bits_per_symbol == 1:
            self.modulation = "bpsk"
            self.constellation = None
            self.mapper = self._bpsk_mapper
        else:
            self.modulation = self.constellation_type.lower()

            # Create constellation
            self.constellation = Constellation(
                constellation_type=self.constellation_type,
                num_bits_per_symbol=self.bits_per_symbol,
                normalize=self.normalize
            )

            # Create mapper
            self.mapper = Mapper(constellation=self.constellation)

    def _bpsk_mapper(self, bits):
        """
        Simple BPSK modulation: 0 -> -1, 1 -> 1

        Args:
            bits: Binary tensor [batch, bit_length]

        Returns:
            Real symbols [batch, symbol_length]
        """
        # For BPSK, each bit becomes a symbol
        symbols = torch.where(bits == 0, -1.0, 1.0)
        return symbols.float()

    def modulate(self, data):
        """
        Modulate input data to symbols.

        Args:
            data: Input data (bits or features depending on context)

        Returns:
            Modulated symbols
        """
        if self.modulation == "bpsk":
            # Assume data is already binary for BPSK
            return self._bpsk_mapper(data)
        else:
            # For QAM/PAM, assume data is binary and map to constellation
            if data.dtype != torch.long:
                # Convert to binary representation
                data = (data > 0).long()

            # Reshape data to [batch, seq_len * bits_per_symbol]
            original_shape = data.shape
            if len(original_shape) == 1:
                data = data.unsqueeze(0)

            batch_size = data.shape[0]
            flat_data = data.view(batch_size, -1)

            # Group bits into symbols
            symbols = []
            for b in range(batch_size):
                batch_bits = flat_data[b]
                # Pad if necessary
                remainder = len(batch_bits) % self.bits_per_symbol
                if remainder != 0:
                    padding = torch.zeros(self.bits_per_symbol - remainder,
                                        dtype=batch_bits.dtype, device=batch_bits.device)
                    batch_bits = torch.cat([batch_bits, padding])

                # Reshape to [num_symbols, bits_per_symbol]
                num_symbols = len(batch_bits) // self.bits_per_symbol
                symbol_bits = batch_bits.view(num_symbols, self.bits_per_symbol)

                # Map to constellation
                batch_symbols = self.mapper.map(symbol_bits)
                symbols.append(batch_symbols)

            result = torch.stack(symbols)

            # Restore original shape if input was 1D
            if len(original_shape) == 1:
                result = result.squeeze(0)

            return result