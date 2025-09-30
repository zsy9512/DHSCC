"""
Demodulation utilities with QAM support.
"""

import torch
import torch.nn as nn
import numpy as np

class Demapper:
    """
    Demapper class for converting constellation symbols to LLRs or binary data
    """

    def __init__(self, constellation, demapping_method="app"):
        """
        Initialize demapper

        Args:
            constellation (Constellation): Constellation object
            demapping_method (str): Demapping method ('app' for APP, 'maxlog' for max-log-APP)
        """
        self.constellation = constellation
        self.num_bits_per_symbol = constellation.num_bits_per_symbol
        self.demapping_method = demapping_method.lower()

        # Precompute bit mappings for constellation points
        self._precompute_bit_mappings()

    def _precompute_bit_mappings(self):
        """
        Precompute bit mappings for the constellation
        Creates C0 and C1 arrays which contain indices of constellation points
        where each bit position is 0 or 1 respectively
        """
        num_points = self.constellation.num_symbols
        num_bits = self.num_bits_per_symbol

        # Generate binary representation of all symbol indices
        a = torch.zeros(num_points, num_bits, dtype=torch.int)
        for i in range(num_points):
            # Convert i to binary and store each bit
            bin_str = format(i, f'0{num_bits}b')
            for j in range(num_bits):
                a[i, j] = int(bin_str[j])

        # Find indices where each bit is 0 or 1
        c0 = []
        c1 = []
        for i in range(num_bits):
            c0.append(torch.where(a[:, i] == 0)[0])
            c1.append(torch.where(a[:, i] == 1)[0])

        self._c0 = c0  # List of tensors with indices where bit i is 0
        self._c1 = c1  # List of tensors with indices where bit i is 1

    def demap(self, symbols, noise_var):
        """
        Demap constellation symbols to LLRs

        Args:
            symbols: Complex symbols [batch, seq_len]
            noise_var: Noise variance

        Returns:
            LLRs [batch, seq_len * bits_per_symbol]
        """
        device = symbols.device

        # Ensure noise_var is a tensor
        if not isinstance(noise_var, torch.Tensor):
            noise_var = torch.tensor(noise_var, device=device)

        # Expand constellation points for broadcasting with symbols
        points = self.constellation.points.to(device)

        # Calculate squared distances from symbols to all constellation points
        # Shape: [batch, seq_len, num_points]
        symbols_expanded = symbols.unsqueeze(-1)  # [batch, seq_len, 1]
        points_expanded = points.reshape(1, 1, -1)  # [1, 1, num_points]
        squared_dist = torch.abs(symbols_expanded - points_expanded) ** 2

        # Compute the exponents for probability calculation
        exponents = -squared_dist / noise_var

        # Compute LLRs for each bit
        llrs = []
        for i in range(self.num_bits_per_symbol):
            # Get exponents for symbols where bit i is 0 or 1
            c0_i = self._c0[i].to(device)
            c1_i = self._c1[i].to(device)

            exponents_0 = exponents[..., c0_i]
            exponents_1 = exponents[..., c1_i]

            if self.demapping_method == "app":
                # Use logarithm of sum of exponentials
                llr_i = torch.logsumexp(exponents_1, dim=-1) - torch.logsumexp(exponents_0, dim=-1)
            else:  # maxlog approximation
                # Use maximum of exponents
                llr_i = torch.max(exponents_1, dim=-1).values - torch.max(exponents_0, dim=-1).values

            llrs.append(llr_i)

        # Stack LLRs along a new dimension and then reshape
        llr_tensor = torch.stack(llrs, dim=-1)

        # Reshape to [batch, seq_len * bits_per_symbol]
        batch_size, seq_len, _ = llr_tensor.shape
        return llr_tensor.view(batch_size, seq_len * self.num_bits_per_symbol)

class Demodulator(nn.Module):
    """
    Demodulator supporting QAM, BPSK, and PAM demodulation.
    """
    def __init__(self, args):
        super().__init__()
        self.bits_per_symbol = getattr(args, 'bits_per_symbol', 4)
        self.constellation_type = getattr(args, 'constellation_type', 'qam')
        self.normalize = getattr(args, 'normalize_constellation', True)
        self.demapping_method = getattr(args, 'demapping_method', 'app')

        if self.constellation_type.lower() == "bpsk" or self.bits_per_symbol == 1:
            self.modulation = "bpsk"
            self.demapper = self._bpsk_demapper
        else:
            self.modulation = self.constellation_type.lower()

            # Import constellation from sender modulation module
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from sender.modulation import Constellation

            # Create constellation
            self.constellation = Constellation(
                constellation_type=self.constellation_type,
                num_bits_per_symbol=self.bits_per_symbol,
                normalize=self.normalize
            )

            # Create demapper
            self.demapper = Demapper(
                constellation=self.constellation,
                demapping_method=self.demapping_method
            )

    def to(self, device):
        """Move the demodulator to the specified device"""
        if hasattr(self, 'constellation') and hasattr(self.constellation, 'points'):
            self.constellation.points = self.constellation.points.to(device)
        return super().to(device)

    def _bpsk_demapper(self, symbols):
        """
        Simple BPSK demodulation: <0 -> 0, >=0 -> 1

        Args:
            symbols: Real symbols [batch, symbol_length]

        Returns:
            Binary tensor [batch, bit_length]
        """
        # For BPSK, each symbol becomes a bit
        bits = torch.where(symbols < 0, 0, 1)
        return bits.long()

    def demodulate(self, symbols, noise_var=None):
        """
        Demodulate symbols to bits or LLRs.

        Args:
            symbols: Complex or real symbols [batch, symbol_length]
            noise_var: Noise variance (required for soft demodulation)

        Returns:
            Binary tensor [batch, bit_length] or LLRs
        """
        # Ensure constellation is on the same device as input
        if hasattr(self, 'constellation') and hasattr(self.constellation, 'points'):
            self.constellation.points = self.constellation.points.to(symbols.device)

        if self.modulation == "bpsk":
            # Assume symbols are real for BPSK
            if symbols.is_complex():
                symbols = symbols.real
            return self._bpsk_demapper(symbols)
        else:
            # For QAM/PAM, use demapper
            if noise_var is None:
                raise ValueError("noise_var must be provided for soft demodulation")
            return self.demapper.demap(symbols, noise_var)