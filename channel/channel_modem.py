"""
Channel and modem controller with AWGN and Rayleigh channel support.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from .interface import WirelessInterface

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WirelessChannel:
    """
    Unified wireless channel supporting AWGN and Rayleigh fading models
    """

    def __init__(self, channel_model: str = "awgn", device: str = "cpu"):
        """
        Initialize the wireless channel

        Args:
            channel_model: Channel model type ('awgn' or 'rayleigh')
            device: Device to run computations on
        """
        self.channel_model = channel_model.lower()
        self.device = device

        if self.channel_model not in ['awgn', 'rayleigh']:
            raise ValueError(f"Unsupported channel model: {channel_model}")

    def add_awgn(self,
                 symbols: torch.Tensor,
                 snr_db: Optional[float] = None,
                 ebn0_db: Optional[float] = None,
                 bits_per_symbol: Optional[int] = None,
                 code_rate: Optional[float] = None) -> Tuple[torch.Tensor, float]:
        """
        Add Additive White Gaussian Noise to symbols

        Args:
            symbols: Real or complex symbols
            snr_db: Signal-to-Noise ratio in dB
            ebn0_db: Energy per bit to noise power spectral density ratio in dB
            bits_per_symbol: Number of bits per symbol (required for Eb/N0)
            code_rate: Code rate (required for coded systems with Eb/N0)

        Returns:
            tuple: (noisy_symbols, noise_power)
        """
        # Check if symbols are complex or real
        is_complex = symbols.is_complex()

        # Calculate symbol energy (assuming normalized constellation)
        symbol_energy = torch.mean(torch.abs(symbols)**2).item()

        # Calculate noise power based on provided parameter
        if snr_db is not None:
            # Convert SNR from dB to linear scale
            snr_linear = 10.0**(snr_db/10.0)
            noise_power = symbol_energy / snr_linear
        elif ebn0_db is not None:
            if bits_per_symbol is None:
                raise ValueError("bits_per_symbol must be provided when using ebn0_db")

            # Convert Eb/N0 from dB to linear scale
            ebn0_linear = 10.0**(ebn0_db/10.0)

            # Calculate bit energy
            bit_energy = symbol_energy / bits_per_symbol

            # Apply code rate if provided (for coded systems)
            if code_rate is not None:
                bit_energy = bit_energy * code_rate

            # Calculate noise power based on bit energy and Eb/N0
            noise_power = bit_energy / ebn0_linear
        else:
            raise ValueError("Either snr_db or ebn0_db must be provided")

        # Generate appropriate noise for complex and real symbols
        if is_complex:
            # Complex signal noise components - each dimension gets half the power
            noise_var_per_dim = noise_power / 2
            if not torch.is_tensor(noise_var_per_dim):
                noise_var_per_dim = torch.tensor(noise_var_per_dim, device=symbols.real.device)
            else:
                noise_var_per_dim = noise_var_per_dim.to(symbols.real.device)
            noise_real = torch.randn_like(symbols.real) * torch.sqrt(noise_var_per_dim)
            noise_imag = torch.randn_like(symbols.imag) * torch.sqrt(noise_var_per_dim)
            noise = torch.complex(noise_real, noise_imag)
        else:
            # Real signal noise components - BPSK uses all noise power for one dimension
            noise = torch.randn_like(symbols) * np.sqrt(noise_power)

        # Add noise to symbols
        noisy_symbols = symbols + noise

        return noisy_symbols, noise_power

    def add_rayleigh_channel(self,
                           symbols: torch.Tensor,
                           snr_db: Optional[float] = None,
                           ebn0_db: Optional[float] = None,
                           bits_per_symbol: Optional[int] = None,
                           code_rate: Optional[float] = None) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Add Rayleigh fading channel with AWGN

        Args:
            symbols: Real or complex symbols
            snr_db: Signal-to-Noise ratio in dB
            ebn0_db: Energy per bit to noise power spectral density ratio in dB
            bits_per_symbol: Number of bits per symbol (required for Eb/N0)
            code_rate: Code rate (required for coded systems with Eb/N0)

        Returns:
            tuple: (received_symbols, noise_power, channel_coefficients)
        """
        # Check if symbols are complex or real
        is_complex = symbols.is_complex()

        # Convert real symbols to complex for Rayleigh fading processing
        if not is_complex:
            symbols_complex = torch.complex(symbols, torch.zeros_like(symbols))
        else:
            symbols_complex = symbols

        # Generate Rayleigh fading coefficients (complex Gaussian random variables, variance=1)
        h_real = torch.randn_like(symbols_complex.real) / np.sqrt(2)
        h_imag = torch.randn_like(symbols_complex.imag) / np.sqrt(2)
        h = torch.complex(h_real, h_imag)

        # Apply Rayleigh fading
        faded_symbols = symbols_complex * h

        # Add AWGN to the faded symbols
        noisy_symbols, noise_power = self.add_awgn(
            faded_symbols, snr_db, ebn0_db, bits_per_symbol, code_rate
        )

        # Convert back to real if input was real
        if not is_complex:
            noisy_symbols = noisy_symbols.real

        return noisy_symbols, noise_power, h

    def transmit(self,
                 symbols: torch.Tensor,
                 snr_db: float,
                 bits_per_symbol: Optional[int] = None,
                 code_rate: Optional[float] = None) -> Tuple[torch.Tensor, float, Optional[torch.Tensor]]:
        """
        Transmit symbols through the wireless channel

        Args:
            symbols: Input symbols to transmit
            snr_db: Signal-to-noise ratio in dB
            bits_per_symbol: Number of bits per symbol (for Eb/N0 calculation)
            code_rate: Code rate (for coded systems)

        Returns:
            tuple: (received_symbols, noise_power, channel_coefficients)
                channel_coefficients is None for AWGN, h tensor for Rayleigh
        """
        if self.channel_model == 'awgn':
            received_symbols, noise_power = self.add_awgn(symbols, snr_db=snr_db)
            return received_symbols, noise_power, None
        elif self.channel_model == 'rayleigh':
            received_symbols, noise_power, h = self.add_rayleigh_channel(
                symbols, snr_db=snr_db, bits_per_symbol=bits_per_symbol, code_rate=code_rate
            )
            return received_symbols, noise_power, h
        else:
            raise ValueError(f"Unsupported channel model: {self.channel_model}")

class Channel(nn.Module):
    """
    Simplified channel interface for dhscc_split
    """
    def __init__(self, args):
        super().__init__()
        self.channel_model = getattr(args, 'channel_model', 'awgn')
        self.wireless_channel = WirelessChannel(
            channel_model=self.channel_model,
            device=str(device)
        )

    def transmit(self, symbols, snr):
        """
        Transmit symbols through channel.

        Args:
            symbols: Input symbols
            snr: SNR in dB

        Returns:
            Received symbols with noise/channel effects
        """
        received_symbols, _, _ = self.wireless_channel.transmit(symbols, snr_db=snr)
        return received_symbols

class ChannelModem:
    """
    Channel controller with AWGN and Rayleigh channel support.
    Modulation/demodulation is handled separately in sender/receiver.
    """
    def __init__(self, args):
        self.channel = Channel(args)
        # Initialize polar coding components
        self.polar_encoder = None
        self.polar_decoder = None
        self.modulator = None
        self.demodulator = None
        
        # Store parameters from args
        self.k = getattr(args, 'K', 128)
        self.n = getattr(args, 'N', 256)
        self.bits_per_symbol = getattr(args, 'bits_per_symbol', 2)
        self.constellation_type = getattr(args, 'constellation_type', 'qam')
        self.demapping_method = getattr(args, 'demapping_method', 'app')
        self.normalize = getattr(args, 'normalize', True)
        self.channel_type = getattr(args, 'channel_type', 'uplink')
        self.decoder_type = getattr(args, 'decoder_type', 'SC')
        self.list_size = getattr(args, 'list_size', 8)
        self.verbose = getattr(args, 'verbose', False)

    def transmit(self, symbols, snr):
        """
        Transmit symbols through channel

        Args:
            symbols: Modulated symbols to transmit
            snr: SNR in dB

        Returns:
            Received symbols after channel transmission
        """
        return self.channel.transmit(symbols, snr)
    
    def polar_encode(self, binary_bits: torch.Tensor) -> torch.Tensor:
        """
        Perform 5G polar encoding on binary bits.

        Args:
            binary_bits: Binary bits tensor, shape [batch_size, k]

        Returns:
            codeword: Encoded codeword, shape [batch_size, n]
        """
        # Ensure polar_encoder is initialized
        if self.polar_encoder is None:
            from .polar_encoding import Polar5GEncoder
            self.polar_encoder = Polar5GEncoder(
                k=self.k,
                n=self.n,
                channel_type=self.channel_type,
                verbose=self.verbose,
                dtype=torch.float32
            ).to(device)
        
        # Perform encoding
        codeword = self.polar_encoder(binary_bits.float())
        return codeword
    
    def modulate(self, codeword: torch.Tensor) -> torch.Tensor:
        """
        Modulate codeword into symbols.

        Args:
            codeword: Codeword tensor, shape [batch_size, n]

        Returns:
            symbols: Modulated symbols, shape [batch_size, n // bits_per_symbol]
        """
        # Ensure modulator is initialized
        if self.modulator is None:
            from .mapping import Mapper
            self.modulator = Mapper(
                constellation_type=self.constellation_type,
                num_bits_per_symbol=self.bits_per_symbol
            )
        
        # Perform modulation
        symbols = self.modulator(codeword)
        return symbols
    
    def add_channel_noise(self, symbols: torch.Tensor, snr_db: float) -> Tuple[torch.Tensor, float]:
        """
        Add channel noise to symbols.

        Args:
            symbols: Input symbols, shape [batch_size, ...]
            snr_db: SNR in dB

        Returns:
            tuple: (noisy_symbols, noise_power)
        """
        # Use existing WirelessChannel
        code_rate = self.k / self.n
        noisy_symbols, noise_power = self.channel.wireless_channel.add_awgn(
            symbols, snr_db=snr_db, bits_per_symbol=self.bits_per_symbol, code_rate=code_rate
        )
        return noisy_symbols, noise_power
    
    def demodulate(self, noisy_symbols: torch.Tensor, noise_power: float) -> torch.Tensor:
        """
        Demodulate noisy symbols to LLRs.

        Args:
            noisy_symbols: Noisy symbols, shape [batch_size, ...]
            noise_power: Noise power (variance)

        Returns:
            llrs: Log-likelihood ratios, shape [batch_size, n]
        """
        # Ensure demodulator is initialized
        if self.demodulator is None:
            from .mapping import Demapper
            self.demodulator = Demapper(
                demapping_method=self.demapping_method,
                constellation_type=self.constellation_type,
                num_bits_per_symbol=self.bits_per_symbol
            )
        
        # Perform demodulation
        llrs = self.demodulator(noisy_symbols, noise_power)
        # Demapper returns [batch_size, n_symbols, bits_per_symbol], flatten to [batch_size, n]
        batch_size, n_symbols = noisy_symbols.shape
        n = n_symbols * self.bits_per_symbol
        llrs = llrs.reshape(batch_size, n)
        return llrs
    
    def polar_decode(self, llrs: torch.Tensor) -> torch.Tensor:
        """
        Perform 5G polar decoding on LLRs.

        Args:
            llrs: Log-likelihood ratios, shape [batch_size, n]

        Returns:
            decoded_bits: Decoded binary bits, shape [batch_size, k]
        """
        # Ensure polar_decoder is initialized
        if self.polar_decoder is None:
            from .polar_decoding import Polar5GDecoder
            # Create a temporary encoder to get the correct parameters for the decoder
            from .polar_encoding import Polar5GEncoder
            temp_encoder = Polar5GEncoder(
                k=self.k,
                n=self.n,
                channel_type=self.channel_type,
                verbose=self.verbose,
                dtype=torch.float32
            )
            self.polar_decoder = Polar5GDecoder(
                enc_polar=temp_encoder,
                dec_type=self.decoder_type,
                list_size=self.list_size,
                return_crc_status=False,
                dtype=torch.float32
            ).to(device)
        
        # Perform decoding
        decoded_bits = self.polar_decoder(llrs)
        return decoded_bits
    
    def transmit_with_polar_coding(self, quantized_features, quantizer_indices, snr_db, use_coding=True, group_to_use=8, quantizer=None):
        """
        Transmit quantized features with polar coding integration.
        
        Args:
            quantized_features: Quantized feature vectors
            quantizer_indices: Quantization indices [batch_size, n_groups, group_size] or [batch_size, total_indices]
            snr_db: SNR in dB
            use_coding: Whether to use polar coding
            group_to_use: Number of groups to transmit (for adaptive selection)
            quantizer: Quantizer instance for decoding received indices back to features
            
        Returns:
            tuple: (received_features, ber, transmit_mse)
        """
        # Use split functions for polar coding
        device = quantized_features.device
        
        # Convert indices to the format expected
        if len(quantizer_indices.shape) == 3:
            selected_indices = quantizer_indices[:, :group_to_use, :]
        else:
            selected_indices = quantizer_indices[:, :group_to_use] if quantizer_indices.shape[1] > group_to_use else quantizer_indices
        
        # Split transmission process
        binary_bits = self.indices_to_binary(selected_indices, bits_per_index=4)
        codeword = self.polar_encode(binary_bits)
        symbols = self.modulate(codeword)
        noisy_symbols, noise_power = self.add_channel_noise(symbols, snr_db)
        llrs = self.demodulate(noisy_symbols, noise_power)
        decoded_bits = self.polar_decode(llrs)
        received_indices = self.binary_to_indices(decoded_bits, selected_indices.shape, bits_per_index=4)
        
        # Calculate BER
        bit_errors = (binary_bits != decoded_bits).sum().float()
        total_bits = binary_bits.numel()
        ber = bit_errors / total_bits
        
        # Calculate index error rate
        index_errors = (selected_indices != received_indices).sum().float()
        total_indices = selected_indices.numel()
        transmit_mse = index_errors / total_indices
        
        # Decode received indices back to features if quantizer is provided
        if quantizer is not None and hasattr(quantizer, 'decode'):
            try:
                received_features = quantizer.decode(received_indices)
            except Exception as e:
                print(f"Warning: Failed to decode received indices: {e}, using original quantized features")
                received_features = quantized_features
        else:
            received_features = quantized_features

        return received_features, ber, transmit_mse
    
    def transmit_encoded_codeword(self, quantized_features, codeword, snr_db, quantizer=None, selected_indices_shape=None):
        """
        Transmit already encoded codeword through channel.
        
        Args:
            quantized_features: Quantized feature vectors
            codeword: Already polar-encoded codeword, shape [batch_size, n]
            snr_db: SNR in dB
            quantizer: Quantizer instance for decoding received indices back to features (not used here)
            selected_indices_shape: Shape of original selected_indices for reconstruction (not used here)
            
        Returns:
            tuple: (llrs, ber, transmit_mse) - llrs for receiver to decode
        """
        # Use split functions for channel transmission (encoding already done)
        device = quantized_features.device
        
        # Modulate codeword
        symbols = self.modulate(codeword)
        noisy_symbols, noise_power = self.add_channel_noise(symbols, snr_db)
        llrs = self.demodulate(noisy_symbols, noise_power)
        
        # Return LLRs for receiver to perform polar decoding
        # BER and transmit_mse are placeholders since decoding happens at receiver
        ber = torch.tensor(0.0, device=device)  # Placeholder BER
        transmit_mse = torch.tensor(0.0, device=device)  # Placeholder
        
        return llrs, ber, transmit_mse
    
    def indices_to_binary(self, indices, bits_per_index=4):
        """Convert quantizer indices to binary representation."""
        device = indices.device
        # Handle different shapes: (batch_size, total_indices) or (batch_size, n_groups, group_size)
        if len(indices.shape) == 2:
            batch_size, total_indices = indices.shape
            flat_indices = indices.reshape(-1)
        else:
            batch_size, n_groups, group_size = indices.shape
            flat_indices = indices.reshape(-1)
            total_indices = n_groups * group_size
        
        # Convert to binary
        binary = []
        for idx in flat_indices:
            bin_str = format(idx.item(), f'0{bits_per_index}b')
            binary.extend([int(b) for b in bin_str])
        
        binary_tensor = torch.tensor(binary, dtype=torch.float, device=device)
        return binary_tensor.view(batch_size, -1)
    
    def binary_to_indices(self, binary, original_shape, bits_per_index=4):
        """Convert binary back to quantizer indices."""
        device = binary.device
        batch_size, total_bits = binary.shape
        indices_per_sample = total_bits // bits_per_index
        
        indices = []
        for b in range(batch_size):
            sample_bits = binary[b]
            sample_indices = []
            for i in range(0, total_bits, bits_per_index):
                bit_chunk = sample_bits[i:i+bits_per_index]
                # Convert to int for calculation
                bit_values = bit_chunk.int().tolist()
                idx = sum(bit_val * (2**(bits_per_index-1-j)) for j, bit_val in enumerate(bit_values))
                sample_indices.append(idx)
            indices.append(sample_indices)
        
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        if len(original_shape) == 2:
            return indices_tensor.view(original_shape)
        else:
            return indices_tensor.view(original_shape)