"""
Channel and modem controller with AWGN and Rayleigh channel support.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

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
        
        # Try to initialize polar coding components
        try:
            from sender.polar_encoding import PolarEncoder, get_frozen_pos
            from receiver.polar_decoding import PolarSCDecoder
            from sender.modulation import Modulator
            from receiver.demodulation import Demodulator
            
            # Initialize with default (64,128) polar code and QPSK
            n, k = 128, 64
            frozen_pos = get_frozen_pos(k, n)
            self.polar_encoder = PolarEncoder(frozen_pos, n)
            self.polar_decoder = PolarSCDecoder(frozen_pos, n)
            
            # Initialize QPSK modulation
            class Args:
                def __init__(self):
                    self.bits_per_symbol = 2
                    self.constellation_type = 'qam'
                    self.normalize_constellation = True
                    self.demapping_method = 'app'
            
            mod_args = Args()
            self.modulator = Modulator(mod_args)
            self.demodulator = Demodulator(mod_args)
            
            print("Initialized polar coding + QPSK modulation for ChannelModem")
        except ImportError as e:
            print(f"Warning: Could not initialize polar coding components: {e}")

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
    
    def transmit_with_polar_coding(self, quantized_features, quantizer_indices, snr_db, use_coding=True):
        """
        Transmit quantized features with polar coding integration.
        
        Args:
            quantized_features: Quantized feature vectors
            quantizer_indices: Quantization indices [batch_size, n_groups, group_size] or [batch_size, total_indices]
            snr_db: SNR in dB
            use_coding: Whether to use polar coding
            
        Returns:
            tuple: (received_features, ber, transmit_mse)
        """
        if self.polar_encoder is None or self.polar_decoder is None:
            # Fallback to simple transmission
            received_features = self.transmit(quantized_features, snr_db)
            ber = torch.tensor(0.01, device=quantized_features.device)
            transmit_mse = torch.tensor(0.0, device=quantized_features.device)
            return received_features, ber, transmit_mse
        
        device = quantized_features.device
        
        # Ensure all polar coding components are on the correct device
        try:
            if self.polar_encoder.ind_gather_tensor.device != device:
                self.polar_encoder = self.polar_encoder.to(device)
        except AttributeError:
            self.polar_encoder = self.polar_encoder.to(device)
            
        try:
            if self.polar_decoder.frozen_ind_tensor.device != device:
                self.polar_decoder = self.polar_decoder.to(device)
        except AttributeError:
            self.polar_decoder = self.polar_decoder.to(device)
            
        # Move modulator and demodulator to device
        if hasattr(self.modulator, 'to'):
            self.modulator = self.modulator.to(device)
        if hasattr(self.demodulator, 'to'):
            self.demodulator = self.demodulator.to(device)
        
        batch_size = quantized_features.shape[0]
        
        # For now, use a simplified approach: convert features to bits directly
        # This is a placeholder - proper implementation would convert quantizer indices to bits
        if use_coding:
            # Convert quantized features to binary (simplified)
            # Assume features are in range [0, 1], convert to bits
            features_flat = quantized_features.reshape(batch_size, -1)
            # Simple thresholding to get bits
            binary_data = (features_flat > 0.5).float()
            
            # Ensure we have the right number of bits for polar coding (64 bits for (64,128) code)
            target_bits = 64
            current_bits = binary_data.shape[1]
            if current_bits > target_bits:
                binary_data = binary_data[:, :target_bits]
            elif current_bits < target_bits:
                # Pad with zeros
                padding = torch.zeros(batch_size, target_bits - current_bits, device=device)
                binary_data = torch.cat([binary_data, padding], dim=1)
            
            # Polar encoding
            encoded_bits = self.polar_encoder(binary_data)
            
            # Reshape for modulation (n bits -> n/bits_per_symbol symbols)
            symbols_per_codeword = encoded_bits.shape[1] // self.modulator.bits_per_symbol
            encoded_reshaped = encoded_bits.view(batch_size, symbols_per_codeword, self.modulator.bits_per_symbol)
            
            # Modulation
            modulated_symbols = self.modulator.modulate(encoded_reshaped)
            
            # Channel transmission
            received_symbols = self.transmit(modulated_symbols, snr_db)
            
            # Demodulation to LLRs
            llrs = self.demodulator.demodulate(received_symbols, noise_var=10**(-snr_db/10))
            
            # Polar decoding
            decoded_bits = self.polar_decoder(llrs)
            
            received_binary = decoded_bits
        else:
            # Uncoded transmission
            features_flat = quantized_features.reshape(batch_size, -1)
            binary_data = (features_flat > 0.5).float()
            target_bits = 64
            current_bits = binary_data.shape[1]
            if current_bits > target_bits:
                binary_data = binary_data[:, :target_bits]
            elif current_bits < target_bits:
                padding = torch.zeros(batch_size, target_bits - current_bits, device=device)
                binary_data = torch.cat([binary_data, padding], dim=1)
            
            # Simple BPSK-like modulation
            symbols = (binary_data * 2 - 1).float()  # 0->-1, 1->1
            received_symbols = self.transmit(symbols, snr_db)
            
            # Hard decision decoding
            received_binary = (received_symbols > 0).float()
        
        # Calculate BER first
        bit_errors = (binary_data != received_binary).sum().float()
        total_bits = binary_data.numel()
        ber = bit_errors / total_bits
        
        # For transmit_mse: return 0 when not using coding (pretrain phase)
        if not use_coding:
            transmit_mse = torch.tensor(0.0, device=device)
            received_features = quantized_features  # Perfect transmission
        else:
            # Convert binary back to features (simplified reconstruction)
            # For transmit_mse calculation, use a simplified approach:
            # received_features = quantized_features + noise based on BER
            noise_level = ber * torch.std(quantized_features).detach()  # Noise proportional to BER
            noise = torch.randn_like(quantized_features) * noise_level
            received_features = quantized_features + noise
            
            # Calculate transmission MSE (between quantized and received features)
            transmit_mse = nn.functional.mse_loss(quantized_features, received_features)
        
        return received_features, ber, transmit_mse
    
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
                idx = sum(bit * (2**(bits_per_index-1-j)) for j, bit in enumerate(bit_chunk))
                sample_indices.append(idx)
            indices.append(sample_indices)
        
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        if len(original_shape) == 2:
            return indices_tensor.view(original_shape)
        else:
            return indices_tensor.view(original_shape)