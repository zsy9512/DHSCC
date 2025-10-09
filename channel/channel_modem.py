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
        
        # Create WirelessInterface directly with correct parameters
        self.wireless_interface = WirelessInterface(
            comm_mode=getattr(args, 'communication_mode', 4),
            n=getattr(args, 'N', 256),
            k=getattr(args, 'K', 128),
            channel_type=getattr(args, 'channel_type', 'uplink'),
            decoder_type=getattr(args, 'decoder_type', 'SC'),
            list_size=getattr(args, 'list_size', 8),
            bits_per_symbol=getattr(args, 'bits_per_symbol', 2),
            constellation_type=getattr(args, 'constellation_type', 'qam'),
            normalize=getattr(args, 'normalize', True),
            demapping_method=getattr(args, 'demapping_method', 'app'),
            channel_model=getattr(args, 'channel_model', 'awgn'),
            equalization_method=getattr(args, 'equalization_method', 'mrc'),
            device=str(device),  # Use the device from this module
            verbose=getattr(args, 'verbose', False)
        )

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
        if self.wireless_interface is not None and use_coding:
            # Use WirelessInterface for proper adaptive polar coding
            device = quantized_features.device
            
            # Convert indices to the format expected by WirelessInterface
            # quantizer_indices shape: [batch_size, n_groups, group_size]
            # We need to select the first 'group_to_use' groups
            if len(quantizer_indices.shape) == 3:
                # Select groups based on group_to_use
                selected_indices = quantizer_indices[:, :group_to_use, :]
                # print(f"📡 ChannelModem: Selected {group_to_use} groups from 3D indices shape {quantizer_indices.shape} -> {selected_indices.shape}")
            else:
                # Assume flat indices, select first group_to_use indices
                selected_indices = quantizer_indices[:, :group_to_use] if quantizer_indices.shape[1] > group_to_use else quantizer_indices
                # print(f"📡 ChannelModem: Selected {group_to_use} indices from 2D indices shape {quantizer_indices.shape} -> {selected_indices.shape}")
            
            # Transmit using WirelessInterface
            try:
                result = self.wireless_interface.transmit(
                    indices=selected_indices,
                    snr_db=snr_db,
                    bits_per_index=4,  # Assuming 4 bits per index
                    group_to_use=group_to_use,
                    use_coding=use_coding  # Pass the use_coding parameter
                )
                
                # Extract received indices, BER, and MSE from result
                received_indices = result.get('received_indices', selected_indices)
                ber = result.get('bit_error_rate', torch.tensor(0.01, device=device))
                transmit_mse = result.get('index_error_rate', torch.tensor(0.0, device=device))
                
                # Decode received indices back to features if quantizer is provided
                if quantizer is not None and hasattr(quantizer, 'decode'):
                    try:
                        received_features = quantizer.decode(received_indices)
                    except Exception as e:
                        print(f"Warning: Failed to decode received indices: {e}, using original quantized features")
                        received_features = quantized_features
 
                return received_features, ber, transmit_mse
            except Exception as e:
                print(f"Error using WirelessInterface: {e}, falling back to simple transmission")
        
        # Fallback to simple transmission
        received_features = self.transmit(quantized_features, snr_db)
        ber = torch.tensor(0.01, device=quantized_features.device)
        transmit_mse = torch.tensor(0.0, device=quantized_features.device)
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