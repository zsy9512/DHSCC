"""
Independent Channel Module for DHSCC

This module implements the complete channel processing including:
- Wireless channel simulation (AWGN, Rayleigh, UDP)
- Modulation and demodulation
- Channel encoding and decoding (separated into sender/receiver sides)
- Interface for sender and receiver components

This is the third independent component as required by the refactoring plan.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from wireless_comm.interface import WirelessInterface, create_wireless_interface
from wireless_comm.wireless_communication import WirelessCodec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ChannelModule:
    """
    Independent channel module handling complete wireless transmission.
    
    This module provides a clean interface between sender and receiver,
    handling all channel-related processing including modulation, demodulation,
    channel coding, and wireless channel simulation.
    
    Args:
        args: Configuration parameters for channel setup
    """
    def __init__(self, args):
        self.comm_mode = getattr(args, 'communication_mode', 1)
        self.channel_model = getattr(args, 'channel_model', 'awgn')
        self.threshold = getattr(args, 'threshold', 0.5)
        
        # Initialize wireless interface for digital communication
        if self.comm_mode in [3, 4]:
            self.wireless_interface = create_wireless_interface(args)
            self.wireless_codec = None
            if hasattr(self.wireless_interface, 'wireless_codec'):
                self.wireless_codec = self.wireless_interface.wireless_codec
            print(f"Initialized channel module with communication mode {self.comm_mode}")
            print(f"Channel model: {self.channel_model}, "
                  f"Equalization: {getattr(args, 'equalization_method', 'mrc')}")
        else:
            self.wireless_interface = None
            self.wireless_codec = None
            print(f"Initialized channel module with communication mode {self.comm_mode}")
            print("Note: Digital interface not required for analog modes")
            
        # Store configuration for reference
        self.config = {
            'comm_mode': self.comm_mode,
            'channel_model': self.channel_model,
            'bits_per_index': getattr(args, 'bits_per_index', 4),
            'max_groups': getattr(args, 'max_groups', 8),
            'constellation_type': getattr(args, 'constellation_type', 'qam'),
            'equalization_method': getattr(args, 'equalization_method', 'mrc')
        }
    
    def transmit_analog(self, features, snr_db, importance_weights=None):
        """
        Transmit features through analog channel (modes 1, 2).
        
        Args:
            features: Features to transmit [batch_size, feature_dim]
            snr_db: Signal-to-noise ratio in dB
            importance_weights: Optional importance weights for feature masking
            
        Returns:
            dict: Transmission results containing received features and metrics
        """
        batch_size = features.size(0)
        
        # Convert SNR to noise variance
        snr_linear = 10 ** (snr_db / 10)
        noise_variance = 1.0 / snr_linear
        
        if self.comm_mode == 1:
            # Mode 1: Direct analog transmission with importance masking
            if importance_weights is not None:
                # Apply threshold-based masking
                mask = (importance_weights > self.threshold).float()
                transmitted_features = features * mask
                zeros_count = (mask == 0).sum().item()
            else:
                transmitted_features = features
                zeros_count = 0
                
            # Add Gaussian noise
            noise = torch.randn_like(transmitted_features) * torch.sqrt(torch.tensor(noise_variance))
            received_features = transmitted_features + noise
            
            # Calculate transmission MSE
            transmit_mse = F.mse_loss(received_features, features)
            
        elif self.comm_mode == 2:
            # Mode 2: Analog transmission with quantization (already quantized features)
            # Add Gaussian noise directly
            noise = torch.randn_like(features) * torch.sqrt(torch.tensor(noise_variance))
            received_features = features + noise
            
            transmit_mse = F.mse_loss(received_features, features)
            zeros_count = 0
            
        return {
            'received_features': received_features,
            'transmit_mse': transmit_mse,
            'ber': torch.tensor(0.0),  # Not applicable for analog
            'zeros_count': zeros_count,
            'type': 'analog'
        }
    
    def transmit_digital(self, indices, snr_db, group_to_use=8, use_coding=True):
        """
        Transmit quantization indices through digital channel (modes 3, 4).
        
        Args:
            indices: Quantization indices [batch_size, n_groups, group_size]
            snr_db: Signal-to-noise ratio in dB
            group_to_use: Number of groups to transmit
            use_coding: Whether to use channel coding (mode 4)
            
        Returns:
            dict: Transmission results containing received indices and metrics
        """
        if self.wireless_interface is None:
            raise RuntimeError("Wireless interface not initialized for digital communication")
        
        batch_size, n_groups, group_size = indices.shape
        
        # Only transmit the first group_to_use groups
        transmit_indices = indices[:, :group_to_use, :]
        
        if self.comm_mode == 3:
            # Mode 3: Digital transmission without channel coding
            result = self.wireless_interface.transmit_mode3(
                transmit_indices,
                snr_db=snr_db,
                bits_per_index=self.config['bits_per_index'],
                group_to_use=group_to_use
            )
        elif self.comm_mode == 4:
            # Mode 4: Digital transmission with channel coding
            result = self.wireless_interface.transmit_mode4(
                transmit_indices,
                snr_db=snr_db,
                bits_per_index=self.config['bits_per_index'],
                group_to_use=group_to_use,
                use_coding=use_coding
            )
        else:
            raise ValueError(f"Invalid communication mode for digital transmission: {self.comm_mode}")
        
        # Reconstruct full indices with zero-padding for untransmitted groups
        received_indices_partial = result['received_indices']
        received_indices_full = torch.zeros_like(indices, device=indices.device)
        received_indices_full[:, :group_to_use, :] = received_indices_partial
        
        # Calculate zeros count for untransmitted groups
        zeros_count = (n_groups - group_to_use) * group_size * batch_size
        
        return {
            'received_indices': received_indices_full,
            'transmitted_indices': transmit_indices,
            'ber': result['bit_error_rate'],
            'zeros_count': zeros_count,
            'group_to_use': group_to_use,
            'type': 'digital'
        }
    
    def process_transmission(self, data, snr_db):
        """
        Main interface for processing transmission through channel.
        
        Args:
            data: Data from sender (features for analog, indices for digital)
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            dict: Processed transmission results
        """
        if isinstance(data, dict):
            # Structured data from sender
            if data['type'] == 'analog':
                return self.transmit_analog(
                    features=data['features'],
                    snr_db=snr_db,
                    importance_weights=data.get('metadata', {}).get('importance_weights')
                )
            elif data['type'] == 'digital':
                return self.transmit_digital(
                    indices=data['indices'],
                    snr_db=snr_db,
                    group_to_use=data.get('group_to_use', 8),
                    use_coding=(self.comm_mode == 4)
                )
            else:
                raise ValueError(f"Unknown data type: {data['type']}")
        else:
            # Assume tensor input for backward compatibility
            if self.comm_mode in [1, 2]:
                return self.transmit_analog(data, snr_db)
            else:
                # For digital modes, assume indices format
                return self.transmit_digital(data, snr_db)
    
    def get_channel_encoding_interface(self):
        """
        Get interface for channel encoding (sender side).
        
        Returns:
            ChannelEncoder: Encoder interface for sender
        """
        return ChannelEncoder(self.wireless_interface, self.comm_mode)
    
    def get_channel_decoding_interface(self):
        """
        Get interface for channel decoding (receiver side).
        
        Returns:
            ChannelDecoder: Decoder interface for receiver
        """
        return ChannelDecoder(self.wireless_interface, self.comm_mode)


class ChannelEncoder:
    """
    Channel encoding interface for sender side.
    
    This class provides encoding functionality that should be integrated
    into the sender training process.
    """
    def __init__(self, wireless_interface, comm_mode):
        self.wireless_interface = wireless_interface
        self.comm_mode = comm_mode
    
    def encode_for_transmission(self, data, metadata=None):
        """
        Encode data for channel transmission.
        
        Args:
            data: Data to encode (features or indices)
            metadata: Additional metadata for encoding
            
        Returns:
            Encoded data ready for modulation
        """
        if self.comm_mode in [3, 4] and self.wireless_interface:
            # Digital encoding (polar codes for mode 4)
            if hasattr(self.wireless_interface, 'prepare_transmission_data'):
                return self.wireless_interface.prepare_transmission_data(
                    data, 
                    bits_per_index=4,
                    group_to_use=metadata.get('group_to_use', 8) if metadata else 8
                )
        
        # For analog modes or fallback
        return data


class ChannelDecoder:
    """
    Channel decoding interface for receiver side.
    
    This class provides decoding functionality that should be integrated
    into the receiver training process.
    """
    def __init__(self, wireless_interface, comm_mode):
        self.wireless_interface = wireless_interface
        self.comm_mode = comm_mode
    
    def decode_from_channel(self, received_data, metadata=None):
        """
        Decode data received from channel.
        
        Args:
            received_data: Data received from channel
            metadata: Additional metadata for decoding
            
        Returns:
            Decoded data ready for receiver processing
        """
        if self.comm_mode in [3, 4] and self.wireless_interface:
            # Digital decoding
            if isinstance(received_data, dict) and 'received_indices' in received_data:
                return received_data['received_indices']
        
        # For analog modes or direct data
        if isinstance(received_data, dict) and 'received_features' in received_data:
            return received_data['received_features']
        
        return received_data


def create_channel_module(args):
    """
    Factory function to create channel module based on configuration.
    
    Args:
        args: Configuration parameters
        
    Returns:
        ChannelModule: Configured channel module
    """
    return ChannelModule(args)