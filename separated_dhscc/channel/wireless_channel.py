"""
Wireless Channel Module for Vibration Signal Classification

This module implements different communication modes for feature transmission:
- Mode 1: Direct analog transmission with importance-based masking
- Mode 2: Quantized analog transmission with Gaussian noise
- Mode 3: Digital transmission without channel coding
- Mode 4: Digital transmission with channel coding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from wireless_comm.interface import WirelessInterface, create_wireless_interface

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WirelessChannelModule(nn.Module):
    """
    Wireless channel module supporting multiple communication modes.
    
    This module simulates transmission through wireless channels with different
    encoding strategies and noise characteristics.
    
    Args:
        args: Configuration parameters containing communication settings
    """
    def __init__(self, args):
        super().__init__()
        
        self.comm_mode = getattr(args, 'communication_mode', 1)  # 使用正确的参数名
        self.threshold = getattr(args, 'threshold', 0.5)
        
        # Wireless interface for digital communication (modes 3 & 4)
        # 只有在通信模式3和4下才需要配置信道模型
        if self.comm_mode in [3, 4]:
            self.wireless_interface = create_wireless_interface(args)
            print(f"Initialized wireless channel with communication mode {self.comm_mode}")
            print(f"Channel model: {getattr(args, 'channel_model', 'awgn')}, "
                  f"Equalization: {getattr(args, 'equalization_method', 'mrc')}")
        else:
            self.wireless_interface = None
            print(f"Initialized wireless channel with communication mode {self.comm_mode}")
            print("Note: Channel model configuration not required for analog modes")
        
    def forward(self, features, channel_noise, psnr_value, **kwargs):
        """
        Simulate transmission through wireless channel based on communication mode.
        
        Args:
            features: Input features to transmit
            channel_noise: Channel noise level
            psnr_value: PSNR value for transmission
            **kwargs: Mode-specific parameters:
                - attn_weights: Attention weights for mode 1
                - quantized_features: Quantized features for modes 2, 3, 4
                - indices: Quantization indices for modes 3, 4
                - group_to_use: Number of groups to transmit for modes 3, 4
                - use_coding: Whether to use channel coding for mode 4
                - quantizer_decode_fn: Function to decode indices
                
        Returns:
            tuple: (received_features, ber, transmit_mse, zeros_count)
                received_features: Features after channel transmission
                ber: Bit error rate (for digital modes)
                transmit_mse: Transmission MSE loss
                zeros_count: Number of pruned features (for mode 1)
        """
        if self.comm_mode == 1:
            return self._mode1_analog_transmission(features, channel_noise, kwargs.get('attn_weights'))
        elif self.comm_mode == 2:
            return self._mode2_quantized_analog(kwargs.get('quantized_features'), channel_noise)
        elif self.comm_mode == 3:
            return self._mode3_digital_no_coding(
                kwargs.get('quantized_features'), 
                kwargs.get('indices'),
                psnr_value, 
                kwargs.get('group_to_use', 8),
                kwargs.get('quantizer_decode_fn')
            )
        elif self.comm_mode == 4:
            # Mode 4: Digital transmission with adaptive coding based on training phase
            use_coding = kwargs.get('use_coding', True)
            return self._mode4_digital_with_coding(
                kwargs.get('quantized_features'),
                kwargs.get('indices'),
                psnr_value,
                kwargs.get('group_to_use', 8),
                kwargs.get('quantizer_decode_fn'),
                use_coding=use_coding
            )
        else:
            raise ValueError(f"Unsupported communication mode: {self.comm_mode}")
    
    def _mode1_analog_transmission(self, features, channel_noise, attn_weights):
        """
        Mode 1: Direct analog transmission with importance-based masking.
        
        Features are pruned based on importance weights and transmitted with Gaussian noise.
        
        Args:
            features: Input features [batch_size, feature_dim]
            channel_noise: Channel noise level
            attn_weights: Attention weights for importance masking
            
        Returns:
            tuple: (received_features, ber, transmit_mse, zeros_count)
        """
        # Generate importance-based mask
        mask = self._get_importance_mask(attn_weights).mean(dim=0, keepdim=True)
        zeros_count = torch.sum(mask == 0).item()
        
        # Apply mask to features
        sent_features = features * mask
        
        # Add Gaussian noise for analog transmission
        noise = torch.randn_like(features) * channel_noise
        received_features = sent_features + noise
        
        # Calculate transmission MSE
        transmit_mse = F.mse_loss(received_features, sent_features, reduction='mean')
        
        return received_features, torch.tensor(0.0).to(device), transmit_mse, sent_features
    
    def _mode2_quantized_analog(self, quantized_features, channel_noise):
        """
        Mode 2: Quantized analog transmission with Gaussian noise.
        
        Quantized features are transmitted through analog channel with additive noise.
        
        Args:
            quantized_features: Pre-quantized features
            channel_noise: Channel noise level
            
        Returns:
            tuple: (received_features, ber, transmit_mse, zeros_count)
        """
        # Add Gaussian noise to quantized features
        noise = torch.randn_like(quantized_features) * channel_noise
        received_features = quantized_features + noise
        
        # Calculate transmission MSE
        transmit_mse = F.mse_loss(received_features, quantized_features, reduction='mean')
        
        return received_features, torch.tensor(0.0).to(device), transmit_mse, quantized_features
    
    def _mode3_digital_no_coding(self, quantized_features, indices, psnr_value, group_to_use, decode_fn):
        """
        Mode 3: Digital transmission without channel coding.
        
        Only the first group_to_use groups are transmitted. Untransmitted groups
        are zero-padded during reconstruction to ensure receiver only uses transmitted data.
        
        Args:
            quantized_features: Original quantized features (for gradient flow only)
            indices: Quantization indices [batch_size, n_groups, group_size] 
            psnr_value: PSNR for transmission
            group_to_use: Number of groups to transmit (1-8)
            decode_fn: Function to decode indices back to features
            
        Returns:
            tuple: (received_features, ber, transmit_mse, zeros_count)
        """
        if self.wireless_interface is None:
            raise RuntimeError("Wireless interface not initialized for digital communication")
        
        batch_size, n_groups, group_size = indices.shape
        
        # === TRANSMISSION: Only transmit the first group_to_use groups ===
        transmit_indices = indices[:, :group_to_use, :]
        
        # Get sent features from only transmitted groups (for loss calculation)
        sent_features = decode_fn(transmit_indices)
        
        # Digital transmission without coding
        result = self.wireless_interface.transmit_mode3(
            transmit_indices,
            snr_db=psnr_value.item() if hasattr(psnr_value, 'item') else psnr_value,
            bits_per_index=4,
            group_to_use=group_to_use
        )
        
        received_indices_partial = result['received_indices']  # Only group_to_use groups
        ber = result['bit_error_rate']
        
        # === RECONSTRUCTION: Create full indices with zero-padding for untransmitted groups ===
        received_indices_full = torch.zeros_like(indices, device=indices.device)
        received_indices_full[:, :group_to_use, :] = received_indices_partial
        # Untransmitted groups remain zero: received_indices_full[:, group_to_use:, :] = 0
        
        # Decode with zero-padded untransmitted groups
        received_features_decoded = decode_fn(received_indices_full)
        
        # For gradient flow: use quantized_features as base and add detached difference
        # This ensures gradients flow through the quantizer but receiver only sees transmitted+zero data
        received_features = quantized_features + (received_features_decoded - quantized_features).detach()
        
        # Calculate transmission MSE between sent and received (only transmitted parts)
        # Extract the parts corresponding to transmitted groups for fair comparison
        features_per_group = received_features_decoded.shape[1] // n_groups
        sent_features_full = decode_fn(indices)  # Decode full indices to get original feature shape
        transmitted_features_sent = sent_features_full[:, :group_to_use * features_per_group]
        transmitted_features_received = received_features_decoded[:, :group_to_use * features_per_group]
        
        transmit_mse = F.mse_loss(transmitted_features_received, transmitted_features_sent, reduction='mean')
        
        return received_features, ber, transmit_mse, sent_features
    
    def _mode4_digital_with_coding(self, quantized_features, indices, psnr_value, group_to_use, decode_fn, use_coding=True):
        """
        Mode 4: Digital transmission with adaptive channel coding.
        
        During pretrain phase (use_coding=False): Uses uncoded transmission through mode4 interface
        During selector/final phases (use_coding=True): Uses polar code encoding/decoding
        
        Only the first group_to_use groups are transmitted. Untransmitted groups
        are zero-padded during reconstruction to ensure receiver only uses transmitted data.
        
        Args:
            quantized_features: Original quantized features (for gradient flow only)
            indices: Quantization indices [batch_size, n_groups, group_size]
            psnr_value: PSNR for transmission
            group_to_use: Number of groups to transmit (1-8)
            decode_fn: Function to decode indices back to features
            use_coding: Whether to use polar coding (True) or uncoded transmission (False)
            
        Returns:
            tuple: (received_features, ber, transmit_mse, zeros_count)
        """
        if self.wireless_interface is None:
            raise RuntimeError("Wireless interface not initialized for digital communication")
        
        batch_size, n_groups, group_size = indices.shape
        
        # === TRANSMISSION: Only transmit the first group_to_use groups ===
        transmit_indices = indices[:, :group_to_use, :]
        
        # Get sent features from only transmitted groups (for loss calculation)
        sent_features = decode_fn(transmit_indices)
        
        # Digital transmission with adaptive coding
        result = self.wireless_interface.transmit_mode4(
            transmit_indices,
            snr_db=psnr_value.item() if hasattr(psnr_value, 'item') else psnr_value,
            bits_per_index=4,
            group_to_use=group_to_use,
            use_coding=use_coding  # Pass coding flag to interface
        )
        
        received_indices_partial = result['received_indices']  # Only group_to_use groups
        ber = result['bit_error_rate']
        
        # === RECONSTRUCTION: Create full indices with zero-padding for untransmitted groups ===
        received_indices_full = torch.zeros_like(indices, device=indices.device)
        received_indices_full[:, :group_to_use, :] = received_indices_partial
        # Untransmitted groups remain zero: received_indices_full[:, group_to_use:, :] = 0
        
        # Decode with zero-padded untransmitted groups
        received_features_decoded = decode_fn(received_indices_full)
        
        # For gradient flow: use quantized_features as base and add detached difference
        # This ensures gradients flow through the quantizer but receiver only sees transmitted+zero data
        received_features = quantized_features + (received_features_decoded - quantized_features).detach()
        
        # Calculate transmission MSE between sent and received (only transmitted parts)
        # Extract the parts corresponding to transmitted groups for fair comparison
        features_per_group = received_features_decoded.shape[1] // n_groups
        sent_features_full = decode_fn(indices)  # Decode full indices to get original feature shape
        transmitted_features_sent = sent_features_full[:, :group_to_use * features_per_group]
        transmitted_features_received = received_features_decoded[:, :group_to_use * features_per_group]
        
        transmit_mse = F.mse_loss(transmitted_features_received, transmitted_features_sent, reduction='mean')
        
        return received_features, ber, transmit_mse, sent_features
    
    def _get_importance_mask(self, importance_weights, threshold=None):
        """
        Generate binary mask based on feature importance.
        
        Args:
            importance_weights: Feature importance scores
            threshold: Threshold for binarization
            
        Returns:
            torch.Tensor: Binary mask (1 for important features, 0 for unimportant)
        """
        if threshold is None:
            threshold = self.threshold
        
        alpha = importance_weights.detach()
        hard_mask = (alpha > threshold).float()
        return hard_mask
    
    def get_transmission_stats(self):
        """
        Get transmission statistics for analysis.
        
        Returns:
            dict: Transmission statistics based on communication mode
        """
        stats = {
            'comm_mode': self.comm_mode,
            'mode_description': self._get_mode_description()
        }
        
        if self.wireless_interface is not None:
            stats.update(self.wireless_interface.get_transmission_stats())
                
        return stats
    
    def _get_mode_description(self):
        """Get description of current communication mode."""
        descriptions = {
            1: "Direct analog transmission with importance masking",
            2: "Quantized analog transmission with Gaussian noise", 
            3: "Digital transmission without channel coding",
            4: "Digital transmission with channel coding"
        }
        return descriptions.get(self.comm_mode, "Unknown mode")
    
    def reset_stats(self):
        """Reset transmission statistics."""
        if self.wireless_interface is not None:
            if hasattr(self.wireless_interface, 'reset_stats'):
                self.wireless_interface.reset_stats()
    
    def save_state(self, filepath):
        """
        Save wireless channel state to file.
        
        Args:
            filepath: Path to save the state
        """
        state = {
            'comm_mode': self.comm_mode,
            'threshold': self.threshold,
        }
        
        if self.wireless_interface is not None:
            state['wireless_interface_stats'] = self.wireless_interface.get_transmission_stats()
            
        torch.save(state, filepath)
        print(f"Wireless channel state saved to {filepath}")
        
    def load_state(self, filepath):
        """
        Load wireless channel state from file.
        
        Args:
            filepath: Path to load the state from
        """
        state = torch.load(filepath, map_location=device)
        
        self.comm_mode = state.get('comm_mode', self.comm_mode)
        self.threshold = state.get('threshold', self.threshold)
        
        print(f"Wireless channel state loaded from {filepath}")
