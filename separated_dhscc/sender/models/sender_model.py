"""
Sender Model for DHSCC - Integrated Sender with Quantizer

This module integrates the sender network and quantizer into a single
training unit as required for the separated architecture.

Key components:
- Sender Network (feature extraction, encoding, importance estimation)
- Quantizer (vector quantization for digital transmission)
- Channel encoding preparation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .sender_network import SenderNetwork
from .quantizer import VibrationVectorQuantizer
from ..utils.signal_processing import psnr_to_noise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SenderModel(nn.Module):
    """
    Integrated sender model combining feature extraction and quantization.
    
    This model handles the complete sender-side processing:
    1. Feature extraction from input vibration signals
    2. Feature encoding and importance estimation
    3. Vector quantization for digital transmission modes
    4. Preparation of features for channel transmission
    
    Args:
        args: Configuration parameters
        num_classes: Number of output classes (for pre-task classification)
    """
    def __init__(self, args, num_classes=10):
        super().__init__()
        
        # Store configuration
        self.args = args
        self.num_classes = num_classes
        self.comm_mode = getattr(args, 'communication_mode', 1)
        self.hidden_channel = args.intermediate_dim
        
        # SNR parameters for training
        self.snr_max = getattr(args, 'snr_max', 15.0)
        self.snr_min = getattr(args, 'snr_min', 5.0)
        self.test = getattr(args, 'test', 0)
        
        # Initialize sender network
        self.sender_network = SenderNetwork(args)
        
        # Initialize quantizer for digital communication modes
        if self.comm_mode in [2, 3, 4]:
            self.quantizer = VibrationVectorQuantizer(
                dimension=128,
                n_codes_group=8,
                bins=16,
                n_q=4*8
            )
            print(f"Initialized quantizer for communication mode {self.comm_mode}")
        else:
            self.quantizer = None
            print(f"No quantizer needed for communication mode {self.comm_mode}")
            
        # Performance tracking
        self.batch_snrs = []
        self.training_metrics = {
            'loss': [],
            'classification_loss': [],
            'commitment_loss': [],
            'transmit_loss': []
        }
        
        print(f"Initialized SenderModel with communication mode {self.comm_mode}")
        
    def forward(self, x, snr=10.0, epoch=0):
        """
        Forward pass through sender model.
        
        Args:
            x: Input vibration signal [batch_size, channels, time]
            snr: Signal-to-noise ratio for training (direct input as required)
            epoch: Current training epoch
            
        Returns:
            dict: Dictionary containing:
                - 'encoded_features': Processed features ready for transmission
                - 'quantized_features': Quantized features (if applicable)
                - 'indices': Quantization indices (if applicable)
                - 'commitment_loss': Quantization commitment loss
                - 'importance_weights': Feature importance scores
                - 'pre_task_logits': Pre-classification results
                - 'feature_shape': Shape information for reconstruction
        """
        batch_size = x.size(0)
        
        # Determine SNR/noise level - use direct SNR input as required
        if self.test:
            snr_value = torch.tensor([snr], device=device).float()
        else:
            # During training, use the provided SNR directly
            snr_value = torch.tensor([snr], device=device).float()
        
        # Convert SNR to noise value
        channel_noise = psnr_to_noise(snr_value)
        
        # === SENDER PROCESSING ===
        encoded_features, importance_weights, pre_task_logits, feature_shape, original_encoded_features = self.sender_network(
            x, channel_noise
        )
        
        # Initialize default values
        commit_loss = torch.tensor(0.0).to(device)
        quantized_features = encoded_features
        indices = None
        
        # === QUANTIZATION (for modes 2, 3, 4) ===
        if self.comm_mode in [2, 3, 4] and self.quantizer is not None:
            # Perform vector quantization
            quantization_result = self.quantizer(encoded_features)
            quantized_features = quantization_result.quantized
            indices = quantization_result.indices
            commit_loss = quantization_result.commit_loss
            
        # Track performance data during training
        if self.training:
            self.batch_snrs.append(snr_value.item())
            
        return {
            'encoded_features': encoded_features,
            'quantized_features': quantized_features,
            'indices': indices,
            'commitment_loss': commit_loss,
            'importance_weights': importance_weights,
            'pre_task_logits': pre_task_logits,
            'feature_shape': feature_shape,
            'original_encoded_features': original_encoded_features,
            'snr_value': snr_value
        }
    
    def get_encoded_features(self, x, snr=10.0):
        """
        Get encoded features ready for transmission.
        
        Args:
            x: Input vibration signal
            snr: Signal-to-noise ratio
            
        Returns:
            tuple: (features_for_transmission, metadata)
        """
        with torch.no_grad():
            result = self.forward(x, snr=snr)
            
            # Return appropriate features based on communication mode
            if self.comm_mode in [2, 3, 4]:
                features = result['quantized_features']
                metadata = {
                    'indices': result['indices'],
                    'feature_shape': result['feature_shape'],
                    'importance_weights': result['importance_weights']
                }
            else:
                features = result['encoded_features']
                metadata = {
                    'feature_shape': result['feature_shape'],
                    'importance_weights': result['importance_weights']
                }
                
            return features, metadata
    
    def save_model(self, filepath):
        """
        Save sender model state.
        
        Args:
            filepath: Path to save the model
        """
        model_state = {
            'model_state_dict': self.state_dict(),
            'config': {
                'comm_mode': self.comm_mode,
                'hidden_channel': self.hidden_channel,
                'num_classes': self.num_classes,
                'snr_min': self.snr_min,
                'snr_max': self.snr_max
            },
            'training_metrics': self.training_metrics
        }
        
        # Save quantizer state if available
        if self.quantizer is not None:
            model_state['quantizer_state'] = self.quantizer.state_dict()
            
        torch.save(model_state, filepath)
        print(f"Sender model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load sender model state.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if 'quantizer_state' in checkpoint and self.quantizer is not None:
            self.quantizer.load_state_dict(checkpoint['quantizer_state'])
            
        if 'training_metrics' in checkpoint:
            self.training_metrics = checkpoint['training_metrics']
            
        print(f"Sender model loaded from {filepath}")
        
    def get_transmission_data(self, x, snr=10.0, group_to_use=8):
        """
        Prepare data for transmission through channel.
        
        Args:
            x: Input vibration signal
            snr: Signal-to-noise ratio
            group_to_use: Number of groups to use for transmission
            
        Returns:
            dict: Data ready for channel transmission
        """
        result = self.forward(x, snr=snr)
        
        if self.comm_mode in [2, 3, 4]:
            # For digital modes, return quantization indices
            return {
                'type': 'digital',
                'indices': result['indices'],
                'quantized_features': result['quantized_features'],
                'group_to_use': group_to_use,
                'metadata': {
                    'feature_shape': result['feature_shape'],
                    'importance_weights': result['importance_weights']
                }
            }
        else:
            # For analog modes, return encoded features
            return {
                'type': 'analog',
                'features': result['encoded_features'],
                'metadata': {
                    'feature_shape': result['feature_shape'],
                    'importance_weights': result['importance_weights']
                }
            }