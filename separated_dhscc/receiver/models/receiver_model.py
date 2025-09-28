"""
Receiver Model for DHSCC - Independent Receiver

This module implements the receiver-side model for independent training
and testing, handling received features and producing classification results.

Key components:
- Receiver Network (adaptive decoding, feature reconstruction, classification)
- Channel decoding handling
- Independent model training and inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .receiver_network import ReceiverNetwork
from ..utils.signal_processing import psnr_to_noise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReceiverModel(nn.Module):
    """
    Integrated receiver model for vibration signal classification.
    
    This model handles the complete receiver-side processing:
    1. Receiving features from channel transmission
    2. Adaptive decoding based on channel conditions
    3. Feature reconstruction
    4. Final classification
    
    Args:
        args: Configuration parameters
        num_classes: Number of output classes
    """
    def __init__(self, args, num_classes=10):
        super().__init__()
        
        # Store configuration
        self.args = args
        self.num_classes = num_classes
        self.comm_mode = getattr(args, 'communication_mode', 1)
        self.hidden_channel = args.intermediate_dim
        
        # SNR parameters for testing
        self.snr_max = getattr(args, 'snr_max', 15.0)
        self.snr_min = getattr(args, 'snr_min', 5.0)
        self.test = getattr(args, 'test', 0)
        
        # Initialize receiver network (without sender reference for independence)
        self.receiver_network = ReceiverNetwork(args, sender_feature_extractor=None)
        
        # Performance tracking
        self.batch_snrs = []
        self.training_metrics = {
            'loss': [],
            'accuracy': [],
            'classification_loss': []
        }
        
        print(f"Initialized ReceiverModel with communication mode {self.comm_mode}")
        
    def forward(self, received_features, snr=10.0, feature_shape=None, epoch=0):
        """
        Forward pass through receiver model.
        
        Args:
            received_features: Features received from channel [batch_size, feature_dim]
            snr: Signal-to-noise ratio (direct input as required)
            feature_shape: Original feature shape information
            epoch: Current training epoch
            
        Returns:
            dict: Dictionary containing:
                - 'classification_output': Final classification logits
                - 'reconstructed_features': Reconstructed features
        """
        batch_size = received_features.size(0)
        
        # Determine SNR/noise level - use direct SNR input as required
        if self.test:
            snr_value = torch.tensor([snr], device=device).float()
        else:
            # During training, use the provided SNR directly
            snr_value = torch.tensor([snr], device=device).float()
        
        # Convert SNR to noise value
        channel_noise = psnr_to_noise(snr_value)
        
        # === RECEIVER PROCESSING ===
        classification_output = self.receiver_network(
            received_features, 
            channel_noise, 
            feature_shape
        )
        
        # Track performance data during training
        if self.training:
            self.batch_snrs.append(snr_value.item())
            
        return {
            'classification_output': classification_output,
            'reconstructed_features': received_features,  # For now, same as input
            'snr_value': snr_value
        }
    
    def classify(self, received_features, snr=10.0, feature_shape=None):
        """
        Perform classification on received features.
        
        Args:
            received_features: Features received from channel
            snr: Signal-to-noise ratio
            feature_shape: Original feature shape information
            
        Returns:
            torch.Tensor: Classification predictions
        """
        with torch.no_grad():
            result = self.forward(received_features, snr=snr, feature_shape=feature_shape)
            return result['classification_output']
    
    def process_channel_output(self, channel_data, snr=10.0):
        """
        Process data received from channel module.
        
        Args:
            channel_data: Data received from channel transmission
            snr: Signal-to-noise ratio
            
        Returns:
            torch.Tensor: Classification results
        """
        if isinstance(channel_data, dict):
            if 'type' in channel_data:
                if channel_data['type'] == 'digital':
                    # For digital modes, features should be reconstructed from indices
                    # This would normally be done by the channel module
                    received_features = channel_data.get('reconstructed_features', 
                                                       channel_data.get('features'))
                    feature_shape = channel_data.get('metadata', {}).get('feature_shape')
                else:
                    # For analog modes
                    received_features = channel_data.get('features')
                    feature_shape = channel_data.get('metadata', {}).get('feature_shape')
            else:
                # Assume direct features
                received_features = channel_data.get('features', channel_data.get('received_features'))
                feature_shape = channel_data.get('feature_shape')
        else:
            # Assume tensor input
            received_features = channel_data
            feature_shape = None
            
        return self.classify(received_features, snr=snr, feature_shape=feature_shape)
    
    def save_model(self, filepath):
        """
        Save receiver model state.
        
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
        
        torch.save(model_state, filepath)
        print(f"Receiver model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load receiver model state.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if 'training_metrics' in checkpoint:
            self.training_metrics = checkpoint['training_metrics']
            
        print(f"Receiver model loaded from {filepath}")
        
    def evaluate_performance(self, test_loader, snr_range=None, device=None):
        """
        Evaluate receiver performance across different SNR levels.
        
        Args:
            test_loader: Test data loader (expecting received features)
            snr_range: Range of SNR values to test
            device: Computing device
            
        Returns:
            dict: Performance metrics
        """
        if device is None:
            device = next(self.parameters()).device
            
        if snr_range is None:
            snr_range = [self.snr_min, (self.snr_min + self.snr_max) / 2, self.snr_max]
            
        self.eval()
        results = {}
        
        for snr in snr_range:
            correct = 0
            total = 0
            total_loss = 0.0
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(test_loader):
                    # Assuming data contains received features
                    data, target = data.to(device), target.to(device)
                    
                    result = self.forward(data, snr=snr)
                    output = result['classification_output']
                    
                    # Calculate loss
                    loss = F.cross_entropy(output, target)
                    total_loss += loss.item()
                    
                    # Calculate accuracy
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
            
            accuracy = correct / total
            avg_loss = total_loss / len(test_loader)
            
            results[f'snr_{snr}'] = {
                'accuracy': accuracy,
                'loss': avg_loss,
                'correct': correct,
                'total': total
            }
            
            print(f"SNR {snr}dB: Accuracy = {accuracy:.4f}, Loss = {avg_loss:.4f}")
        
        return results