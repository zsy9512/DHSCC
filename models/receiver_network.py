"""
Receiver Network Module for Vibration Signal Classification

This module contains the receiver-side components including:
- Noise-adaptive decoding with attention mechanism
- Feature reconstruction
- Final classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReceiverNoiseAttention(nn.Module):
    """
    Noise-condition attention module for adaptive feature processing at receiver.
    
    This module implements attention mechanism that adjusts feature importance
    based on the current noise level, allowing the decoder to adaptively
    process features under different channel conditions.
    
    Args:
        hidden_dim (int): Hidden layer dimension
        noise_feat_dim (int): Noise feature dimension
    """
    def __init__(self, hidden_dim, noise_feat_dim=16):
        super(ReceiverNoiseAttention, self).__init__()
        
        # Noise encoder - maps scalar noise to feature representation
        self.noise_encoder = nn.Sequential(
            nn.Linear(1, noise_feat_dim),
            nn.ReLU(),
            nn.Linear(noise_feat_dim, noise_feat_dim),
            nn.ReLU(),
            nn.Linear(noise_feat_dim, noise_feat_dim),
            nn.ReLU()
        )
        
        # Attention generator - generates attention weights from noise features
        self.attention_generator = nn.Sequential(
            nn.Linear(noise_feat_dim, noise_feat_dim),
            nn.Tanh(),
            nn.Linear(noise_feat_dim, hidden_dim),
            nn.Sigmoid()  # Ensure weights are in [0,1] range
        )
        
    def forward(self, x, noise_level):
        """
        Forward propagation.
        
        Args:
            x: Input features [batch_size, hidden_dim]
            noise_level: Noise level [batch_size, 1]
            
        Returns:
            Features adjusted according to noise conditions [batch_size, hidden_dim]
        """
        # Encode noise
        noise_feat = self.noise_encoder(noise_level)
        
        # Generate attention weights
        noise_attention = self.attention_generator(noise_feat)
        
        # Apply attention weights
        return x * noise_attention

class ReceiverAdaptiveDecoder(nn.Module):
    """
    Noise-adaptive decoder with attention mechanism for vibration signal features.
    
    This module takes encoded features and channel noise information as input,
    and uses noise-conditioned attention to adaptively process features.
    
    Args:
        hidden_dim (int): Dimension of hidden features
        output_flat_dim (int): Flattened output dimension for reshaping
        output_channels (int): Number of output channels after reshaping
    """
    def __init__(self, hidden_dim, output_flat_dim, output_channels=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_flat_dim = output_flat_dim
        self.output_channels = output_channels
        
        # Initial feature processing
        self.feature_decoder = nn.Linear(hidden_dim, hidden_dim)
        
        # Noise condition attention - core improvement
        self.noise_condition_attention = ReceiverNoiseAttention(hidden_dim)
        
        # Feature refinement
        self.feature_refinement = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Reshape layer for convolutional processing
        self.reshape_layer = nn.Linear(hidden_dim, output_flat_dim)
        
    def forward(self, x, noise_level):
        """
        Forward pass through the noise-adaptive decoder.
        
        Args:
            x: Input features [batch_size, hidden_dim]
            noise_level: Scalar noise level [batch_size, 1] or [1]
            
        Returns:
            Reshaped features ready for convolution
        """
        batch_size = x.size(0)
        
        # Initial feature processing
        x = F.relu(self.feature_decoder(x))
        
        # Apply noise condition attention - adaptive feature weighting
        x = self.noise_condition_attention(x, noise_level)
        
        # Further feature refinement
        x = self.feature_refinement(x)
        
        # Reshape for convolutional processing
        x_reshaped = self.reshape_layer(x)
        
        return x_reshaped

class ReceiverFeatureReconstructor(nn.Module):
    """
    Module for reconstructing high-dimensional features from decoder output.
    
    Args:
        in_channels (int): Input channel dimension
        out_channels (int): Output channel dimension
    """
    def __init__(self, in_channels=8, out_channels=512):
        super().__init__()
        
        # Convolutional reconstruction network
        self.conv_reconstruction = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x, target_shape=None):
        """
        Reconstruct high-dimensional features.
        
        Args:
            x: Input tensor [batch, in_channels, seq_len]
            target_shape: Optional target shape for adaptive pooling
            
        Returns:
            Reconstructed features with shape [batch, out_channels, seq_len]
        """
        # Convolutional reconstruction
        features = self.conv_reconstruction(x)
        
        # Ensure output has the correct shape if specified
        if target_shape is not None and features.size() != target_shape:
            features = F.adaptive_avg_pool1d(features, target_shape[2])
            
        return features

class VibrationClassifier(nn.Module):
    """
    CNN-based classifier for vibration signals.
    
    Implements an efficient CNN-based model for processing sequential data,
    optimized for vibration signal classification tasks.
    
    Args:
        input_dim: Input feature dimension
        seq_len: Sequence length
        hidden_dim: Hidden dimension for processing
        num_classes: Number of output classes
    """
    def __init__(self, input_dim, seq_len, hidden_dim, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Sequential model for classification
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Output projection
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # Input shape: [batch_size, channels, seq_len]
        x = self.feature_extractor(x)
        x = x.squeeze(-1)  # Remove the sequence dimension
        output = self.classifier(x)
        return output

class ReceiverClassifierModule(nn.Module):
    """
    Classification module for vibration signals at receiver side.
    
    This module handles the final processing and classification of reconstructed features.
    
    Args:
        input_dim (int): Input feature dimension
        seq_len (int): Target sequence length after pooling
        hidden_dim (int): Hidden dimension for classifier
        num_classes (int): Number of output classes
    """
    def __init__(self, input_dim=512, seq_len=8, hidden_dim=256, num_classes=10):
        super().__init__()
        
        # Fixed pooling to standardize sequence length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(seq_len)
        
        # Vibration signal classifier
        self.vibration_classifier = VibrationClassifier(
            input_dim=input_dim,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )
        
    def forward(self, x):
        """
        Forward pass through the classifier module.
        
        Args:
            x: Input features [batch, channels, variable_seq_len]
            
        Returns:
            Classification scores [batch, num_classes]
        """
        # Standardize sequence length
        x = self.adaptive_pool(x)
        
        # Classify using vibration classifier
        output = self.vibration_classifier(x)
        
        return output

class ReceiverNetwork(nn.Module):
    """
    Complete receiver network integrating all receiver-side components.
    
    This module combines adaptive decoding, feature reconstruction, and classification
    to process received features and produce final predictions.
    
    Args:
        args: Configuration parameters containing model settings
        sender_feature_extractor: Reference to sender's feature extractor for residual connections
    """
    def __init__(self, args, sender_feature_extractor=None):
        super().__init__()
        
        self.hidden_channel = args.intermediate_dim
        self.num_classes = getattr(args, 'num_classes', 10)
        
        # Calculate flat dimension for decoder output
        flat_dim = 8 * 16  # This should match sender encoder's flat_dim
        
        # Noise-adaptive decoder
        self.adaptive_decoder = ReceiverAdaptiveDecoder(
            hidden_dim=self.hidden_channel,
            output_flat_dim=flat_dim,
            output_channels=8
        )
        
        # Feature reconstruction module
        self.feature_reconstructor = ReceiverFeatureReconstructor(
            in_channels=8,
            out_channels=512
        )
        
        # Classification module
        self.classifier_module = ReceiverClassifierModule(
            input_dim=512,
            seq_len=8,
            hidden_dim=256,
            num_classes=self.num_classes
        )
        
        # Store reference to sender's feature extractor for residual connections
        self.sender_feature_extractor = sender_feature_extractor
        
    def forward(self, received_features, channel_noise, target_shape):
        """
        Forward pass through the receiver network.
        
        Args:
            received_features: Features received from wireless channel [batch_size, hidden_dim]
            channel_noise: Channel noise level [batch_size, 1]
            target_shape: Target shape for feature reconstruction
            
        Returns:
            Classification output [batch_size, num_classes]
        """
        batch_size = received_features.size(0)
        
        # Adaptive decoding with noise conditioning
        #decoded_flat = self.adaptive_decoder(received_features, channel_noise)这个trick没用
        decoded_flat=received_features
        # Reshape for convolutional processing
        decoded_reshaped = decoded_flat.view(batch_size, 8, 16)
        #print(received_features.shape,decoded_reshaped.shape,decoded_flat.shape)
        # Reconstruct high-dimensional features
        reconstructed_features = self.feature_reconstructor(decoded_reshaped, target_shape)
        
        # Apply residual connection if sender extractor is available
        if self.sender_feature_extractor is not None:
            reconstructed_features = self.sender_feature_extractor.apply_residual(reconstructed_features)
        
        # Final classification
        output = self.classifier_module(reconstructed_features)
        
        return output
    
    def save_state(self, filepath):
        """
        Save receiver network state to file.
        
        Args:
            filepath: Path to save the model state
        """
        state = {
            'adaptive_decoder': self.adaptive_decoder.state_dict(),
            'feature_reconstructor': self.feature_reconstructor.state_dict(),
            'classifier_module': self.classifier_module.state_dict(),
        }
        torch.save(state, filepath)
        print(f"Receiver network state saved to {filepath}")
        
    def load_state(self, filepath):
        """
        Load receiver network state from file.
        
        Args:
            filepath: Path to load the model state from
        """
        state = torch.load(filepath, map_location=device)
        self.adaptive_decoder.load_state_dict(state['adaptive_decoder'])
        self.feature_reconstructor.load_state_dict(state['feature_reconstructor'])
        self.classifier_module.load_state_dict(state['classifier_module'])
        print(f"Receiver network state loaded from {filepath}")
