"""
DTJ-VIB Network Architecture

VIB-specialized network architecture adapted from DTJ for processing 2x16x16 pseudo-images
derived from 512-sample vibration signals.

Key adaptations from original DTJ:
1. Input channels: 3 → 2 (for pseudo-image channels)
2. Input size: 32x32 → 16x16 (vibration pseudo-image size)
3. Feature dimensions: Optimized for smaller input
4. Output classes: Configurable for VIB classification (default 10)
"""
import os
import sys
import torch.nn as nn
from .modules import Resblock, RPQSampler,MaskAttentionSampler

print("✓ DHSCC无线通信模块导入成功")

class DTJSCC_VIB(nn.Module):
    """
    DTJ network specialized for vibration classification.
    
    Processes 2x16x16 pseudo-images derived from 512-sample vibration signals
    through joint source-channel coding for task-oriented communication.
    
    Architecture:
    Input: [2, 16, 16] pseudo-image
    Encoder: CNN feature extraction
    Sampler: Vector quantization with channel effects
    Decoder: Classification head
    Output: Class probabilities
    """
    
    def __init__(self, in_channels=2, latent_channels=256, out_classes=10, num_embeddings=16, snr=8.0):
        """
        Initialize DTJ-VIB network.
        
        Args:
            in_channels: Number of input channels (2 for pseudo-image)
            latent_channels: Latent feature dimension
            out_classes: Number of output classes
            num_embeddings: Size of quantization codebook
        """
        super().__init__()
        self.latent_d = latent_channels
        self.in_channels = in_channels
        self.out_classes = out_classes
        
        # Initial convolution - adapted for 2-channel input
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels//8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels//8),
            nn.ReLU()
        )
        
        # First layer: 16x16 → 8x8
        self.layer1 = nn.Sequential(
            nn.Conv2d(latent_channels//8, latent_channels//4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels//4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        
        # Second layer: 8x8 → 4x4
        self.layer2 = nn.Sequential(
            nn.Conv2d(latent_channels//4, latent_channels//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels//2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Third layer: 4x4 → 2x2
        self.layer3 = nn.Sequential(
            nn.Conv2d(latent_channels//2, latent_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        
        # Complete encoder
        self.encoder = nn.Sequential(
            self.prep,                          # 32x16x16  (latent_channels//8 = 32)
            self.layer1,                        # 64x8x8    (latent_channels//4 = 64)
            Resblock(latent_channels//4),       # 64x8x8
            self.layer2,                        # 128x8x8   (latent_channels//2 = 128)
            self.layer3,                        # 256x8x8   (latent_channels = 256)
            Resblock(latent_channels),          # 256x8x8
            Resblock(latent_channels)           # 256x8x8
        )
        # Vector quantizer with attention mechanism
        self.sampler = MaskAttentionSampler(latent_channels, num_embeddings, snr=snr)
        self.RPQsampler =RPQSampler(latent_channels,num_embeddings, snr=snr)
        # Classification decoder
        self.decoder = nn.Sequential(
            Resblock(latent_channels),          # 256x2x2
            Resblock(latent_channels),          # 256x2x2
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),            # 256x1x1 (use average pooling for better features)
            nn.Flatten(),                       # 256
            nn.Dropout(0.1),                    # Add light dropout for regularization
            nn.Linear(latent_channels, out_classes)
        )
        
    def encode(self, X):
        """
        Encode input pseudo-images to latent features.
        
        Args:
            X: Input tensor of shape [batch_size, 2, 16, 16]
            
        Returns:
            Tuple of (encoded_features, original_shape)
        """
        en_X = self.encoder(X)
        former_shape = en_X.shape
        #print(f"Encoded shape: {former_shape}")
        # Reshape for vector quantization: [batch_size, height*width, channels]
        en_X = en_X.permute(0, 2, 3, 1).contiguous().view(-1, self.latent_d)
        return en_X, former_shape
    
    def decode(self, features, former_shape):
        """
        Decode quantized features to classification output.
        
        Args:
            features: Quantized features
            former_shape: Original feature map shape
            
        Returns:
            Classification logits
        """
        b, c, h, w = former_shape
        # Reshape back to feature maps
        features = features.view(b, h, w, c)
        features = features.permute(0, 3, 1, 2).contiguous()
        tilde_X = self.decoder(features)
        return tilde_X
        
    def forward(self, X, mod=None):
        """
        Forward pass through the complete DTJ-VIB network.
        
        Args:
            X: Input pseudo-images [batch_size, 2, 16, 16]
            mod: Modulation/demodulation module for channel simulation
            
        Returns:
            Tuple of (classification_output, quantization_distribution)
        """
        # Encode to latent features
        out, former_shape = self.encode(X)
        # Vector quantization with channel effects
        #out, dist,None = self.sampler(out, mod=mod,former_shape=former_shape)
        out, dist,commit_loss = self.RPQsampler(out, mod=mod,former_shape=former_shape)
        # Decode to classification output
        tilde_X = self.decode(out, former_shape)
        
        return tilde_X, dist,commit_loss


class DTJSCC_VIB_Small(nn.Module):
    """
    A smaller variant of DTJ-VIB for faster training or resource-constrained scenarios.
    
    This version uses fewer parameters while maintaining the core DTJ functionality.
    """
    
    def __init__(self, in_channels=2, latent_channels=128, out_classes=10, num_embeddings=16):
        """
        Initialize smaller DTJ-VIB network.
        
        Args:
            in_channels: Number of input channels (2 for pseudo-image)
            latent_channels: Latent feature dimension (reduced from 256 to 128)
            out_classes: Number of output classes
            num_embeddings: Size of quantization codebook
        """
        super().__init__()
        self.latent_d = latent_channels
        self.in_channels = in_channels
        self.out_classes = out_classes
        
        # Simplified encoder for smaller model
        self.encoder = nn.Sequential(
            # Initial conv: 2x16x16 → 32x16x16
            nn.Conv2d(in_channels, latent_channels//4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels//4),
            nn.ReLU(),
            
            # First reduction: 32x16x16 → 64x8x8
            nn.Conv2d(latent_channels//4, latent_channels//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels//2),
            nn.ReLU(),
            
            # Second reduction: 64x8x8 → 128x4x4
            nn.Conv2d(latent_channels//2, latent_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(),
            
            # Final reduction: 128x4x4 → 128x2x2
            nn.AdaptiveAvgPool2d(2),
            
            # Residual processing
            Resblock(latent_channels)
        )
        
        # Vector quantizer
        self.sampler = MaskAttentionSampler(latent_channels, num_embeddings)
        
        # Simplified decoder
        self.decoder = nn.Sequential(
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),            # 128x1x1
            nn.Flatten(),                       # 128
            nn.Linear(latent_channels, latent_channels//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_channels//2, out_classes)
        )
        
    def encode(self, X):
        """Encode input to latent features."""
        en_X = self.encoder(X)
        former_shape = en_X.shape
        en_X = en_X.permute(0, 2, 3, 1).contiguous().view(-1, self.latent_d)
        return en_X, former_shape
    
    def decode(self, features, former_shape):
        """Decode quantized features to output."""
        b, c, h, w = former_shape
        features = features.view(b, h, w, c)
        features = features.permute(0, 3, 1, 2).contiguous()
        return self.decoder(features)
        
    def forward(self, X, mod=None):
        """Forward pass through the network."""
        out, former_shape = self.encode(X)
        out, dist = self.sampler(out, mod=mod)
        tilde_X = self.decode(out, former_shape)
        return tilde_X, dist


def create_dtj_vib_model(model_type='standard', **kwargs):
    """
    Factory function to create DTJ-VIB models.
    
    Args:
        model_type: 'standard' or 'small'
        **kwargs: Model parameters
        
    Returns:
        DTJ-VIB model instance
    """
    if model_type == 'standard':
        return DTJSCC_VIB(**kwargs)
    elif model_type == 'small':
        return DTJSCC_VIB_Small(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    """
    Test the DTJ-VIB network architectures.
    """
    import torch
    
    # Test standard model
    print("Testing DTJ-VIB Standard Model...")
    model = DTJSCC_VIB(in_channels=2, latent_channels=256, out_classes=10, num_embeddings=16)
    
    # Create test input (batch of pseudo-images)
    batch_size = 4
    test_input = torch.randn(batch_size, 2, 16, 16)
    
    print(f"Input shape: {test_input.shape}")
    
    # Forward pass without modulation (training mode)
    model.train()
    output, dist = model(test_input, mod=None)
    
    print(f"Output shape: {output.shape}")
    print(f"Distribution shape: {dist.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test small model
    print("\nTesting DTJ-VIB Small Model...")
    small_model = DTJSCC_VIB_Small(in_channels=2, latent_channels=128, out_classes=10, num_embeddings=16)
    
    small_output, small_dist = small_model(test_input, mod=None)
    
    print(f"Small output shape: {small_output.shape}")
    print(f"Small distribution shape: {small_dist.shape}")
    print(f"Small model parameters: {sum(p.numel() for p in small_model.parameters()):,}")
    
    print("\nDTJ-VIB model tests completed successfully!")
