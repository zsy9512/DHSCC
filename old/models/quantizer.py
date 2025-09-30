"""
Quantizer Module for Vibration Signal Classification

This module contains the vector quantization components:
- Wrapped GroupVectorQuantizer optimized for vibration signals
- Model state management for independent deployment
"""

import torch
import torch.nn as nn
from models.vq import GroupVectorQuantizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VibrationVectorQuantizer(nn.Module):
    """
    Specialized vector quantizer for vibration signal features.
    
    This module wraps the GroupVectorQuantizer with vibration-specific
    configurations and provides unified interface for quantization operations.
    
    Args:
        dimension (int): Feature dimension for quantization
        n_codes_group (int): Number of quantization groups
        bins (int): Codebook size per group
        n_q (int): Total number of quantizers across all groups
    """
    def __init__(self, dimension=128, n_codes_group=8, bins=16, n_q=32):
        super().__init__()
        
        self.dimension = dimension
        self.n_codes_group = n_codes_group
        self.bins = bins
        self.n_q = n_q
        
        # Initialize the core vector quantizer
        self.gvq = GroupVectorQuantizer(
            dimension=dimension,
            n_codes_group=n_codes_group,
            bins=bins,
            n_q=n_q
        )
        
    def forward(self, features):
        """
        Forward pass through vector quantization.
        
        Args:
            features: Input features to be quantized [batch_size, feature_dim]
            
        Returns:
            tuple: (quantized_features, indices, commitment_loss)
                quantized_features: Quantized feature representations
                indices: Quantization indices for each group
                commitment_loss: VQ commitment loss for training
        """
        return self.gvq(features)
    
    def encode(self, features):
        """
        Encode features to quantization indices.
        
        Args:
            features: Input features [batch_size, feature_dim]
            
        Returns:
            tuple: (quantized_features, indices, commitment_loss)
        """
        return self.forward(features)
    
    def decode(self, indices):
        """
        Decode quantization indices back to features.
        
        Args:
            indices: Quantization indices [batch_size, n_groups, group_size]
            
        Returns:
            torch.Tensor: Reconstructed features from indices
        """
        return self.gvq.decode(indices)
    
    def get_codebook_usage(self):
        """
        Get statistics about codebook usage for analysis.
        
        Returns:
            dict: Codebook usage statistics
        """
        if hasattr(self.gvq, 'get_codebook_usage'):
            return self.gvq.get_codebook_usage()
        else:
            return {"message": "Codebook usage tracking not available"}
    
    def reset_codebook_usage(self):
        """Reset codebook usage statistics."""
        if hasattr(self.gvq, 'reset_codebook_usage'):
            self.gvq.reset_codebook_usage()
    
    def save_state(self, filepath):
        """
        Save quantizer state to file.
        
        Args:
            filepath: Path to save the quantizer state
        """
        state = {
            'gvq_state': self.gvq.state_dict(),
            'config': {
                'dimension': self.dimension,
                'n_codes_group': self.n_codes_group,
                'bins': self.bins,
                'n_q': self.n_q
            }
        }
        torch.save(state, filepath)
        print(f"Quantizer state saved to {filepath}")
        
    def load_state(self, filepath):
        """
        Load quantizer state from file.
        
        Args:
            filepath: Path to load the quantizer state from
        """
        state = torch.load(filepath, map_location=device)
        
        # Verify configuration compatibility
        config = state.get('config', {})
        if config:
            assert config['dimension'] == self.dimension, f"Dimension mismatch: {config['dimension']} vs {self.dimension}"
            assert config['n_codes_group'] == self.n_codes_group, f"Groups mismatch: {config['n_codes_group']} vs {self.n_codes_group}"
            assert config['bins'] == self.bins, f"Bins mismatch: {config['bins']} vs {self.bins}"
            assert config['n_q'] == self.n_q, f"Quantizers mismatch: {config['n_q']} vs {self.n_q}"
        
        self.gvq.load_state_dict(state['gvq_state'])
        print(f"Quantizer state loaded from {filepath}")
    
    def get_config(self):
        """
        Get quantizer configuration.
        
        Returns:
            dict: Configuration parameters
        """
        return {
            'dimension': self.dimension,
            'n_codes_group': self.n_codes_group,
            'bins': self.bins,
            'n_q': self.n_q
        }
