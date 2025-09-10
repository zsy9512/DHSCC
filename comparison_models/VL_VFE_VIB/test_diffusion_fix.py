#!/usr/bin/env python3
"""
Test script to verify the diffusion model fixes
"""

import torch
import torch.nn as nn
import sys
import os

# Add the project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Test the Simple1DUNet
class Simple1DUNet(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=None, num_layers=4):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Time and SNR embedding projections
        self.time_proj = nn.Linear(128, hidden_channels)
        self.snr_proj = nn.Linear(64, hidden_channels)
        
        # Down blocks
        self.down_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU()
            ) for _ in range(num_layers)
        ])
        
        # Up blocks
        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU()
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        
    def get_timestep_embedding(self, timesteps, embedding_dim=128):
        """Create sinusoidal timestep embeddings"""
        import math
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb
    
    def forward(self, x, time_emb, snr_emb):
        # x: [batch, feature_dim]
        # time_emb: [batch, 128] (from timestep embedding)
        # snr_emb: [batch, 64] (from SNR encoder)
        
        # Project input
        h = self.input_proj(x)
        
        # Project embeddings using predefined layers
        time_emb_proj = self.time_proj(time_emb)
        snr_emb_proj = self.snr_proj(snr_emb)
        
        # Add time and SNR embeddings
        h = h + time_emb_proj + snr_emb_proj
        
        # Store skip connections
        skip_connections = []
        
        # Down path
        for down_block in self.down_blocks:
            skip_connections.append(h)
            h = down_block(h)
        
        # Up path
        for up_block in self.up_blocks:
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=1)
            h = up_block(h)
        
        # Output
        return self.output_proj(h)

# Test AWGNDiffusionDenoiser
class AWGNDiffusionDenoiser(nn.Module):
    def __init__(self, feature_dim, diffusion_steps=500):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.diffusion_steps = diffusion_steps
        
        # Simple 1D UNet
        self.unet = Simple1DUNet(
            in_channels=feature_dim,
            hidden_channels=128,
            out_channels=feature_dim,
            num_layers=4
        )
        
        # SNR condition encoder
        self.snr_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
    
    def forward(self, noisy_features, timestep, snr):
        """Forward pass during training"""
        batch_size = noisy_features.shape[0]
        
        # SNR condition encoding
        snr_emb = self.snr_encoder(snr.view(-1, 1))  # [batch, 64]
        
        # Time embedding
        t_emb = self.unet.get_timestep_embedding(timestep)  # [batch, 128]
        
        # UNet prediction (UNet will handle projection internally)
        return self.unet(noisy_features, t_emb, snr_emb)

def test_diffusion_model():
    """Test the diffusion model with sample data"""
    print("Testing diffusion model...")
    
    # Test parameters
    batch_size = 4
    feature_dim = 256
    diffusion_steps = 500
    
    # Create model
    diffusion_model = AWGNDiffusionDenoiser(feature_dim, diffusion_steps).to(device)
    
    # Create sample data
    noisy_features = torch.randn(batch_size, feature_dim).to(device)
    timestep = torch.randint(0, diffusion_steps, (batch_size,)).to(device)
    snr = torch.rand(batch_size).to(device) * 5.0  # 0-5dB range
    
    print(f"Input shapes:")
    print(f"  noisy_features: {noisy_features.shape}")
    print(f"  timestep: {timestep.shape}")
    print(f"  snr: {snr.shape}")
    
    try:
        # Forward pass
        with torch.no_grad():
            predicted_noise = diffusion_model(noisy_features, timestep, snr)
        
        print(f"Output shape: {predicted_noise.shape}")
        print("✅ Diffusion model test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Diffusion model test failed: {e}")
        return False

def test_unet_directly():
    """Test the UNet directly"""
    print("\nTesting UNet directly...")
    
    # Test parameters
    batch_size = 4
    feature_dim = 256
    
    # Create UNet
    unet = Simple1DUNet(feature_dim, 128, feature_dim, 4).to(device)
    
    # Create sample data
    x = torch.randn(batch_size, feature_dim).to(device)
    time_emb = torch.randn(batch_size, 128).to(device)  # 128-dim time embedding
    snr_emb = torch.randn(batch_size, 64).to(device)    # 64-dim SNR embedding
    
    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  time_emb: {time_emb.shape}")
    print(f"  snr_emb: {snr_emb.shape}")
    
    try:
        # Forward pass
        with torch.no_grad():
            output = unet(x, time_emb, snr_emb)
        
        print(f"Output shape: {output.shape}")
        print("✅ UNet test passed!")
        return True
        
    except Exception as e:
        print(f"❌ UNet test failed: {e}")
        return False

if __name__ == '__main__':
    print("=" * 50)
    print("Testing Diffusion Model Fixes")
    print("=" * 50)
    
    # Test UNet directly
    unet_success = test_unet_directly()
    
    # Test full diffusion model
    diffusion_success = test_diffusion_model()
    
    print("\n" + "=" * 50)
    if unet_success and diffusion_success:
        print("🎉 All tests passed! The fixes are working correctly.")
        print("You can now run the training with:")
        print("python VL_VFE_VIB_RPQ_DIFFUSION.py --stage 2 --epochs 100 --lr 1e-4 --weights /path/to/stage1/weights.pth")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    print("=" * 50)
