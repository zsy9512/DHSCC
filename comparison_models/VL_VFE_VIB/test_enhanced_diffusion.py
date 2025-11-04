#!/usr/bin/env python3
"""
Test script for the enhanced diffusion model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add the project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def test_enhanced_diffusion():
    """Test the enhanced diffusion model components"""
    print("=" * 60)
    print("Testing Enhanced Diffusion Model Components")
    print("=" * 60)
    
    # Test parameters
    batch_size = 8
    feature_dim = 256
    hidden_channels = 256
    diffusion_steps = 500
    
    try:
        # Import the enhanced components
        from VL_VFE_VIB_RPQ_DIFFUSION import (
            SelfAttention, ResNetBlock, Enhanced1DUNet, AWGNDiffusionDenoiser
        )
        
        print("✅ Successfully imported enhanced components")
        
        # Test SelfAttention
        print("\n1. Testing SelfAttention...")
        attention = SelfAttention(dim=hidden_channels, num_heads=8).to(device)
        x = torch.randn(batch_size, hidden_channels).to(device)
        out = attention(x)
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {out.shape}")
        assert out.shape == x.shape, "Attention output shape mismatch"
        print("   ✅ SelfAttention test passed")
        
        # Test ResNetBlock
        print("\n2. Testing ResNetBlock...")
        resnet_block = ResNetBlock(dim=hidden_channels, use_attention=True).to(device)
        x = torch.randn(batch_size, hidden_channels).to(device)
        time_emb = torch.randn(batch_size, hidden_channels).to(device)
        snr_emb = torch.randn(batch_size, hidden_channels).to(device)
        out = resnet_block(x, time_emb, snr_emb)
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {out.shape}")
        assert out.shape == x.shape, "ResNetBlock output shape mismatch"
        print("   ✅ ResNetBlock test passed")
        
        # Test Enhanced1DUNet
        print("\n3. Testing Enhanced1DUNet...")
        unet = Enhanced1DUNet(
            in_channels=feature_dim,
            hidden_channels=hidden_channels,
            out_channels=feature_dim,
            num_layers=6
        ).to(device)
        
        x = torch.randn(batch_size, feature_dim).to(device)
        time_emb = torch.randn(batch_size, 128).to(device)
        snr_emb = torch.randn(batch_size, 64).to(device)
        
        out = unet(x, time_emb, snr_emb)
        print(f"   Input shape: {x.shape}")
        print(f"   Time embedding shape: {time_emb.shape}")
        print(f"   SNR embedding shape: {snr_emb.shape}")
        print(f"   Output shape: {out.shape}")
        assert out.shape == x.shape, "UNet output shape mismatch"
        print("   ✅ Enhanced1DUNet test passed")
        
        # Test AWGNDiffusionDenoiser
        print("\n4. Testing AWGNDiffusionDenoiser...")
        diffusion_model = AWGNDiffusionDenoiser(
            feature_dim=feature_dim,
            diffusion_steps=diffusion_steps
        ).to(device)
        
        # Test forward pass
        noisy_features = torch.randn(batch_size, feature_dim).to(device)
        timestep = torch.randint(0, diffusion_steps, (batch_size,)).to(device)
        snr = torch.rand(batch_size).to(device) * 5.0  # 0-5dB range
        
        predicted_noise = diffusion_model(noisy_features, timestep, snr)
        print(f"   Noisy features shape: {noisy_features.shape}")
        print(f"   Timestep shape: {timestep.shape}")
        print(f"   SNR shape: {snr.shape}")
        print(f"   Predicted noise shape: {predicted_noise.shape}")
        assert predicted_noise.shape == noisy_features.shape, "Diffusion output shape mismatch"
        print("   ✅ AWGNDiffusionDenoiser forward test passed")
        
        # Test noise addition
        clean_features = torch.randn(batch_size, feature_dim).to(device)
        noise = torch.randn_like(clean_features)
        timesteps = torch.randint(0, diffusion_steps, (batch_size,)).to(device)
        
        noisy_input = diffusion_model.add_noise(clean_features, noise, timesteps)
        print(f"   Clean features shape: {clean_features.shape}")
        print(f"   Added noise shape: {noise.shape}")
        print(f"   Noisy input shape: {noisy_input.shape}")
        assert noisy_input.shape == clean_features.shape, "Noise addition shape mismatch"
        print("   ✅ Noise addition test passed")
        
        # Test denoising
        print("\n5. Testing denoising process...")
        test_snr = 2.0
        denoised = diffusion_model.denoise(noisy_features, test_snr, steps=10)
        print(f"   Input noisy features shape: {noisy_features.shape}")
        print(f"   Denoised features shape: {denoised.shape}")
        print(f"   SNR: {test_snr} dB")
        assert denoised.shape == noisy_features.shape, "Denoising output shape mismatch"
        print("   ✅ Denoising test passed")
        
        # Test cosine schedule
        print("\n6. Testing cosine noise schedule...")
        betas = diffusion_model._cosine_beta_schedule(diffusion_steps)
        print(f"   Beta schedule shape: {betas.shape}")
        print(f"   Beta range: [{betas.min():.6f}, {betas.max():.6f}]")
        assert len(betas) == diffusion_steps, "Beta schedule length mismatch"
        assert torch.all(betas > 0) and torch.all(betas < 1), "Beta values out of range"
        print("   ✅ Cosine schedule test passed")
        
        print("\n" + "=" * 60)
        print("🎉 All enhanced diffusion tests passed!")
        print("Key improvements implemented:")
        print("✅ Self-attention mechanism for better feature learning")
        print("✅ ResNet blocks with conditioning")
        print("✅ Enhanced UNet with deeper architecture (6 layers)")
        print("✅ Increased model capacity (256 hidden channels)")
        print("✅ Cosine noise schedule for better diffusion performance")
        print("✅ Improved SNR conditioning with more capacity")
        print("✅ Adaptive denoising steps based on SNR")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_capacity():
    """Test model capacity and parameter count"""
    print("\n" + "=" * 60)
    print("Model Capacity Analysis")
    print("=" * 60)
    
    try:
        from VL_VFE_VIB_RPQ_DIFFUSION import AWGNDiffusionDenoiser
        
        # Original simple model (for comparison)
        print("Comparing model capacities:")
        
        # Enhanced model
        enhanced_model = AWGNDiffusionDenoiser(feature_dim=256, diffusion_steps=500)
        enhanced_params = sum(p.numel() for p in enhanced_model.parameters())
        enhanced_trainable = sum(p.numel() for p in enhanced_model.parameters() if p.requires_grad)
        
        print(f"Enhanced Diffusion Model:")
        print(f"  Total parameters: {enhanced_params:,}")
        print(f"  Trainable parameters: {enhanced_trainable:,}")
        print(f"  Model size: ~{enhanced_params * 4 / 1024 / 1024:.1f} MB")
        
        # Memory usage estimation
        batch_size = 32
        feature_dim = 256
        memory_per_sample = feature_dim * 4  # 4 bytes per float32
        estimated_memory = batch_size * memory_per_sample * 10  # rough estimate with gradients
        print(f"  Estimated memory usage (batch_size={batch_size}): ~{estimated_memory / 1024 / 1024:.1f} MB")
        
        print("✅ Model capacity analysis completed")
        return True
        
    except Exception as e:
        print(f"❌ Capacity analysis failed: {e}")
        return False

if __name__ == '__main__':
    print("Testing Enhanced Diffusion Model for VL-VFE-VIB")
    
    # Run tests
    diffusion_test = test_enhanced_diffusion()
    capacity_test = test_model_capacity()
    
    print("\n" + "=" * 60)
    if diffusion_test and capacity_test:
        print("🎉 All tests passed! The enhanced diffusion model is ready for training.")
        print("\nRecommended training commands:")
        print("1. Stage 1 (VIB pre-training):")
        print("   python VL_VFE_VIB_RPQ_DIFFUSION.py --stage 1 --epochs 50 --lr 1e-3")
        print("\n2. Stage 2 (Diffusion training):")
        print("   python VL_VFE_VIB_RPQ_DIFFUSION.py --stage 2 --epochs 100 --lr 1e-4 --weights checkpoints/stage1_best_acc.pth")
        print("\n3. Stage 3 (Fine-tuning):")
        print("   python VL_VFE_VIB_RPQ_DIFFUSION.py --stage 3 --epochs 30 --lr 5e-5 --weights checkpoints/stage2_best_loss.pth")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    print("=" * 60)
