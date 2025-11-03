#!/usr/bin/env python3
"""
Test script to verify DHSCC receiver initialization with corrected 5G polar coding
"""

import torch
import numpy as np
import sys
import os

# Set deterministic behavior for reproducible results
torch.manual_seed(42)
np.random.seed(42)
if hasattr(torch, 'use_deterministic_algorithms'):
    torch.use_deterministic_algorithms(True)

# Add paths
sys.path.append('.')

def test_dhscc_receiver_initialization():
    """Test DHSCC receiver initialization with corrected 5G polar coding"""

    print("Testing DHSCC receiver initialization...")

    # Import the corrected receiver
    from dhscc_receiver import DHSCC_Receiver

    # Configuration matching the successful test_decoder_outputs.py
    polar_config = {
        'k_polar': 128,  # Rate 1/2, k=128
        'n_polar': 256,  # Matching sender N=256
        'channel_type': 'uplink'
    }

    # Use the actual checkpoint path
    model_checkpoint = "models/final_receiver_model.pth"

    try:
        # Create receiver with 5G decoder
        receiver = DHSCC_Receiver(
            polar_config=polar_config,
            model_checkpoint=model_checkpoint,
            intermediate_dim=128,
            num_classes=10,
            use_5g_decoder=True
        )

        print("✓ DHSCC Receiver initialized successfully with corrected 5G polar coding!")

        # Test with dummy data to verify the decoder works
        print("\nTesting decoder with dummy data...")

        batch_size = 2
        n = 256  # Codeword length

        # Create dummy LLRs (simulating received signal) on the correct device
        device = receiver.device
        dummy_llrs = torch.randn(batch_size, n, dtype=torch.float32, device=device)

        # Test polar decoding
        decoded_bits = receiver.polar_decoder(dummy_llrs)
        print(f"✓ Polar decoder output shape: {decoded_bits.shape}")
        print(f"✓ Polar decoder output type: {type(decoded_bits)}")

        # Verify output properties
        assert decoded_bits.shape == (batch_size, 128), f"Expected shape (2, 128), got {decoded_bits.shape}"
        assert decoded_bits.dtype == torch.float32, f"Expected float32, got {decoded_bits.dtype}"

        # Check that output contains binary values (0 or 1)
        unique_values = torch.unique(decoded_bits)
        print(f"✓ Unique values in decoded output: {unique_values}")

        # Should be close to 0 or 1 (binary)
        assert torch.all((decoded_bits >= -0.1) & (decoded_bits <= 1.1)), "Decoded bits should be binary-like"

        print("✓ All decoder tests passed!")

        return True

    except Exception as e:
        print(f"❌ DHSCC Receiver initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dhscc_receiver_initialization()
    if success:
        print("\n🎉 DHSCC receiver is now correctly initialized with modified 5G polar coding scheme!")
    else:
        print("\n❌ DHSCC receiver initialization still has issues.")
        sys.exit(1)