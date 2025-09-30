#!/usr/bin/env python3
"""
Model Connectivity Test for DHSCC System

This test validates that the sender -> quantizer -> receiver pipeline
is properly connected and functional.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sender.quantizer import VibrationVectorQuantizer
from sender.sender_model import SenderNet
from receiver.receiver_model import ReceiverNet

def test_model_connectivity():
    """
    Test that sender -> quantizer -> receiver pipeline works correctly
    """
    print("="*60)
    print("MODEL CONNECTIVITY TEST (Sender -> Quantizer -> Receiver)")
    print("="*60)

    print("Initializing models...")

    # Model configuration
    class Args:
        def __init__(self):
            self.input_dim = 128
            self.hidden_dim = 256
            self.intermediate_dim = 128
            self.num_classes = 10
            self.codebook_size = 512
            self.embedding_dim = 64
            self.commitment_cost = 0.25
            self.num_embeddings = 512

    args = Args()

    try:
        # Initialize models
        sender = SenderNet(args)
        quantizer = VibrationVectorQuantizer(args)
        receiver = ReceiverNet(args)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sender = sender.to(device)
        quantizer = quantizer.to(device)
        receiver = receiver.to(device)

        print("[OK] All models initialized successfully!")

        # Create test input
        batch_size = 4
        test_input = torch.randn(batch_size, 1, args.input_dim).to(device)
        print(f"Test input shape: {test_input.shape}")

        # Add some channel noise simulation
        channel_noise = torch.randn(batch_size, 1).to(device) * 0.1
        print(f"Channel noise shape: {channel_noise.shape}")

        print("\nTesting forward pass...")

        # Test sender
        sender_output = sender(test_input, channel_noise)
        sender_features = sender_output[0]  # final_features
        sender_importance = sender_output[1]  # sorted_importance
        print(f"[OK] Sender output - features: {sender_features.shape}, importance: {sender_importance.shape}")

        # Test quantizer
        quantizer_output = quantizer(sender_features)
        quantized_features = quantizer_output[0]  # quantized_st
        quantizer_indices = quantizer_output[1]  # indices
        print(f"[OK] Quantizer output - quantized: {quantized_features.shape}, indices: {quantizer_indices.shape}")

        # Test receiver
        target_shape = torch.Size([batch_size, 512, 8])  # Example target shape
        receiver_logits = receiver(quantized_features, channel_noise, target_shape)
        receiver_predictions = torch.argmax(receiver_logits, dim=-1)
        print(f"[OK] Receiver output - logits: {receiver_logits.shape}, predictions: {receiver_predictions.tolist()}")

        print("\n[SUCCESS] Model connectivity test PASSED!")
        print("All components are properly connected and functional.")

        return True

    except Exception as e:
        print(f"[ERROR] Model connectivity test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting Model Connectivity Test")
    success = test_model_connectivity()
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed!")
        sys.exit(1)