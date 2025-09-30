#!/usr/bin/env python3
"""
Sender Inference Script for DHSCC Split System

This script loads the trained sender model and quantizer, processes vibration
signal data in batches of 5 samples, and prepares the encoded features for
wireless transmission to a receiver machine.

Usage:
    python sender_inference.py --data_dir ../data/vibcls/vibrodata \
                              --test_list ../data/test.txt \
                              --sender_model checkpoints/final_sender_model.pth \
                              --quantizer_model checkpoints/final_quantizer.pth \
                              --batch_size 5
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import sys
import os
from torch.utils.data import DataLoader, Subset

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.insert(0, parent_dir)  # Add dhscc_split

# Import from local data_loader
from data_loader import WavDataset, load_file_list, load_class_names

from sender.sender_model import SenderNet
from sender.quantizer import VibrationVectorQuantizer
from ..channel.channel_modem import ChannelModem


def create_sender_inference_parser():
    """Create argument parser for sender inference."""
    parser = argparse.ArgumentParser(description='DHSCC Sender Inference')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default="../data/vibcls/vibrodata",
                        help='Directory containing data files')
    parser.add_argument('--test_list', type=str, default="../data/test.txt",
                        help='Path to test file list')
    parser.add_argument('--class_map', type=str, default="../data/clsmap.txt",
                        help='Path to class mapping file')

    # Model parameters
    parser.add_argument('--intermediate_dim', type=int, default=128,
                        help='Dimension of intermediate representation')
    parser.add_argument('--checkpoint', type=str, default="../../checkpoints/end_to_end_training_20250930_093233/final/best_end_to_end_model.pth",
                        help='Path to trained end-to-end model checkpoint')

    # Inference parameters
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Number of samples to process per batch (default: 5)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (default: all)')
    parser.add_argument('--channel_snr', type=float, default=10.0,
                        help='Signal-to-Noise Ratio for channel simulation (dB)')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default="./sender_output",
                        help='Directory to save processed features for transmission')

    return parser


def load_sender_models(args, device):
    """Load the trained sender model and quantizer from checkpoint."""
    print("Loading sender models from checkpoint...")

    # Create model arguments
    class ModelArgs:
        def __init__(self, args):
            self.intermediate_dim = args.intermediate_dim
            self.num_classes = 10  # Will be determined from data

    model_args = ModelArgs(args)

    # Load class names to determine number of classes
    class_names = load_class_names(args.class_map)
    model_args.num_classes = len(class_names)

    # Create quantizer arguments
    class QuantizerArgs:
        def __init__(self):
            self.quantizer_dimension = args.intermediate_dim
            self.quantizer_n_codes_group = 8
            self.quantizer_bins = 16
            self.quantizer_n_q = 32

    quantizer_args = QuantizerArgs()

    # Initialize models
    sender_model = SenderNet(model_args).to(device)
    quantizer = VibrationVectorQuantizer(quantizer_args).to(device)

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Load sender model
    if 'sender_state_dict' in checkpoint:
        sender_model.load_state_dict(checkpoint['sender_state_dict'])
    else:
        raise ValueError("Sender state dict not found in checkpoint")

    # Load quantizer model
    if 'quantizer_state_dict' in checkpoint:
        quantizer.load_state_dict(checkpoint['quantizer_state_dict'])
    else:
        raise ValueError("Quantizer state dict not found in checkpoint")

    # Set models to evaluation mode
    sender_model.eval()
    quantizer.eval()

    print("✓ Sender models loaded successfully")
    return sender_model, quantizer, model_args


def create_test_data_loader(args):
    """Create data loader for test data."""
    print("Creating test data loader...")

    # Load test file list
    test_files = load_file_list(args.test_list)

    # Create dataset
    test_dataset = WavDataset(
        file_list=test_files,
        data_dir=args.data_dir,
        augment=False  # No augmentation for testing
    )

    # Limit number of samples if specified
    if args.max_samples is not None and args.max_samples < len(test_dataset):
        indices = np.random.choice(len(test_dataset), args.max_samples, replace=False)
        test_dataset = Subset(test_dataset, indices)

    # Create data loader with specified batch size
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Process in order for consistent results
        num_workers=0,  # Avoid multiprocessing issues
        drop_last=False  # Keep all samples
    )

    print(f"✓ Test data loader created with {len(test_dataset)} samples, batch size {args.batch_size}")
    return test_loader


def process_batch(sender_model, quantizer, batch_data, batch_labels, device, channel_snr):
    """Process a batch of data through sender model and quantizer."""
    batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

    with torch.no_grad():
        # Add channel noise (as done in training)
        batch_size = batch_data.size(0)
        channel_noise = torch.randn(batch_size, 1).to(device) * 0.1 + 1.0

        # Forward pass through sender model
        sender_output = sender_model(batch_data, channel_noise)
        encoded_features, importance_weights, pre_task_logits, feature_shape, original_encoded_features = sender_output

        # Quantize the features
        quantized_features, quantization_loss, encoding_indices = quantizer(encoded_features)

        # Prepare transmission data
        transmission_data = {
            'quantized_features': quantized_features.cpu(),
            'importance_weights': importance_weights.cpu(),
            'feature_shape': feature_shape,
            'original_labels': batch_labels.cpu(),
            'encoding_indices': encoding_indices.cpu() if encoding_indices is not None else None
        }

    return transmission_data


def save_transmission_data(transmission_data, batch_idx, output_dir):
    """Save processed data for wireless transmission."""
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"batch_{batch_idx:04d}.pt")
    torch.save(transmission_data, output_file)

    print(f"✓ Saved transmission data for batch {batch_idx} to {output_file}")
    return output_file


def main():
    """Main inference function."""
    parser = create_sender_inference_parser()
    args = parser.parse_args()

    print("="*60)
    print("DHSCC SENDER INFERENCE")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Test list: {args.test_list}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Batch size: {args.batch_size}")
    print(f"Channel SNR: {args.channel_snr}dB")
    print(f"Output directory: {args.output_dir}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Load models
        sender_model, quantizer, model_args = load_sender_models(args, device)

        # Create data loader
        test_loader = create_test_data_loader(args)

        # Initialize channel modem (for potential future use)
        channel_modem = ChannelModem(model_args)

        # Process data in batches
        print("\n" + "-"*60)
        print("STARTING INFERENCE")
        print("-"*60)

        processed_batches = []
        for batch_idx, (batch_data, batch_labels) in enumerate(test_loader):
            print(f"\nProcessing batch {batch_idx + 1}/{len(test_loader)} (size: {batch_data.size(0)})")

            # Process batch
            transmission_data = process_batch(
                sender_model, quantizer, batch_data, batch_labels,
                device, args.channel_snr
            )

            # Save transmission data
            output_file = save_transmission_data(transmission_data, batch_idx, args.output_dir)
            processed_batches.append(output_file)

            # Print batch statistics
            print(f"  Features shape: {transmission_data['quantized_features'].shape}")
            print(f"  Importance weights shape: {transmission_data['importance_weights'].shape}")
            print(f"  Feature shape info: {transmission_data['feature_shape']}")

        print("\n" + "="*60)
        print("INFERENCE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Processed {len(processed_batches)} batches")
        print(f"Output directory: {args.output_dir}")
        print("\nReady for wireless transmission to receiver machine!")

    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Sender inference completed successfully!")
    else:
        print("\n❌ Sender inference failed!")
        sys.exit(1)