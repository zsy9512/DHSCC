#!/usr/bin/env python3
"""
Receiver Inference Script for DHSCC Split System

This script loads the trained receiver model, receives encoded features
from the wireless channel (simulated or real), decodes them, and performs
vibration signal classification.

Usage:
    python receiver_inference.py --receiver_model checkpoints/final_receiver_model.pth \
                                --input_dir ./sender_output \
                                --channel_snr 10.0
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import glob
from torch.utils.data import DataLoader

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, grandparent_dir)  # Add project root
sys.path.insert(0, parent_dir)      # Add dhscc_split
sys.path.insert(0, current_dir)     # Add receiver

from receiver.receiver_model import ReceiverNet
from ..channel.channel_modem import ChannelModem


def create_receiver_inference_parser():
    """Create argument parser for receiver inference."""
    parser = argparse.ArgumentParser(description='DHSCC Receiver Inference')

    # Model parameters
    parser.add_argument('--intermediate_dim', type=int, default=128,
                        help='Dimension of intermediate representation')
    parser.add_argument('--checkpoint', type=str, default="../../checkpoints/end_to_end_training_20250930_093233/final/best_end_to_end_model.pth",
                        help='Path to trained end-to-end model checkpoint')

    # Input parameters
    parser.add_argument('--input_dir', type=str, default="./sender_output",
                        help='Directory containing transmission data from sender')
    parser.add_argument('--batch_pattern', type=str, default="batch_*.pt",
                        help='Pattern to match transmission batch files')

    # Channel parameters
    parser.add_argument('--channel_snr', type=float, default=10.0,
                        help='Signal-to-Noise Ratio for channel simulation (dB)')
    parser.add_argument('--channel_model', type=str, default='rayleigh',
                        choices=['awgn', 'rayleigh', 'udp'],
                        help='Channel type for simulation')

    # Output parameters
    parser.add_argument('--output_file', type=str, default="./receiver_results.txt",
                        help='File to save classification results')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save detailed predictions to file')

    return parser


def load_receiver_model(args, device):
    """Load the trained receiver model from checkpoint."""
    print("Loading receiver model from checkpoint...")

    # Create model arguments
    class ModelArgs:
        def __init__(self, args):
            self.intermediate_dim = args.intermediate_dim
            self.num_classes = 10  # Will be determined from model

    model_args = ModelArgs(args)

    # Initialize model
    receiver_model = ReceiverNet(model_args, model_args.num_classes).to(device)

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Load receiver model
    if 'receiver_state_dict' in checkpoint:
        receiver_model.load_state_dict(checkpoint['receiver_state_dict'])
    else:
        raise ValueError("Receiver state dict not found in checkpoint")

    # Set model to evaluation mode
    receiver_model.eval()

    print("✓ Receiver model loaded successfully")
    return receiver_model, model_args


def find_transmission_batches(input_dir, batch_pattern):
    """Find all transmission batch files."""
    pattern = os.path.join(input_dir, batch_pattern)
    batch_files = sorted(glob.glob(pattern))

    if not batch_files:
        raise FileNotFoundError(f"No batch files found matching pattern: {pattern}")

    print(f"Found {len(batch_files)} transmission batches")
    return batch_files


def load_transmission_batch(batch_file):
    """Load a transmission batch from file."""
    if not os.path.exists(batch_file):
        raise FileNotFoundError(f"Transmission batch file not found: {batch_file}")

    transmission_data = torch.load(batch_file)

    # Validate required keys
    required_keys = ['quantized_features', 'importance_weights', 'feature_shape', 'original_labels']
    for key in required_keys:
        if key not in transmission_data:
            raise ValueError(f"Missing required key '{key}' in transmission batch: {batch_file}")

    return transmission_data


def simulate_channel_transmission(quantized_features, channel_modem, snr):
    """Simulate wireless channel transmission."""
    # Transmit through channel
    transmitted_signal = channel_modem.transmit(quantized_features, snr)

    # The transmit method already applies channel effects and returns received signal
    return transmitted_signal


def process_batch(receiver_model, transmission_data, channel_modem, device, channel_snr):
    """Process a received batch through the receiver model."""
    # Extract data
    quantized_features = transmission_data['quantized_features'].to(device)
    importance_weights = transmission_data['importance_weights'].to(device)
    feature_shape = transmission_data['feature_shape']
    original_labels = transmission_data['original_labels'].to(device)

    with torch.no_grad():
        # Simulate channel transmission
        received_signal = simulate_channel_transmission(quantized_features, channel_modem, channel_snr)

        # Forward pass through receiver model
        receiver_output = receiver_model(received_signal, channel_snr, feature_shape)

        # Get predictions
        predictions = receiver_output.argmax(dim=1)
        confidence_scores = torch.softmax(receiver_output, dim=1)

        # Calculate accuracy
        correct = (predictions == original_labels).sum().item()
        total = original_labels.size(0)
        accuracy = correct / total * 100.0

    return {
        'predictions': predictions.cpu(),
        'original_labels': original_labels.cpu(),
        'confidence_scores': confidence_scores.cpu(),
        'accuracy': accuracy,
        'batch_size': total
    }


def save_results(results, output_file, save_predictions=False):
    """Save classification results to file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DHSCC RECEIVER INFERENCE RESULTS\n")
        f.write("="*60 + "\n\n")

        total_samples = 0
        total_correct = 0

        for batch_idx, batch_result in enumerate(results):
            f.write(f"Batch {batch_idx + 1}:\n")
            f.write(f"  Samples: {batch_result['batch_size']}\n")
            f.write(f"  Accuracy: {batch_result['accuracy']:.2f}%\n")

            total_samples += batch_result['batch_size']
            total_correct += int(batch_result['batch_size'] * batch_result['accuracy'] / 100.0)

            if save_predictions:
                f.write("  Predictions:\n")
                predictions = batch_result['predictions']
                original_labels = batch_result['original_labels']
                confidence_scores = batch_result['confidence_scores']

                for i in range(len(predictions)):
                    pred = predictions[i].item()
                    true = original_labels[i].item()
                    conf = confidence_scores[i][pred].item()
                    status = "✓" if pred == true else "✗"
                    f.write(f"    Sample {i}: Pred={pred}, True={true}, Conf={conf:.3f} {status}\n")

            f.write("\n")

        overall_accuracy = total_correct / total_samples * 100.0 if total_samples > 0 else 0.0

        f.write("="*60 + "\n")
        f.write("OVERALL RESULTS:\n")
        f.write(f"  Total samples: {total_samples}\n")
        f.write(f"  Total correct: {total_correct}\n")
        f.write(f"  Overall accuracy: {overall_accuracy:.2f}%\n")
        f.write("="*60 + "\n")

    print(f"✓ Results saved to {output_file}")


def main():
    """Main inference function."""
    parser = create_receiver_inference_parser()
    args = parser.parse_args()

    print("="*60)
    print("DHSCC RECEIVER INFERENCE")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input directory: {args.input_dir}")
    print(f"Channel SNR: {args.channel_snr}dB")
    print(f"Channel model: {args.channel_model}")
    print(f"Output file: {args.output_file}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Load receiver model
        receiver_model, model_args = load_receiver_model(args, device)

        # Initialize channel modem
        channel_modem = ChannelModem(model_args)

        # Find transmission batches
        batch_files = find_transmission_batches(args.input_dir, args.batch_pattern)

        # Process all batches
        print("\n" + "-"*60)
        print("STARTING INFERENCE")
        print("-"*60)

        results = []
        total_samples = 0
        total_correct = 0

        for batch_idx, batch_file in enumerate(batch_files):
            print(f"\nProcessing batch {batch_idx + 1}/{len(batch_files)}: {os.path.basename(batch_file)}")

            # Load transmission data
            transmission_data = load_transmission_batch(batch_file)

            # Process batch
            batch_result = process_batch(
                receiver_model, transmission_data, channel_modem,
                device, args.channel_snr
            )

            results.append(batch_result)

            # Update totals
            total_samples += batch_result['batch_size']
            total_correct += int(batch_result['batch_size'] * batch_result['accuracy'] / 100.0)

            print(f"  Batch accuracy: {batch_result['accuracy']:.2f}% ({batch_result['batch_size']} samples)")

        # Calculate overall accuracy
        overall_accuracy = total_correct / total_samples * 100.0 if total_samples > 0 else 0.0

        print("\n" + "="*60)
        print("INFERENCE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Processed {len(batch_files)} batches, {total_samples} total samples")
        print(f"Overall accuracy: {overall_accuracy:.2f}%")

        # Save results
        save_results(results, args.output_file, args.save_predictions)

    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Receiver inference completed successfully!")
    else:
        print("\n❌ Receiver inference failed!")
        sys.exit(1)