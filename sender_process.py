"""
Sender Process for Separated End-to-End Deployment

This module implements the sender side of t    if quantizer_path and os.path.exists(quantizer_path):
        quantizer_checkpoint = torch.load(quantizer_path, map_location=device)
        # Handle different checkpoint formats
        if 'state_dict' in quantizer_checkpoint:
            quantizer.load_state_dict(quantizer_checkpoint['state_dict'])
        else:
            quantizer.load_state_dict(quantizer_checkpoint)
        # print(f"✅ Loaded quantizer from {quantizer_path}")
    else:
        # print("⚠️  Quantizer not found, using randomly initialized quantizer")
        passed communication system.
It loads the trained sender model and quantizer, processes input data,
and sends encoded bits through IPC to the channel process.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import argparse
from multiprocessing import Queue
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sender.sender_model import SenderNet
from sender.quantizer import VibrationVectorQuantizer
from data.data_loader import create_data_loaders_from_args
from channel.channel_modem import ChannelModem
from sender.polar_encoding import PolarEncoder


def indices_to_binary(indices, bits_per_index=4):
    """Convert quantizer indices to binary representation."""
    device = indices.device
    # Handle different shapes: (batch_size, total_indices) or (batch_size, n_groups, group_size)
    if len(indices.shape) == 2:
        batch_size, total_indices = indices.shape
        flat_indices = indices.reshape(-1)
    else:
        batch_size, n_groups, group_size = indices.shape
        flat_indices = indices.reshape(-1)
        total_indices = n_groups * group_size

    # Convert to binary
    binary = []
    for idx in flat_indices:
        bin_str = format(idx.item(), f'0{bits_per_index}b')
        binary.extend([int(b) for b in bin_str])

    binary_tensor = torch.tensor(binary, dtype=torch.float, device=device)
    return binary_tensor.view(batch_size, -1)


def load_sender_models(args, device):
    """Load trained sender model and quantizer."""

    # Create models
    sender_model = SenderNet(args).to(device)
    quantizer = VibrationVectorQuantizer(args).to(device)

    # Load sender model
    sender_model_path = getattr(args, 'sender_model_path', None)
    if not sender_model_path:
        # Try to find the final_sender_model.pth
        checkpoint_dir = getattr(args, 'checkpoint_dir', './checkpoints')
        # Look for training directories
        if os.path.exists(checkpoint_dir):
            subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
            # Find directories that match the pattern
            training_dirs = [d for d in subdirs if 'training' in d or 'direct_polar' in d]
            if training_dirs:
                # Use the most recent training directory
                training_dirs.sort(key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)), reverse=True)
                latest_dir = training_dirs[0]
                sender_model_path = os.path.join(checkpoint_dir, latest_dir, 'final_sender_model.pth')
                # print(f"📂 Found sender model: {sender_model_path}")

    if sender_model_path and os.path.exists(sender_model_path):
        sender_checkpoint = torch.load(sender_model_path, map_location=device)
        # Handle different checkpoint formats
        if 'state_dict' in sender_checkpoint:
            sender_model.load_state_dict(sender_checkpoint['state_dict'])
        else:
            sender_model.load_state_dict(sender_checkpoint)
        # print(f"✅ Loaded sender model from {sender_model_path}")
    else:
        # print("⚠️  Sender model not found, using randomly initialized model")
        pass

    # Load quantizer
    quantizer_path = getattr(args, 'quantizer_path', None)
    if not quantizer_path:
        # Try to find the final_quantizer.pth
        checkpoint_dir = getattr(args, 'checkpoint_dir', './checkpoints')
        # Look for training directories
        if os.path.exists(checkpoint_dir):
            subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
            # Find directories that match the pattern
            training_dirs = [d for d in subdirs if 'training' in d or 'direct_polar' in d]
            if training_dirs:
                # Use the most recent training directory
                training_dirs.sort(key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)), reverse=True)
                latest_dir = training_dirs[0]
                quantizer_path = os.path.join(checkpoint_dir, latest_dir, 'final_quantizer.pth')
                # print(f"📂 Found quantizer: {quantizer_path}")

    if quantizer_path and os.path.exists(quantizer_path):
        quantizer_checkpoint = torch.load(quantizer_path, map_location=device)
        # Handle different checkpoint formats
        if 'state_dict' in quantizer_checkpoint:
            quantizer.load_state_dict(quantizer_checkpoint['state_dict'])
        else:
            quantizer.load_state_dict(quantizer_checkpoint)
        # print(f"✅ Loaded quantizer from {quantizer_path}")
    else:
        # print("⚠️  Quantizer weights not found, using random reconstruction")
        pass

    sender_model.eval()
    quantizer.eval()

    return sender_model, quantizer


def sender_process(args, to_channel_queue, from_channel_queue=None):
    """Main sender process function with polar encoding."""

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    sender_model, quantizer = load_sender_models(args, device)

    # Create data loader
    train_loader, test_loader, num_classes = create_data_loaders_from_args(args)
    data_loader = test_loader if args.mode == 'test' else train_loader

    # Process data
    total_samples = 0
    processed_batches = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            total_samples += batch_size

            # Generate channel noise (same as training)
            channel_noise = torch.randn(batch_size, 1).to(device) * 0.1 + 1.0

            # Forward pass through sender
            sender_output = sender_model(data, channel_noise)
            features, importance_weights, pre_task_logits, feature_shape, original_encoded_features = sender_output

            # Quantize features
            quantized_features, quantizer_indices, vq_loss = quantizer.encode(features)

            # Adaptive group selection (same as training final phase)
            group_to_use = getattr(args, 'group_to_use', 4)  # Default to 4 groups for final phase
            if len(quantizer_indices.shape) == 3:
                # Shape: [batch_size, n_groups, group_size]
                selected_indices = quantizer_indices[:, :group_to_use, :]
            else:
                # Shape: [batch_size, total_indices] - select first group_to_use indices
                selected_indices = quantizer_indices[:, :group_to_use] if quantizer_indices.shape[1] > group_to_use else quantizer_indices

            # Convert indices to binary for transmission
            binary_bits = indices_to_binary(selected_indices, bits_per_index=args.bits_per_index)

            # Convert to CPU for IPC (channel process uses CPU)
            selected_indices_cpu = selected_indices.detach().cpu().numpy()
            target_cpu = target.detach().cpu().numpy()
            quantized_features_cpu = quantized_features.detach().cpu().numpy()

            # Prepare data packet (send indices directly for channel_modem)
            data_packet = {
                'quantizer_indices': selected_indices_cpu,
                'quantized_features': quantized_features_cpu,
                'target': target_cpu,
                'batch_size': batch_size,
                'batch_idx': batch_idx,
                'feature_shape': feature_shape  # Add feature_shape for receiver reconstruction
            }

            # Send to channel process
            to_channel_queue.put(data_packet)

            processed_batches += 1

            # Optional: limit batches for testing
            if hasattr(args, 'max_batches') and args.max_batches and processed_batches >= args.max_batches:
                break

    # Send end signal
    to_channel_queue.put({'type': 'end'})


def create_sender_args():
    """Create argument parser for sender process."""
    parser = argparse.ArgumentParser(description='Sender Process')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default="data/vibcls/vibrodata")
    parser.add_argument('--train_list', type=str, default="data/vibcls/train.txt")
    parser.add_argument('--test_list', type=str, default="data/vibcls/test.txt")
    parser.add_argument('--class_map', type=str, default="data/vibcls/clsmap.txt")

    # Model parameters
    parser.add_argument('--intermediate_dim', type=int, default=128)
    parser.add_argument('--quantizer_dimension', type=int, default=128)
    parser.add_argument('--quantizer_n_codes_group', type=int, default=8)
    parser.add_argument('--quantizer_bins', type=int, default=16)
    parser.add_argument('--quantizer_n_q', type=int, default=32)

    # Communication parameters
    parser.add_argument('--bits_per_index', type=int, default=4)
    parser.add_argument('--max_groups', type=int, default=8)
    parser.add_argument('--group_to_use', type=int, default=8, help='Number of groups to transmit')
    parser.add_argument('--N', type=int, default=256, help='Polar code length')
    parser.add_argument('--K', type=int, default=128, help='Information bits')

    # Model paths
    parser.add_argument('--sender_model_path', type=str, help='Path to sender model weights')
    parser.add_argument('--quantizer_path', type=str, help='Path to quantizer weights')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')

    # Other parameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--max_batches', type=int, help='Maximum batches to process (for testing)')

    return parser


if __name__ == "__main__":
    # This is for standalone testing
    parser = create_sender_args()
    args = parser.parse_args()

    # Create dummy queues for testing
    to_channel = Queue()
    from_channel = Queue()

    sender_process(args, to_channel, from_channel)