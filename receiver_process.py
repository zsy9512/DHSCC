"""
Receiver Process for Separated End-to-End Deployment

This module implements the receiver side of the separated communication system.
It loads the trained receiver model, receives demodulated bits from channel process,
and performs classification.
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

from receiver.receiver_model import ReceiverNet
from data.data_loader import create_data_loaders_from_args


def load_receiver_models(args, device, num_classes):
    """Load trained receiver model."""
    # print("🔄 Loading receiver models...")

    # Create models
    receiver_model = ReceiverNet(args, num_classes).to(device)

    # Load receiver model
    receiver_model_path = getattr(args, 'receiver_model_path', None)
    if not receiver_model_path:
        # Try to find the final_receiver_model.pth
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
                receiver_model_path = os.path.join(checkpoint_dir, latest_dir, 'final_receiver_model.pth')
                # print(f"📂 Found receiver model: {receiver_model_path}")

    if receiver_model_path and os.path.exists(receiver_model_path):
        receiver_checkpoint = torch.load(receiver_model_path, map_location=device)
        # Handle different checkpoint formats
        if 'state_dict' in receiver_checkpoint:
            receiver_model.load_state_dict(receiver_checkpoint['state_dict'])
        else:
            receiver_model.load_state_dict(receiver_checkpoint)
        # print(f"✅ Loaded receiver model from {receiver_model_path}")
    else:
        # print("⚠️  Receiver model not found, using randomly initialized model")
        pass

    receiver_model.eval()

    return receiver_model


def receiver_process(args, from_channel_queue):
    """Main receiver process function - directly classify reconstructed features."""
    # print("\n" + "="*50)
    # print("RECEIVER PROCESS STARTED")
    # print("="*50)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"📱 Receiver using device: {device}")

    # Create data loader to get num_classes
    train_loader, test_loader, num_classes = create_data_loaders_from_args(args)
    # print(f"📊 Number of classes: {num_classes}")

    # Load models
    receiver_model = load_receiver_models(args, device, num_classes)

    # Initialize metrics (will be dynamically created)
    results = {}
    last_batch_size = None  # Store the last batch size for loss calculation

    # print("📥 Receiver begin processing data")

    with torch.no_grad():
        while True:
            # Receive data from channel
            data_packet = from_channel_queue.get()

            if data_packet.get('type') == 'end':
                # print("📡 Received end signal from channel")
                break

            batch_idx = data_packet['batch_idx']
            batch_size = data_packet['batch_size']
            last_batch_size = batch_size  # Update last batch size
            snr_db = data_packet.get('snr_db', 15)  # Default SNR if not provided

            # print(f"📡 Processing batch {batch_idx + 1}, size: {batch_size}, SNR: {snr_db}dB")

            # Convert received data to tensors
            reconstructed_features = torch.tensor(data_packet['reconstructed_features'], dtype=torch.float, device=device)
            target = torch.tensor(data_packet['target'], dtype=torch.long, device=device)
            feature_shape = data_packet.get('feature_shape', None)  # Extract feature_shape from packet

            # Temporarily restore stdout for printing
            import sys
            original_stdout = sys.__stdout__
            sys.stdout = original_stdout
            print(f"Receive shapes: Features {reconstructed_features.shape}")
            sys.stdout = open(os.devnull, 'w')

            # Use received feature_shape if available, otherwise use default
            if feature_shape is not None:
                # Convert list/tuple back to torch.Size if needed
                if isinstance(feature_shape, (list, tuple)):
                    feature_shape = torch.Size(feature_shape)
            else:
                # Fallback to default shape (should match training)
                feature_shape = (batch_size, 8, 4)  # Default shape for 8 groups, 4 indices per group

            # Generate channel noise (same as training)
            channel_noise = torch.randn(batch_size, 1, device=device) * 0.1 + 1.0

            # Receiver forward pass (directly classify reconstructed features)
            receiver_output = receiver_model(reconstructed_features, channel_noise, feature_shape)

            # Calculate loss
            loss = nn.CrossEntropyLoss()(receiver_output, target)

            # Calculate accuracy
            predictions = receiver_output.argmax(dim=1)
            batch_correct = (predictions == target).sum().item()

            # Update results (dynamically create SNR entry if needed)
            if snr_db not in results:
                results[snr_db] = {'correct': 0, 'total': 0, 'loss': 0.0}
            results[snr_db]['correct'] += batch_correct
            results[snr_db]['total'] += batch_size
            results[snr_db]['loss'] += loss.item()

    # Output final results
    # Temporarily restore stdout for final results
    import sys
    original_stdout = sys.__stdout__
    sys.stdout = original_stdout
    print("\n分离式部署测试结果:")
    print("="*30)
    for snr in sorted(results.keys()):
        acc = results[snr]['correct'] / results[snr]['total'] * 100
        print(f"SNR {snr}dB: Accuracy = {acc:.2f}%")
    sys.stdout = open(os.devnull, 'w')

    # print("📥 Receiver process completed")


def create_receiver_args():
    """Create argument parser for receiver process."""
    parser = argparse.ArgumentParser(description='Receiver Process')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default="data/vibcls/vibrodata")
    parser.add_argument('--train_list', type=str, default="data/vibcls/train.txt")
    parser.add_argument('--test_list', type=str, default="data/vibcls/test.txt")
    parser.add_argument('--class_map', type=str, default="data/vibcls/clsmap.txt")

    # Model parameters
    parser.add_argument('--intermediate_dim', type=int, default=128)
    parser.add_argument('--N', type=int, default=256, help='Polar code length')
    parser.add_argument('--K', type=int, default=128, help='Information bits')
    parser.add_argument('--snr_min', type=int, default=10, help='Minimum SNR for testing')
    parser.add_argument('--snr_max', type=int, default=21, help='Maximum SNR for testing')

    # Other parameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--max_batches', type=int, help='Maximum batches to process (for testing)')

    # Model paths
    parser.add_argument('--receiver_model_path', type=str, help='Path to receiver model weights')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')

    return parser


if __name__ == "__main__":
    # This is for standalone testing
    parser = create_receiver_args()
    args = parser.parse_args()

    # Create dummy queue for testing
    from_channel = Queue()

    receiver_process(args, from_channel)