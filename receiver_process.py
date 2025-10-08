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
from sender.quantizer import VibrationVectorQuantizer
from data.data_loader import create_data_loaders_from_args


def load_receiver_models(args, device, num_classes):
    """Load trained receiver model."""
    print("🔄 Loading receiver models...")

    # Create models
    receiver_model = ReceiverNet(args, num_classes).to(device)

    # Load receiver model
    receiver_model_path = getattr(args, 'receiver_model_path', None)
    if not receiver_model_path:
        # Try to find the final_receiver_model.pth
        checkpoint_dir = getattr(args, 'checkpoint_dir', './checkpoints')
        receiver_model_path = os.path.join(checkpoint_dir, 'end_to_end_training_*', 'final_receiver_model.pth')
        import glob
        matching_files = glob.glob(receiver_model_path)
        if matching_files:
            receiver_model_path = max(matching_files, key=os.path.getctime)
            print(f"📂 Found receiver model: {receiver_model_path}")

    if receiver_model_path and os.path.exists(receiver_model_path):
        receiver_checkpoint = torch.load(receiver_model_path, map_location=device)
        receiver_model.load_state_dict(receiver_checkpoint['state_dict'])
        print(f"✅ Loaded receiver model from {receiver_model_path}")
    else:
        print("⚠️  Receiver model not found, using randomly initialized model")

    receiver_model.eval()

    return receiver_model


def receiver_process(args, from_channel_queue):
    """Main receiver process function."""
    print("\n" + "="*50)
    print("RECEIVER PROCESS STARTED")
    print("="*50)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Receiver using device: {device}")

    # Create data loader to get num_classes
    train_loader, test_loader, num_classes = create_data_loaders_from_args(args)
    print(f"📊 Number of classes: {num_classes}")

    # Load models
    receiver_model = load_receiver_models(args, device, num_classes)

    # Initialize metrics
    total_samples = 0
    correct_predictions = 0
    total_loss = 0.0
    processed_batches = 0

    while True:
        # Receive data from channel
        data_packet = from_channel_queue.get()

        if data_packet.get('type') == 'end':
            print("📡 Received end signal from channel")
            break

        batch_idx = data_packet['batch_idx']
        batch_size = data_packet['batch_size']
        total_samples += batch_size

        print(f"📡 Processing batch {batch_idx + 1}, size: {batch_size}")

        # Convert received data to tensors
        received_bits = torch.tensor(data_packet['received_bits'], dtype=torch.float, device=device)
        target = torch.tensor(data_packet['target'], dtype=torch.long, device=device)
        feature_shape = data_packet['feature_shape']

        # Convert received bits back to quantized features
        # This is a simplified reconstruction - in practice, this would be more complex
        # For now, assume received_bits can be directly used as features
        received_features = received_bits.view(batch_size, -1)

        # Generate channel noise (same as training)
        channel_noise = torch.randn(batch_size, 1, device=device) * 0.1 + 1.0

        # Receiver forward pass
        with torch.no_grad():
            receiver_output = receiver_model(received_features, channel_noise, feature_shape)

            # Calculate loss
            loss = nn.CrossEntropyLoss()(receiver_output, target)
            total_loss += loss.item()

            # Calculate accuracy
            predictions = receiver_output.argmax(dim=1)
            batch_correct = (predictions == target).sum().item()
            correct_predictions += batch_correct

        batch_accuracy = batch_correct / batch_size * 100
        print(f"📊 Batch {batch_idx + 1} - Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.2f}%")

        processed_batches += 1

    # Calculate final metrics
    if total_samples > 0:
        final_accuracy = correct_predictions / total_samples * 100
        average_loss = total_loss / processed_batches

        print("\n" + "="*50)
        print("RECEIVER PROCESS RESULTS")
        print("="*50)
        print(f"✅ Total samples processed: {total_samples}")
        print(f"✅ Final accuracy: {final_accuracy:.2f}%")
        print(f"✅ Average loss: {average_loss:.4f}")
        print(f"✅ Processed batches: {processed_batches}")

    print("✅ Receiver process completed.")


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