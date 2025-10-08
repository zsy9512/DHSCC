"""
Test Script for Separated End-to-End Deployment

This script tests the IPC communication between the three processes
using randomly initialized models (for development/testing purposes).
"""

import argparse
import multiprocessing as mp
import sys
import os
import numpy as np
import torch

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sender_process import sender_process, create_sender_args
from channel_process import channel_process, create_channel_args
from receiver_process import receiver_process, create_receiver_args


def test_ipc_communication():
    """Test IPC communication with dummy data."""
    print("🧪 Testing IPC communication...")

    # Create queues
    sender_to_channel = mp.Queue()
    channel_to_receiver = mp.Queue()

    # Create test data packet
    test_packet = {
        'binary_bits': np.random.rand(4, 32).astype(np.float32),  # 4 samples, 32 bits each
        'target': np.array([0, 1, 2, 1], dtype=np.int64),
        'feature_shape': (4, 8, 4),  # Example shape
        'batch_size': 4,
        'batch_idx': 0
    }

    # Test sender -> channel
    print("📤 Testing sender to channel...")
    sender_to_channel.put(test_packet)
    received_packet = sender_to_channel.get()
    assert np.allclose(received_packet['binary_bits'], test_packet['binary_bits'])
    print("✅ Sender to channel communication OK")

    # Test channel -> receiver
    print("📡 Testing channel to receiver...")
    # Simulate what channel process does - create output_packet
    output_packet = {
        'received_bits': received_packet['binary_bits'],  # For simple test, just pass through
        'target': received_packet['target'],
        'feature_shape': received_packet['feature_shape'],
        'batch_size': received_packet['batch_size'],
        'batch_idx': received_packet['batch_idx']
    }
    channel_to_receiver.put(output_packet)
    final_packet = channel_to_receiver.get()
    assert np.allclose(final_packet['received_bits'], received_packet['binary_bits'])
    print("✅ Channel to receiver communication OK")

    print("🎉 IPC communication test passed!")


def test_with_dummy_models():
    """Test the full pipeline with dummy models."""
    print("\n🧪 Testing full pipeline with dummy models...")

    # Create argument parsers
    sender_parser = create_sender_args()
    channel_parser = create_channel_args()
    receiver_parser = create_receiver_args()

    # Parse minimal args
    sender_args = sender_parser.parse_args([
        '--mode', 'test',
        '--max_batches', '1'  # Only process 1 batch for testing
    ])

    channel_args = channel_parser.parse_args([
        '--communication_mode', '4',
        '--channel_snr', '10.0'
    ])

    receiver_args = receiver_parser.parse_args([
        '--mode', 'test'
    ])

    # Create queues
    sender_to_channel = mp.Queue()
    channel_to_receiver = mp.Queue()

    # Create processes
    sender_proc = mp.Process(
        target=sender_process,
        args=(sender_args, sender_to_channel, None),
        name='TestSender'
    )

    channel_proc = mp.Process(
        target=channel_process,
        args=(channel_args, sender_to_channel, channel_to_receiver),
        name='TestChannel'
    )

    receiver_proc = mp.Process(
        target=receiver_process,
        args=(receiver_args, channel_to_receiver),
        name='TestReceiver'
    )

    # Start processes
    print("🚀 Starting test processes...")
    receiver_proc.start()
    channel_proc.start()
    sender_proc.start()

    # Wait for completion
    sender_proc.join(timeout=30)
    if sender_proc.is_alive():
        print("⚠️  Sender process timed out")
        sender_proc.terminate()

    channel_proc.join(timeout=30)
    if channel_proc.is_alive():
        print("⚠️  Channel process timed out")
        channel_proc.terminate()

    receiver_proc.join(timeout=30)
    if receiver_proc.is_alive():
        print("⚠️  Receiver process timed out")
        receiver_proc.terminate()

    print("✅ Test processes completed")


def main():
    """Main test function."""
    print("="*60)
    print("SEPARATED END-TO-END DEPLOYMENT TEST")
    print("="*60)

    # Test IPC communication
    test_ipc_communication()

    # Test full pipeline
    test_with_dummy_models()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    # Set multiprocessing start method
    if sys.platform == 'win32':
        mp.set_start_method('spawn', force=True)

    main()