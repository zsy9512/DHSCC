"""
Main Script for Separated End-to-End Deployment

This script launches three independent processes:
1. Sender Process: Loads sender model, processes data, sends to channel
2. Channel Process: Simulates wireless channel with modulation/demodulation
3. Receiver Process: Loads receiver model, performs classification

All processes communicate via multiprocessing Queues (IPC).
"""

import argparse
import multiprocessing as mp
import sys
import os
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sender_process import sender_process, create_sender_args
from channel_process import channel_process, create_channel_args
from receiver_process import receiver_process, create_receiver_args


def create_main_args():
    """Create argument parser for main script."""
    parser = argparse.ArgumentParser(description='Separated End-to-End Deployment')

    # Common parameters
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'],
                        help='Mode: train or test')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default="data/vibcls/vibrodata")
    parser.add_argument('--train_list', type=str, default="data/vibcls/train.txt")
    parser.add_argument('--test_list', type=str, default="data/vibcls/test.txt")
    parser.add_argument('--class_map', type=str, default="data/vibcls/clsmap.txt")

    # Model parameters
    parser.add_argument('--intermediate_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')

    # Channel parameters
    parser.add_argument('--channel_model', type=str, default='rayleigh',
                        choices=['awgn', 'rayleigh', 'udp'])
    parser.add_argument('--channel_snr', type=float, default=10.0)

    # Communication parameters
    parser.add_argument('--communication_mode', type=int, default=4, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--bits_per_index', type=int, default=4)
    parser.add_argument('--max_groups', type=int, default=8)

    # Digital transmission parameters
    parser.add_argument('--N', type=int, default=128)
    parser.add_argument('--K', type=int, default=64)
    parser.add_argument('--channel_type', type=str, default='uplink', choices=['uplink', 'downlink'])
    parser.add_argument('--decoder_type', type=str, default='SC', choices=['SC', 'SCL'])
    parser.add_argument('--list_size', type=int, default=8)

    # Modulation parameters
    parser.add_argument('--constellation_type', type=str, default='qam', choices=['qam', 'bpsk', 'pam'])
    parser.add_argument('--normalize', action='store_true', default=True)
    parser.add_argument('--demapping_method', type=str, default='app', choices=['app', 'maxlog'])

    # Equalization
    parser.add_argument('--equalization_method', type=str, default='mrc', choices=['identity', 'zf', 'mrc', 'mmse'])

    # Control parameters
    parser.add_argument('--max_batches', type=int, help='Maximum batches to process (for testing)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    return parser


def main():
    """Main function to launch separated processes."""
    parser = create_main_args()
    args = parser.parse_args()

    print("\n" + "="*60)
    print("SEPARATED END-TO-END DEPLOYMENT")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Communication mode: {args.communication_mode}")
    print(f"Channel: {args.channel_model} at {args.channel_snr}dB SNR")
    print(f"Constellation: {args.constellation_type}")
    print()

    # Create IPC queues
    sender_to_channel = mp.Queue(maxsize=10)  # Sender -> Channel
    channel_to_receiver = mp.Queue(maxsize=10)  # Channel -> Receiver

    # Create process arguments
    sender_args = create_sender_args().parse_args([
        '--data_dir', args.data_dir,
        '--train_list', args.train_list,
        '--test_list', args.test_list,
        '--class_map', args.class_map,
        '--intermediate_dim', str(args.intermediate_dim),
        '--batch_size', str(args.batch_size),
        '--mode', args.mode,
        '--model_path', args.model_path or '',
        '--checkpoint_dir', args.checkpoint_dir,
        '--bits_per_index', str(args.bits_per_index),
        '--max_groups', str(args.max_groups),
    ])
    if args.max_batches:
        sender_args.max_batches = args.max_batches

    channel_args = create_channel_args().parse_args([
        '--channel_model', args.channel_model,
        '--channel_snr', str(args.channel_snr),
        '--communication_mode', str(args.communication_mode),
        '--bits_per_index', str(args.bits_per_index),
        '--N', str(args.N),
        '--K', str(args.K),
        '--channel_type', args.channel_type,
        '--decoder_type', args.decoder_type,
        '--list_size', str(args.list_size),
        '--constellation_type', args.constellation_type,
        '--normalize' if args.normalize else '--no-normalize',
        '--demapping_method', args.demapping_method,
        '--equalization_method', args.equalization_method,
    ])
    if args.verbose:
        channel_args.verbose = True

    receiver_args = create_receiver_args().parse_args([
        '--data_dir', args.data_dir,
        '--train_list', args.train_list,
        '--test_list', args.test_list,
        '--class_map', args.class_map,
        '--intermediate_dim', str(args.intermediate_dim),
        '--batch_size', str(args.batch_size),
        '--mode', args.mode,
        '--model_path', args.model_path or '',
        '--checkpoint_dir', args.checkpoint_dir,
    ])
    if args.max_batches:
        receiver_args.max_batches = args.max_batches

    # Create processes
    sender_proc = mp.Process(
        target=sender_process,
        args=(sender_args, sender_to_channel, None),
        name='SenderProcess'
    )

    channel_proc = mp.Process(
        target=channel_process,
        args=(channel_args, sender_to_channel, channel_to_receiver),
        name='ChannelProcess'
    )

    receiver_proc = mp.Process(
        target=receiver_process,
        args=(receiver_args, channel_to_receiver),
        name='ReceiverProcess'
    )

    print("🚀 Starting processes...")
    print("📤 Sender Process")
    print("📡 Channel Process")
    print("📥 Receiver Process")
    print()

    # Start processes
    start_time = time.time()

    receiver_proc.start()  # Start receiver first to be ready
    time.sleep(0.1)  # Small delay

    channel_proc.start()  # Start channel
    time.sleep(0.1)  # Small delay

    sender_proc.start()  # Start sender last

    # Wait for processes to complete
    sender_proc.join()
    print("✅ Sender process finished")

    channel_proc.join()
    print("✅ Channel process finished")

    receiver_proc.join()
    print("✅ Receiver process finished")

    end_time = time.time()

    print("\n" + "="*60)
    print("ALL PROCESSES COMPLETED")
    print("="*60)
    print(".2f")
    print("="*60)


if __name__ == "__main__":
    # Set multiprocessing start method for Windows compatibility
    if sys.platform == 'win32':
        mp.set_start_method('spawn', force=True)

    main()