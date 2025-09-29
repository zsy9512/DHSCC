"""
Main entry point for DHSCC Split End-to-End Training and Testing.

This module implements the entry point for training and evaluating the
end-to-end sender-receiver communication system for vibration signal classification.

Key features:
- End-to-end training of sender and receiver models
- Channel simulation with variable SNR
- Performance evaluation across different channel conditions
- Modular architecture with separated training phases
"""

import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
import sys
from torch.utils.data import TensorDataset, DataLoader

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from sender.sender_model import SenderNet
from receiver.receiver_model import ReceiverNet
from sender.quantizer import VibrationVectorQuantizer
from channel_modem.channel_modem import ChannelModem
from train_end_to_end import train_end_to_end
from tests.end_to_end_test import run_end_to_end_tests

# Import data utilities from parent directory
from utils.data import WavDataset, load_file_list, load_class_names

# Import common data loader
from data_loader import create_data_loaders_from_args


def create_argument_parser():
    """Create and configure argument parser for end-to-end system."""
    parser = argparse.ArgumentParser(description='DHSCC Split End-to-End Training')

    # ==================== 数据参数 ====================
    parser.add_argument('--data_dir', type=str, default="dataset/vibcls/vibrodata",
                        help='Directory containing data files')
    parser.add_argument('--train_list', type=str, default="dataset/vibcls/train.txt",
                        help='Path to training file list')
    parser.add_argument('--test_list', type=str, default="dataset/vibcls/test.txt",
                        help='Path to test file list')
    parser.add_argument('--class_map', type=str, default="dataset/vibcls/clsmap.txt",
                        help='Path to class mapping file')

    # ==================== 模型参数 ====================
    parser.add_argument('--intermediate_dim', type=int, default=128,
                        help='Dimension of intermediate representation (feature vector size)')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classification classes')
    
    # ==================== 量化器参数 ====================
    parser.add_argument('--quantizer_dimension', type=int, default=128,
                        help='Dimension for vector quantization')
    parser.add_argument('--quantizer_n_codes_group', type=int, default=8,
                        help='Number of quantization groups')
    parser.add_argument('--quantizer_bins', type=int, default=256,
                        help='Codebook size per group')
    parser.add_argument('--quantizer_n_q', type=int, default=4,
                        help='Total number of quantizers')
    parser.add_argument('--vq_loss_weight', type=float, default=0.1,
                        help='Weight for VQ commitment loss')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--lr_step', type=int, default=50,
                        help='Learning rate scheduler step size')
    parser.add_argument('--lr_gamma', type=float, default=0.5,
                        help='Learning rate scheduler gamma')

    # ==================== 三阶段训练参数 ====================
    parser.add_argument('--pretrain_epochs', type=int, default=80,
                        help='End epoch for pretraining phase (sender only)')
    parser.add_argument('--selector_epochs', type=int, default=100,
                        help='End epoch for joint training phase')
    parser.add_argument('--final_epochs', type=int, default=120,
                        help='End epoch for final fine-tuning phase')

    # ==================== 信道参数 ====================
    parser.add_argument('--channel_snr', type=float, default=3.0,
                        help='Signal-to-Noise Ratio for channel simulation (dB)')
    parser.add_argument('--snr_min', type=float, default=0.0,
                        help='Minimum SNR for training/testing range (dB)')
    parser.add_argument('--snr_max', type=float, default=5.0,
                        help='Maximum SNR for training/testing range (dB)')
    parser.add_argument('--channel_type', type=str, default='awgn',
                        choices=['awgn', 'rayleigh', 'rician'],
                        help='Channel type for simulation')

    # ==================== 训练控制参数 ====================
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints",
                        help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu), auto-detect if None')

    # ==================== 模式选择 ====================
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'both'],
                        help='Mode: train, test, or both')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')

    return parser


def setup_device(device_str):
    """Setup torch device."""
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")

    return device


def load_checkpoint(checkpoint_path, sender_model, receiver_model, quantizer, sender_optimizer, receiver_optimizer):
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    sender_model.load_state_dict(checkpoint['sender_state_dict'])
    receiver_model.load_state_dict(checkpoint['receiver_state_dict'])
    quantizer.load_state_dict(checkpoint['quantizer_state_dict'])

    if sender_optimizer is not None:
        sender_optimizer.load_state_dict(checkpoint['sender_optimizer_state_dict'])
    if receiver_optimizer is not None:
        receiver_optimizer.load_state_dict(checkpoint['receiver_optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    print(f"Loaded checkpoint from epoch {epoch}")

    return epoch


def create_data_loaders(args):
    """Create data loaders for training and testing using actual vibration dataset."""
    return create_data_loaders_from_args(args)


def train_model(args):
    """Train the end-to-end model."""
    print("\n" + "="*60)
    print("STARTING END-TO-END TRAINING")
    print("="*60)

    # Setup
    device = setup_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create models
    sender_model = SenderNet(args).to(device)
    receiver_model = ReceiverNet(args, args.num_classes).to(device)
    quantizer = VibrationVectorQuantizer(args).to(device)
    channel_modem = ChannelModem(args)

    # Create data loaders
    train_loader, test_loader, num_classes = create_data_loaders(args)

    # Update args.num_classes if it was auto-detected
    if hasattr(args, 'num_classes') and args.num_classes != num_classes:
        print(f"Auto-detected {num_classes} classes from dataset")
        args.num_classes = num_classes

    # Resume from checkpoint if specified
    if args.resume:
        sender_optimizer = torch.optim.Adam(sender_model.parameters(), lr=args.lr)
        receiver_optimizer = torch.optim.Adam(receiver_model.parameters(), lr=args.lr)
        start_epoch = load_checkpoint(args.resume, sender_model, receiver_model, quantizer, sender_optimizer, receiver_optimizer)
        print(f"Resuming training from epoch {start_epoch}")
    else:
        sender_optimizer = None
        receiver_optimizer = None
        start_epoch = 0

    # Train the model
    results = train_end_to_end(
        sender_model=sender_model,
        receiver_model=receiver_model,
        quantizer=quantizer,
        channel_modem=channel_modem,
        train_loader=train_loader,
        device=device,
        args=args,
        test_loader=test_loader
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Training directory: {results['training_dir']}")
    print(f"Best checkpoint: {results['best_checkpoint']}")
    print(f"Final train accuracy: {results['final_metrics']['train_acc']:.2f}%")
    if results['final_metrics']['test_acc'] is not None:
        print(f"Final test accuracy: {results['final_metrics']['test_acc']:.2f}%")
    print(f"Final BER: {results['final_metrics']['ber']:.6f}")

    return results


def test_model(args):
    """Test the trained model."""
    print("\n" + "="*60)
    print("STARTING END-TO-END TESTING")
    print("="*60)

    # Run comprehensive tests
    success = run_end_to_end_tests()

    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")

    return success


def main():
    """Main function."""
    parser = create_argument_parser()
    args = parser.parse_args()

    print("DHSCC Split End-to-End System")
    print(f"Mode: {args.mode}")
    print(f"Intermediate dim: {args.intermediate_dim}")
    print(f"Classes: {args.num_classes}")
    print(f"Channel SNR: {args.channel_snr}dB")

    if args.mode in ['train', 'both']:
        results = train_model(args)

    if args.mode in ['test', 'both']:
        test_success = test_model(args)

    print("\n" + "="*60)
    print("ALL TASKS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()