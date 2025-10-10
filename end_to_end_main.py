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
from channel.channel_modem import ChannelModem
from train_combined import train_direct_polar, train_two_stage
from end_to_end_test import run_end_to_end_tests

# Import data utilities from parent directory
from data.data import WavDataset, load_file_list, load_class_names

# Import common data loader
from data.data_loader import create_data_loaders_from_args


def create_argument_parser():
    """Create and configure argument parser for end-to-end system."""
    parser = argparse.ArgumentParser(description='DHSCC Split End-to-End Training')

    # ==================== 数据参数 ====================
    parser.add_argument('--data_dir', type=str, default="data/vibcls/vibrodata",
                        help='Directory containing data files')
    parser.add_argument('--train_list', type=str, default="data/vibcls/train.txt",
                        help='Path to training file list')
    parser.add_argument('--test_list', type=str, default="data/vibcls/test.txt",
                        help='Path to test file list')
    parser.add_argument('--class_map', type=str, default="data/vibcls/clsmap.txt",
                        help='Path to class mapping file')

    # ==================== 模型参数 ====================
    parser.add_argument('--intermediate_dim', type=int, default=128,
                        help='Dimension of intermediate representation (feature vector size)')
    
    # ==================== 量化器参数 ====================
    parser.add_argument('--quantizer_dimension', type=int, default=128,
                        help='Dimension for vector quantization')
    parser.add_argument('--quantizer_n_codes_group', type=int, default=8,
                        help='Number of quantization groups')
    parser.add_argument('--quantizer_bins', type=int, default=16,
                        help='Codebook size per group')
    parser.add_argument('--quantizer_n_q', type=int, default=32,
                        help='Total number of quantizers')
    parser.add_argument('--epochs', type=int, default=120, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--lr_step', type=int, default=20,
                        help='Learning rate scheduler step size')
    parser.add_argument('--lr_gamma', type=float, default=0.5,
                        help='Learning rate scheduler gamma')

    # ==================== 三阶段训练参数 ====================
    parser.add_argument('--pretrain_epochs', type=int, default=20,
                        help='Number of epochs for pretraining phase')
    parser.add_argument('--selector_epochs', type=int, default=40,
                        help='Number of epochs for joint training phase')
    parser.add_argument('--final_epochs', type=int, default=200,
                        help='Number of epochs for final fine-tuning phase')

    # ==================== 信道参数 ====================
    parser.add_argument('--channel_snr', type=float, default=15.0,
                        help='Signal-to-Noise Ratio for channel simulation (dB)')
    parser.add_argument('--snr_min', type=float, default=10.0,
                        help='Minimum SNR for training/testing range (dB)')
    parser.add_argument('--snr_max', type=float, default=20.0,
                        help='Maximum SNR for training/testing range (dB)')
    parser.add_argument('--test_snr_steps', type=int, default=11, 
                        help='Number of SNR steps for testing')
    parser.add_argument('--channel_model', type=str, default='awgn',
                        choices=['awgn', 'rayleigh', 'udp'],
                        help='Channel type for simulation')

    # ==================== 训练控制参数 ====================
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints",
                        help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu), auto-detect if None')

    # ==================== 模式选择 ====================
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='Mode: train or test')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    # ==================== 通信模式配置 ====================
    parser.add_argument('--communication_mode', type=int, default=4, choices=[0, 1, 2, 3, 4],
                        help='Communication mode: '
                             '1: Analog transmission (direct Gaussian noise), '
                             '2: Analog transmission with vector quantization, '
                             '3: Digital transmission without channel coding, '
                             '4: Digital transmission with channel coding')

    # ==================== 训练模式配置 ====================
    parser.add_argument('--training_mode', type=str, default='direct_polar',
                        choices=['direct_polar', 'two_stage'],
                        help='Training mode: '
                             'direct_polar: Direct training with polar coding, '
                             'two_stage: First train without polar coding, then with polar coding')

    # ==================== 数字传输参数 (Modes 3,4) ====================
    parser.add_argument('--bits_per_index', type=int, default=4,
                        help='Number of bits per quantization index for digital transmission')
    parser.add_argument('--max_groups', type=int, default=8,
                        help='Maximum number of groups available for transmission')
    
    # 信道编码参数
    parser.add_argument('--N', type=int, default=256,
                        help='Polar code length (must be power of 2)')
    parser.add_argument('--K', type=int, default=128,
                        help='Information bits for polar coding')
    parser.add_argument('--channel_type', type=str, default='uplink',
                        choices=['uplink', 'downlink'],
                        help='5G polar coding channel type')
    parser.add_argument('--decoder_type', type=str, default='SC',
                        choices=['SC', 'SCL'],
                        help='Polar decoder type')
    parser.add_argument('--list_size', type=int, default=8,
                        help='List size for SCL decoder')
    
    # 调制参数
    parser.add_argument('--constellation_type', type=str, default='qam',
                        choices=['qam', 'bpsk', 'pam'],
                        help='Constellation type for modulation')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Whether to normalize constellation to unit power')
    parser.add_argument('--demapping_method', type=str, default='app',
                        choices=['app', 'maxlog'],
                        help='Demapping method for soft decisions')

    # ==================== 自适应组选择器参数 (Modes 3,4) ====================
    parser.add_argument('--selector_snr_bins', type=int, default=21,
                        help='Number of SNR bins for adaptive group selector')
    parser.add_argument('--selector_algorithm', type=str, default='heuristic', 
                        choices=['heuristic', 'rl'],
                        help='Adaptive group selector algorithm')
    parser.add_argument('--fixed_group', type=int, default=None,
                        help='Fixed number of groups to use (default: None for adaptive selection)')

    # ==================== 无线信道模型参数 ====================
    parser.add_argument('--equalization_method', type=str, default='mrc',
                        choices=['identity', 'zf', 'mrc', 'mmse'],
                        help='Channel equalization method for fading channels')
    
    # UDP信道参数
    parser.add_argument('--udp_tx_host', type=str, default='localhost',
                        help='UDP transmission host address')
    parser.add_argument('--udp_tx_port', type=int, default=12345,
                        help='UDP transmission port')
    parser.add_argument('--udp_rx_host', type=str, default='localhost',
                        help='UDP reception host address')
    parser.add_argument('--udp_rx_port', type=int, default=12346,
                        help='UDP reception port')
    parser.add_argument('--udp_timeout', type=float, default=10.0,
                        help='UDP socket timeout in seconds')
    parser.add_argument('--udp_buffer_size', type=int, default=8192,
                        help='UDP buffer size in bytes')
    parser.add_argument('--udp_max_retries', type=int, default=3,
                        help='Maximum UDP transmission retries')

    # ==================== 运行模式参数 ====================
    parser.add_argument('--test_all_modes', action='store_true',
                        help='Test all communication modes')
    parser.add_argument('--model_path', type=str, 
                        help='Path to model weights for testing')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output for debugging')

    # ==================== 重要性阈值参数 ====================
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Importance threshold for feature selection')
    
    # ==================== 输出目录参数 ====================
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Base output directory for results')

    # ==================== 文件路径参数 ====================
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory for saving test results')

    return parser


def setup_device(device_str):
    """Setup torch device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
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
    print(f"Training mode: {args.training_mode}")

    # Setup
    device = setup_device(None)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create data loaders
    train_loader, test_loader, num_classes = create_data_loaders(args)

    if args.training_mode == 'direct_polar':
        # Direct training with polar coding
        print("Using direct polar coding training...")
        
        # Create models
        sender_model = SenderNet(args).to(device)
        receiver_model = ReceiverNet(args, num_classes).to(device)
        quantizer = VibrationVectorQuantizer(args).to(device)
        channel_modem = ChannelModem(args)

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

        # Train the model with polar coding
        results = train_direct_polar(
            sender_model=sender_model,
            receiver_model=receiver_model,
            quantizer=quantizer,
            channel_modem=channel_modem,
            train_loader=train_loader,
            device=device,
            args=args,
            test_loader=test_loader
        )

    elif args.training_mode == 'two_stage':
        # Two-stage training: first without polar coding, then with polar coding
        print("Using two-stage training...")
        
        # Create models for two-stage training
        sender_model = SenderNet(args).to(device)
        receiver_model = ReceiverNet(args, num_classes).to(device)
        quantizer = VibrationVectorQuantizer(args).to(device)
        channel_modem = ChannelModem(args)

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

        # Train using two-stage approach
        results = train_two_stage(
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

    # Create test data loader
    test_loader = None
    num_classes = 10  # Default fallback
    try:
        from torch.utils.data import DataLoader
        train_loader, test_loader_temp, num_classes = create_data_loaders(args)
        test_loader = test_loader_temp
        print(f"✓ Created test data loader with {num_classes} classes")
    except Exception as e:
        print(f"⚠️  Could not create test data loader: {e}")
        print("Using dummy data for testing")

    # Determine model path
    model_path = None
    if hasattr(args, 'model_path') and args.model_path:
        model_path = args.model_path
        print(f"📁 Using specified model path: {model_path}")
    elif hasattr(args, 'checkpoint_dir') and args.checkpoint_dir:
        print(f"📁 Using checkpoint directory: {args.checkpoint_dir}")

    # Run comprehensive tests
    success = run_end_to_end_tests(
        checkpoint_dir=args.checkpoint_dir if not model_path else None,
        test_loader=test_loader,
        args=args,
        num_classes=num_classes,
        model_path=model_path
    )

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
    print(f"Training mode: {args.training_mode}")
    print(f"Communication mode: {args.communication_mode}")
    print(f"Intermediate dim: {args.intermediate_dim}")
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