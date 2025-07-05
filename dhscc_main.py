"""
Main entry point for DHSCC vibration classification model training and testing.

This module implements the entry point for training and evaluating the DHSCC
(Deep Hybrid Semantic-Channel Coding) neural network for vibration signal
classification. The system supports multiple communication modes with various
channel conditions, simulating wireless transmission.

Key features:
- Training and testing of DHSCC models with three-phase training strategy
- Support for different communication modes (analog/digital)
- Variable bitrate feature transmission based on channel conditions
- Testing across different Signal-to-Noise Ratio (SNR) levels
- Modular architecture with separated training phases
"""

import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np

from models.dhsccnet import DHSCCNet
from utils.data import WavDataset, load_file_list, load_class_names
from utils.signal_processing import set_seed
from utils.file_io import ensure_dir
from dhscc_train import train
from dhscc_test import run_full_test

def create_argument_parser():
    """Create and configure argument parser for DHSCC system."""
    parser = argparse.ArgumentParser(description='DHSCC for Vibration Classification')
    
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
    parser.add_argument('--epochs', type=int, default=300, help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_step', type=int, default=20, help='Learning rate decay step')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='Learning rate decay factor')

    # ==================== 三阶段训练参数 ====================
    parser.add_argument('--pretrain_epochs', type=int, default=100,
                        help='Number of epochs for pretraining phase')
    parser.add_argument('--selector_epochs', type=int, default=100, 
                        help='Number of epochs for adaptive selector training phase')
    parser.add_argument('--final_epochs', type=int, default=100,
                        help='Number of epochs for final joint training phase')
    # ==================== 信道参数 (统一使用SNR) ====================
    parser.add_argument('--snr', type=float, default=3.0, 
                        help='Training SNR level (dB)')
    parser.add_argument('--snr_min', type=float, default=0.0, 
                        help='Minimum SNR for range testing (dB)')
    parser.add_argument('--snr_max', type=float, default=5.0, 
                        help='Maximum SNR for range testing (dB)')
    parser.add_argument('--test_snr_steps', type=int, default=11, 
                        help='Number of SNR steps for testing')

    # ==================== 通信模式配置 ====================
    parser.add_argument('--communication_mode', type=int, default=4, choices=[0, 1, 2, 3, 4],
                        help='Communication mode: '
                             '0: No wireless channel (baseline), '
                             '1: Analog transmission (direct Gaussian noise), '
                             '2: Analog transmission with vector quantization, '
                             '3: Digital transmission without channel coding, '
                             '4: Digital transmission with channel coding')

    # ==================== 量化器参数 (Modes 2,3,4) ====================
    parser.add_argument('--quantizer_dimension', type=int, default=128,
                        help='Quantizer feature dimension')
    parser.add_argument('--quantizer_n_codes_group', type=int, default=8,
                        help='Number of codes per group in quantizer')
    parser.add_argument('--quantizer_bins', type=int, default=16,
                        help='Number of quantization bins')
    parser.add_argument('--quantizer_n_q', type=int, default=32,
                        help='Total number of quantization groups (4*8)')

    # ==================== 数字传输参数 (Modes 3,4) ====================
    parser.add_argument('--bits_per_index', type=int, default=4,
                        help='Number of bits per quantization index for digital transmission')
    parser.add_argument('--max_groups', type=int, default=8,
                        help='Maximum number of groups available for transmission')
    
    # 信道编码参数
    parser.add_argument('--N', type=int, default=256,
                        help='Polar code length (must be power of 2)')
    parser.add_argument('--k', type=int, default=128,
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
    parser.add_argument('--selector_snr_bins', type=int, default=11,
                        help='Number of SNR bins for adaptive group selector')
    parser.add_argument('--selector_algorithm', type=str, default='heuristic', 
                        choices=['heuristic', 'rl'],
                        help='Adaptive group selector algorithm')
    parser.add_argument('--fixed_group', type=int, default=None,
                        help='Fixed number of groups to use (default: None for adaptive selection)')

    # ==================== 无线信道模型参数 ====================
    parser.add_argument('--channel_model', type=str, default='awgn',
                        choices=['awgn', 'rayleigh', 'udp'],
                        help='Wireless channel model: awgn, rayleigh, or udp')
    parser.add_argument('--equalization_method', type=str, default='zf',
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
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Run mode: train or test')
    parser.add_argument('--test_all_modes', action='store_true',
                        help='Test all communication modes')
    parser.add_argument('--model_path', type=str, 
                        help='Path to model weights for testing')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output for debugging')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computing device (cuda or cpu)')
    # Add test mode parameter for compatibility - set based on mode
    # Test parameter will be set automatically based on --mode argument
    
    # ==================== 重要性阈值参数 ====================
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Importance threshold for feature selection')
    
    # ==================== 输出目录参数 ====================
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Base output directory for results')

    # ==================== 文件路径参数 ====================
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory for saving training checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory for saving test results')
    parser.add_argument('--seed', type=int, default=127, 
                        help='Random seed for reproducibility')

    return parser
def validate_arguments(args):
    """Validate and adjust command line arguments."""
    # 根据mode参数设置test参数值
    args.test = 1 if args.mode == 'test' else 0
    
    # 确保channel_snr和snr参数一致
    args.channel_snr = args.snr
    
    # 验证数据文件路径
    required_files = [args.train_list, args.test_list, args.class_map]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file not found: {file_path}")
            return False
    
    # 验证数据目录
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        return False
    
    # 验证测试模式参数
    if args.mode == 'test' and not args.model_path:
        print("Error: --model_path is required for testing mode")
        return False
    
    # 验证通信模式和信道模型的兼容性
    if args.communication_mode in [3, 4]:
        if args.channel_model == 'udp':
            print(f"Info: Using UDP channel for real hardware transmission")
            print(f"UDP configuration: TX={args.udp_tx_host}:{args.udp_tx_port}, "
                  f"RX={args.udp_rx_host}:{args.udp_rx_port}")
        else:
            print(f"Info: Using {args.channel_model.upper()} channel model with "
                  f"{args.equalization_method.upper()} equalization")
    
    # 验证Polar码参数
    if args.communication_mode == 4:
        if args.N & (args.N - 1) != 0:  # 检查是否为2的幂
            print("Error: Polar code length N must be a power of 2")
            return False
        if args.k >= args.N:
            print("Error: Information bits k must be less than code length N")
            return False
        print(f"Info: Polar coding - N={args.N}, k={args.k}, rate={args.k/args.N:.3f}")
    
    # 计算测试SNR范围 - 修复参数名
    if hasattr(args, 'snr_min') and hasattr(args, 'snr_max') and hasattr(args, 'test_snr_steps'):
        args.test_snr_range = np.linspace(args.snr_min, args.snr_max, args.test_snr_steps).tolist()
    else:
        args.test_snr_range = [args.snr]  # Use single SNR if range not specified
    
    return True


def load_datasets(args):
    """Load and create datasets and data loaders."""
    # Load data files and class information
    train_file_list = load_file_list(args.train_list)
    test_file_list = load_file_list(args.test_list)
    class_names = load_class_names(args.class_map)
    
    print(f"Loaded {len(train_file_list)} training samples and {len(test_file_list)} test samples")
    print(f"Classes: {list(class_names.values())}")
    num_classes = len(class_names)
    
    # Create datasets
    train_dataset = WavDataset(train_file_list, args.data_dir, augment=True)
    test_dataset = WavDataset(test_file_list, args.data_dir, augment=False)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader, num_classes


def main():
    """Main execution function for training or testing the model."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not validate_arguments(args):
        return
    
    # Set random seed and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    
    print(f"Using device: {device}")
    print(f"Communication mode: {args.communication_mode}")
    if args.communication_mode in [3, 4]:
        print(f"Channel model: {args.channel_model}")
        print(f"Equalization method: {args.equalization_method}")
        if args.channel_model == 'udp':
            print(f"UDP TX: {args.udp_tx_host}:{args.udp_tx_port}")
            print(f"UDP RX: {args.udp_rx_host}:{args.udp_rx_port}")
    print(f"Training SNR: {args.snr}dB")
    print(f"Testing SNR range: {args.test_snr_range}")
    
    # Load datasets
    train_loader, test_loader, num_classes = load_datasets(args)
    
    # Initialize model
    model = DHSCCNet(args, num_classes=num_classes).to(device)
    
    # Set SNR parameters for the model - 修复参数名
    model.snr_min = args.snr_min
    model.snr_max = args.snr_max
    model.test = args.test  # 使用根据mode设置的test值
    
    print(f"Model created with {args.intermediate_dim} intermediate dimensions")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if args.mode == 'train':
        # Training mode
        print("Starting three-phase training...")
        
        # Create checkpoint directory
        ensure_dir(args.checkpoint_dir)
        
        # Run training
        training_results = train(model, train_loader, device, args, test_loader)
        
        print(f"Training completed successfully!")
        
        # Test the final model
        final_checkpoint = training_results['final']['final_checkpoint']
        if final_checkpoint:
            print("Testing final model...")
            checkpoint = torch.load(final_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load adaptive selector state if available
            if 'adaptive_selector_state' in checkpoint and hasattr(model, 'load_adaptive_selector_state'):
                model.load_adaptive_selector_state(checkpoint['adaptive_selector_state'])
                print("Adaptive selector state loaded successfully!")
                
                # Debug: Check if the selector is properly finalized
                if hasattr(model, 'adaptive_group_selector') and model.adaptive_group_selector is not None:
                    is_finalized = hasattr(model.adaptive_group_selector, '_finalized') and model.adaptive_group_selector._finalized
                    print(f"Adaptive selector finalized status: {is_finalized}")
                    if is_finalized:
                        print("✓ Adaptive selector was properly finalized during training")
                    else:
                        print("⚠️  Adaptive selector was not finalized, will be finalized during testing preparation")
            else:
                print("No adaptive selector state found in checkpoint")
            
            # Run comprehensive test
            ensure_dir(args.results_dir)
            test_results = run_full_test(model, test_loader, device, args)
            
            print("Training and testing completed!")
        
    else:
        # Testing mode
        print("Testing mode...")
        print(f"Loading model from {args.model_path}")
        
        # Load model
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load adaptive selector state if available
        if 'adaptive_selector_state' in checkpoint and hasattr(model, 'load_adaptive_selector_state'):
            model.load_adaptive_selector_state(checkpoint['adaptive_selector_state'])
            print("Adaptive selector state loaded successfully!")
            
            # Debug: Check if the selector is properly finalized
            if hasattr(model, 'adaptive_group_selector') and model.adaptive_group_selector is not None:
                is_finalized = hasattr(model.adaptive_group_selector, '_finalized') and model.adaptive_group_selector._finalized
                print(f"Adaptive selector finalized status: {is_finalized}")
                if is_finalized:
                    print("✓ Adaptive selector was properly finalized during training")
                else:
                    print("⚠️  Adaptive selector was not finalized, will be finalized during testing preparation")
        else:
            print("No adaptive selector state found in checkpoint")
        
        print("Model loaded successfully!")
        
        # Create results directory
        ensure_dir(args.results_dir)
        
        # Run comprehensive test
        test_results = run_full_test(model, test_loader, device, args)
        
        print("Testing completed!")


if __name__ == '__main__':
    main()
