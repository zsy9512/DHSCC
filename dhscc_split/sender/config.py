"""
Configuration for sender.
"""

import argparse

def create_argument_parser():
    """Create and configure argument parser for sender."""
    parser = argparse.ArgumentParser(description='DHSCC Sender')

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
    parser.add_argument('--epochs', type=int, default=220, help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_step', type=int, default=20, help='Learning rate decay step')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='Learning rate decay factor')

    # ==================== 三阶段训练参数 ====================
    parser.add_argument('--pretrain_epochs', type=int, default=120,
                        help='Number of epochs for pretraining phase')
    parser.add_argument('--selector_epochs', type=int, default=180,
                        help='Number of epochs for adaptive selector training phase')
    parser.add_argument('--final_epochs', type=int, default=200,
                        help='Number of epochs for final joint training phase')

    # ==================== 信道参数 (统一使用SNR) ====================
    parser.add_argument('--snr', type=float, default=10.0,
                        help='Training SNR level (dB)')
    parser.add_argument('--snr_min', type=float, default=5.0,
                        help='Minimum SNR for range testing (dB)')
    parser.add_argument('--snr_max', type=float, default=15.0,
                        help='Maximum SNR for range testing (dB)')
    parser.add_argument('--test_snr_steps', type=int, default=11,
                        help='Number of SNR steps for testing')

    # ==================== 通信模式配置 ====================
    parser.add_argument('--communication_mode', type=int, default=4, choices=[0, 1, 2, 3, 4],
                        help='Communication mode: '
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
    parser.add_argument('--N', type=int, default=128,
                        help='Polar code length (must be power of 2)')
    parser.add_argument('--K', type=int, default=96,
                        help='Information bits for polar coding')
    parser.add_argument('--channel_type', type=str, default='uplink',
                        choices=['uplink', 'downlink'],
                        help='5G polar coding channel type')

    # 调制参数
    parser.add_argument('--constellation_type', type=str, default='qam',
                        choices=['qam', 'bpsk', 'pam'],
                        help='Constellation type for modulation')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Whether to normalize constellation to unit power')

    # ==================== 自适应组选择器参数 (Modes 3,4) ====================
    parser.add_argument('--selector_snr_bins', type=int, default=21,
                        help='Number of SNR bins for adaptive group selector')
    parser.add_argument('--selector_algorithm', type=str, default='heuristic',
                        choices=['heuristic', 'rl'],
                        help='Adaptive group selector algorithm')
    parser.add_argument('--fixed_group', type=int, default=None,
                        help='Fixed number of groups to use (default: None for adaptive selection)')

    # ==================== 无线信道模型参数 ====================
    parser.add_argument('--channel_model', type=str, default='rayleigh',
                        choices=['awgn', 'rayleigh', 'udp'],
                        help='Wireless channel model: awgn, rayleigh, or udp')
    parser.add_argument('--equalization_method', type=str, default='mrc',
                        choices=['identity', 'zf', 'mrc', 'mmse'],
                        help='Channel equalization method for fading channels')

    # UDP信道参数
    parser.add_argument('--udp_tx_host', type=str, default='localhost',
                        help='UDP transmission host address')
    parser.add_argument('--udp_tx_port', type=int, default=12345,
                        help='UDP transmission port')
    parser.add_argument('--udp_timeout', type=float, default=10.0,
                        help='UDP socket timeout in seconds')
    parser.add_argument('--udp_buffer_size', type=int, default=8192,
                        help='UDP buffer size in bytes')
    parser.add_argument('--udp_max_retries', type=int, default=3,
                        help='Maximum UDP transmission retries')

    # ==================== 运行模式参数 ====================
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Run mode: train or test')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output for debugging')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computing device (cuda or cpu)')

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
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed for reproducibility')

    return parser

def validate_arguments(args):
    """Validate and adjust command line arguments."""
    # 根据mode参数设置test参数值
    args.test = 1 if args.mode == 'test' else 0

    # 确保channel_snr和snr参数一致
    args.channel_snr = args.snr

    # 验证数据文件路径
    import os
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
    if args.mode == 'test' and not hasattr(args, 'model_path'):
        print("Error: --model_path is required for testing mode")
        return False

    # 验证通信模式和信道模型的兼容性
    if args.communication_mode in [3, 4]:
        if args.channel_model == 'udp':
            print(f"Info: Using UDP channel for real hardware transmission")
            print(f"UDP configuration: TX={args.udp_tx_host}:{args.udp_tx_port}")
        else:
            print(f"Info: Using {args.channel_model.upper()} channel model with "
                  f"{args.equalization_method.upper()} equalization")

    # 计算测试SNR范围 - 修复参数名
    import numpy as np
    if hasattr(args, 'snr_min') and hasattr(args, 'snr_max') and hasattr(args, 'test_snr_steps'):
        args.test_snr_range = np.linspace(args.snr_min, args.snr_max, args.test_snr_steps).tolist()
    else:
        args.test_snr_range = [args.snr]  # Use single SNR if range not specified

    return True