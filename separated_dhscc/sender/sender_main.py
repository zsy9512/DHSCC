"""
Main entry point for DHSCC Sender training and testing.

This module provides the entry point for training and testing the sender
component independently, including the quantizer as specified in the
refactoring requirements.

Key features:
- Independent sender training with quantizer integration
- Direct SNR input during training
- Model saving and loading for deployment
- Testing interface for sender functionality
"""

import argparse
import torch
import torch.nn as nn
import os
import numpy as np

from models.sender_model import SenderModel
from utils.data import WavDataset, load_file_list, load_class_names
from utils.signal_processing import set_seed
from utils.file_io import ensure_dir
from sender_train import train_sender

def create_sender_argument_parser():
    """Create and configure argument parser for sender training."""
    parser = argparse.ArgumentParser(description='DHSCC Sender Training')
    
    # ==================== 数据参数 ====================
    parser.add_argument('--data_dir', type=str, default="../dataset/vibcls/vibrodata",
                        help='Directory containing data files')
    parser.add_argument('--train_list', type=str, default="../dataset/vibcls/train.txt", 
                        help='Path to training file list')
    parser.add_argument('--test_list', type=str, default="../dataset/vibcls/test.txt",
                        help='Path to test file list') 
    parser.add_argument('--class_map', type=str, default="../dataset/vibcls/clsmap.txt",
                        help='Path to class mapping file')

    # ==================== 模型参数 ====================
    parser.add_argument('--intermediate_dim', type=int, default=128,
                        help='Dimension of intermediate representation')
    parser.add_argument('--epochs', type=int, default=200, help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_step', type=int, default=20, help='Learning rate decay step')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='Learning rate decay factor')

    # ==================== 训练阶段参数 ====================
    parser.add_argument('--pretrain_epochs', type=int, default=80,
                        help='Number of epochs for pretraining phase')
    parser.add_argument('--selector_epochs', type=int, default=140, 
                        help='Number of epochs for quantizer optimization phase')

    # ==================== SNR参数 (直接输入训练) ====================
    parser.add_argument('--snr_min', type=float, default=5.0, 
                        help='Minimum SNR for training range (dB)')
    parser.add_argument('--snr_max', type=float, default=15.0, 
                        help='Maximum SNR for training range (dB)')

    # ==================== 通信模式配置 ====================
    parser.add_argument('--communication_mode', type=int, default=4, choices=[1, 2, 3, 4],
                        help='Communication mode: '
                             '1: Analog transmission, '
                             '2: Analog transmission with vector quantization, '
                             '3: Digital transmission without channel coding, '
                             '4: Digital transmission with channel coding')

    # ==================== 量化器参数 ====================
    parser.add_argument('--quantizer_dimension', type=int, default=128,
                        help='Quantizer feature dimension')
    parser.add_argument('--quantizer_n_codes_group', type=int, default=8,
                        help='Number of codes per group in quantizer')
    parser.add_argument('--quantizer_bins', type=int, default=16,
                        help='Number of quantization bins')
    parser.add_argument('--quantizer_n_q', type=int, default=32,
                        help='Total number of quantization groups')

    # ==================== 运行模式参数 ====================
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Run mode: train or test')
    parser.add_argument('--model_path', type=str, 
                        help='Path to model weights for testing')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computing device (cuda or cpu)')

    # ==================== 输出目录参数 ====================
    parser.add_argument('--checkpoint_dir', type=str, default='./sender_checkpoints',
                        help='Directory for saving training checkpoints')
    parser.add_argument('--results_dir', type=str, default='./sender_results',
                        help='Directory for saving test results')
    parser.add_argument('--seed', type=int, default=123, 
                        help='Random seed for reproducibility')

    return parser


def validate_sender_arguments(args):
    """Validate and adjust command line arguments."""
    # Set test flag based on mode
    args.test = 1 if args.mode == 'test' else 0
    
    # Verify data files exist (with relative path handling)
    required_files = [args.train_list, args.test_list, args.class_map]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file not found: {file_path}")
            return False
    
    # Verify data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        return False
    
    # Verify test mode parameters
    if args.mode == 'test' and not args.model_path:
        print("Error: --model_path is required for testing mode")
        return False
    
    return True


def load_sender_datasets(args):
    """Load and create datasets and data loaders for sender training."""
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


def test_sender(model, test_loader, device, args):
    """Test sender model functionality."""
    model.eval()
    results = {}
    
    # Test across different SNR levels
    snr_range = np.linspace(args.snr_min, args.snr_max, 5)
    
    print("Testing sender across SNR range...")
    for snr_db in snr_range:
        correct = 0
        total = 0
        total_commitment_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                
                # Get sender output
                result = model(data, snr=snr_db)
                
                # Test pre-task classification
                pred = result['pre_task_logits'].argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Accumulate commitment loss
                total_commitment_loss += result['commitment_loss'].item()
                
                if batch_idx == 0:
                    # Print some debug info for first batch
                    print(f"SNR {snr_db:.1f}dB - Batch 0 info:")
                    print(f"  Input shape: {data.shape}")
                    print(f"  Encoded features shape: {result['encoded_features'].shape}")
                    if result['indices'] is not None:
                        print(f"  Quantization indices shape: {result['indices'].shape}")
                    print(f"  Commitment loss: {result['commitment_loss'].item():.4f}")
        
        accuracy = correct / total
        avg_commitment_loss = total_commitment_loss / len(test_loader)
        
        results[f'snr_{snr_db:.1f}'] = {
            'accuracy': accuracy,
            'commitment_loss': avg_commitment_loss
        }
        
        print(f"SNR {snr_db:.1f}dB: Accuracy = {accuracy:.4f}, Commitment Loss = {avg_commitment_loss:.4f}")
    
    return results


def main():
    """Main execution function for sender training or testing."""
    # Parse arguments
    parser = create_sender_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not validate_sender_arguments(args):
        return
    
    # Set random seed and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    
    print("=" * 60)
    print("DHSCC SENDER TRAINING/TESTING")
    print("=" * 60)
    print(f"Using device: {device}")
    print(f"Communication mode: {args.communication_mode}")
    print(f"SNR range: {args.snr_min} - {args.snr_max} dB")
    print(f"Mode: {args.mode}")
    
    # Load datasets
    train_loader, test_loader, num_classes = load_sender_datasets(args)
    
    # Initialize sender model
    model = SenderModel(args, num_classes=num_classes).to(device)
    
    print(f"Sender model created with {args.intermediate_dim} intermediate dimensions")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if args.mode == 'train':
        # Training mode
        print("\nStarting sender training...")
        
        # Create checkpoint directory
        ensure_dir(args.checkpoint_dir)
        
        # Run training
        training_results = train_sender(model, train_loader, device, args, test_loader)
        
        print(f"\nSender training completed successfully!")
        print(f"Best accuracy: {training_results['best_accuracy']:.4f}")
        print(f"Training directory: {training_results['training_dir']}")
        
        # Test the final model
        if training_results['best_checkpoint']:
            print("\nTesting final sender model...")
            checkpoint = torch.load(training_results['best_checkpoint'], map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            test_results = test_sender(model, test_loader, device, args)
            print("\nSender testing completed!")
        
    else:
        # Testing mode
        print(f"\nLoading sender model from {args.model_path}")
        
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print("Sender model loaded successfully!")
        
        # Create results directory
        ensure_dir(args.results_dir)
        
        # Run test
        test_results = test_sender(model, test_loader, device, args)
        
        print("\nSender testing completed!")


if __name__ == '__main__':
    main()