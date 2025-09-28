"""
Main entry point for DHSCC Receiver training and testing.

This module provides the entry point for training and testing the receiver
component independently, processing received features to produce classifications.

Key features:
- Independent receiver training
- Direct SNR input during training
- Model saving and loading for deployment
- Testing interface for receiver functionality
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from models.receiver_model import ReceiverModel
from utils.data import WavDataset, load_file_list, load_class_names
from utils.signal_processing import set_seed
from utils.file_io import ensure_dir
from receiver_train import train_receiver

def create_receiver_argument_parser():
    """Create and configure argument parser for receiver training."""
    parser = argparse.ArgumentParser(description='DHSCC Receiver Training')
    
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

    # ==================== 运行模式参数 ====================
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Run mode: train or test')
    parser.add_argument('--model_path', type=str, 
                        help='Path to model weights for testing')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computing device (cuda or cpu)')

    # ==================== 输出目录参数 ====================
    parser.add_argument('--checkpoint_dir', type=str, default='./receiver_checkpoints',
                        help='Directory for saving training checkpoints')
    parser.add_argument('--results_dir', type=str, default='./receiver_results',
                        help='Directory for saving test results')
    parser.add_argument('--seed', type=int, default=123, 
                        help='Random seed for reproducibility')

    return parser


def validate_receiver_arguments(args):
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


def load_receiver_datasets(args):
    """Load and create datasets and data loaders for receiver training."""
    # Load data files and class information
    train_file_list = load_file_list(args.train_list)
    test_file_list = load_file_list(args.test_list)
    class_names = load_class_names(args.class_map)
    
    print(f"Loaded {len(train_file_list)} training samples and {len(test_file_list)} test samples")
    print(f"Classes: {list(class_names.values())}")
    num_classes = len(class_names)
    
    # Create datasets
    # Note: For receiver training, we use the raw data and simulate channel effects
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


def simulate_received_features(data, snr_db, comm_mode):
    """
    Simulate features as they would be received from the channel.
    
    In real deployment, this would be replaced by actual received features
    from the channel module.
    """
    # Convert SNR to noise variance
    snr_linear = 10 ** (snr_db / 10)
    noise_variance = 1.0 / snr_linear
    
    # Simulate feature extraction and encoding (simplified)
    # In real system, this would be output from sender model
    batch_size, channels, time_samples = data.shape
    
    # Simulate encoded features (dimension matching sender output)
    # This is a simplified simulation - real features would come from sender
    encoded_features = torch.randn(batch_size, 128, device=data.device)
    
    if comm_mode in [1, 2]:
        # Analog transmission - add Gaussian noise
        noise = torch.randn_like(encoded_features) * torch.sqrt(torch.tensor(noise_variance))
        received_features = encoded_features + noise
        
    elif comm_mode in [3, 4]:
        # Digital transmission - simulate quantization and bit errors
        # Add quantization-like distortion
        quantization_noise = torch.randn_like(encoded_features) * 0.1
        
        # Add channel noise
        channel_noise = torch.randn_like(encoded_features) * torch.sqrt(torch.tensor(noise_variance))
        
        # Simulate bit errors
        bit_error_prob = max(0.001, (1.0 / snr_linear) * 0.1)
        error_mask = torch.rand_like(encoded_features) < bit_error_prob
        error_values = torch.randn_like(encoded_features) * 0.5
        
        received_features = encoded_features + quantization_noise + channel_noise
        received_features = torch.where(error_mask, error_values, received_features)
    
    else:
        # Default: add Gaussian noise
        noise = torch.randn_like(encoded_features) * torch.sqrt(torch.tensor(noise_variance))
        received_features = encoded_features + noise
    
    return received_features


def test_receiver(model, test_loader, device, args):
    """Test receiver model functionality."""
    model.eval()
    results = {}
    
    # Test across different SNR levels
    snr_range = np.linspace(args.snr_min, args.snr_max, 5)
    
    print("Testing receiver across SNR range...")
    for snr_db in snr_range:
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                
                # Simulate received features
                received_features = simulate_received_features(data, snr_db, args.communication_mode)
                
                # Get receiver output
                result = model(received_features, snr=snr_db)
                
                # Calculate loss
                loss = F.cross_entropy(result['classification_output'], target)
                total_loss += loss.item()
                
                # Test classification
                pred = result['classification_output'].argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                if batch_idx == 0:
                    # Print some debug info for first batch
                    print(f"SNR {snr_db:.1f}dB - Batch 0 info:")
                    print(f"  Original data shape: {data.shape}")
                    print(f"  Received features shape: {received_features.shape}")
                    print(f"  Classification output shape: {result['classification_output'].shape}")
                    print(f"  Loss: {loss.item():.4f}")
        
        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)
        
        results[f'snr_{snr_db:.1f}'] = {
            'accuracy': accuracy,
            'loss': avg_loss
        }
        
        print(f"SNR {snr_db:.1f}dB: Accuracy = {accuracy:.4f}, Loss = {avg_loss:.4f}")
    
    return results


def main():
    """Main execution function for receiver training or testing."""
    # Parse arguments
    parser = create_receiver_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not validate_receiver_arguments(args):
        return
    
    # Set random seed and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    
    print("=" * 60)
    print("DHSCC RECEIVER TRAINING/TESTING")
    print("=" * 60)
    print(f"Using device: {device}")
    print(f"Communication mode: {args.communication_mode}")
    print(f"SNR range: {args.snr_min} - {args.snr_max} dB")
    print(f"Mode: {args.mode}")
    
    # Load datasets
    train_loader, test_loader, num_classes = load_receiver_datasets(args)
    
    # Initialize receiver model
    model = ReceiverModel(args, num_classes=num_classes).to(device)
    
    print(f"Receiver model created with {args.intermediate_dim} intermediate dimensions")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if args.mode == 'train':
        # Training mode
        print("\nStarting receiver training...")
        
        # Create checkpoint directory
        ensure_dir(args.checkpoint_dir)
        
        # Run training
        training_results = train_receiver(model, train_loader, device, args, test_loader)
        
        print(f"\nReceiver training completed successfully!")
        print(f"Best accuracy: {training_results['best_accuracy']:.4f}")
        print(f"Training directory: {training_results['training_dir']}")
        
        # Test the final model
        if training_results['best_checkpoint']:
            print("\nTesting final receiver model...")
            checkpoint = torch.load(training_results['best_checkpoint'], map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            test_results = test_receiver(model, test_loader, device, args)
            print("\nReceiver testing completed!")
        
    else:
        # Testing mode
        print(f"\nLoading receiver model from {args.model_path}")
        
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print("Receiver model loaded successfully!")
        
        # Create results directory
        ensure_dir(args.results_dir)
        
        # Run test
        test_results = test_receiver(model, test_loader, device, args)
        
        print("\nReceiver testing completed!")


if __name__ == '__main__':
    main()