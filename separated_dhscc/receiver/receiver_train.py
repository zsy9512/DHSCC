"""
Receiver Training Module for DHSCC

This module implements independent training for the receiver component.
The receiver is trained to process received features and produce classifications.

Key features:
- Independent receiver training
- Direct SNR input during training
- Adaptive to different communication modes
- Model checkpointing and metrics tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import time

from models.receiver_model import ReceiverModel
from utils.file_io import ensure_dir
from utils.signal_processing import set_seed

def create_receiver_training_directory(base_dir="receiver_checkpoints"):
    """
    Create timestamped directory for receiver training outputs.
    
    Args:
        base_dir: Base directory for checkpoints
        
    Returns:
        str: Path to created training directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_dir = os.path.join(base_dir, f"receiver_training_{timestamp}")
    ensure_dir(training_dir)
    
    # Create subdirectories
    subdirs = ['models', 'metrics', 'plots', 'logs']
    for subdir in subdirs:
        ensure_dir(os.path.join(training_dir, subdir))
    
    print(f"📁 Created receiver training directory: {training_dir}")
    return training_dir


def save_receiver_metrics(history, save_dir):
    """Save training metrics to files."""
    metrics_file = os.path.join(save_dir, "metrics", "training_metrics.pkl")
    with open(metrics_file, 'wb') as f:
        pickle.dump(history, f)
    print(f"Metrics saved to {metrics_file}")


def plot_receiver_training_progress(history, save_dir):
    """Plot and save training progress."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Receiver Training Progress', fontsize=16)
    
    # Loss plot
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    if 'test_loss' in history and history['test_loss']:
        test_epochs = range(1, len(history['test_loss']) + 1)
        axes[0, 0].plot(test_epochs, history['test_loss'], 'r-', label='Test Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    if 'test_acc' in history and history['test_acc']:
        test_epochs = range(1, len(history['test_acc']) + 1)
        axes[0, 1].plot(test_epochs, history['test_acc'], 'r-', label='Test Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Classification Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # SNR distribution
    if 'snr_values' in history and history['snr_values']:
        axes[1, 0].hist(history['snr_values'], bins=20, alpha=0.7)
        axes[1, 0].set_xlabel('SNR (dB)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Training SNR Distribution')
        axes[1, 0].grid(True)
    
    # Learning rate
    if 'learning_rates' in history and history['learning_rates']:
        axes[1, 1].plot(epochs, history['learning_rates'], 'g-')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_file = os.path.join(save_dir, "plots", "receiver_training_progress.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training plot saved to {plot_file}")


def save_receiver_checkpoint(model, optimizer, epoch, metrics, save_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': {
            'comm_mode': model.comm_mode,
            'hidden_channel': model.hidden_channel,
            'num_classes': model.num_classes
        }
    }
    
    filename = f"receiver_checkpoint_epoch_{epoch}.pth"
    if is_best:
        filename = f"receiver_best_model.pth"
    
    filepath = os.path.join(save_dir, "models", filename)
    torch.save(checkpoint, filepath)
    print(f"Receiver checkpoint saved: {filepath}")
    return filepath


def simulate_channel_effects(features, snr_db, comm_mode, noise_type='gaussian'):
    """
    Simulate channel effects on features for receiver training.
    
    This function simulates what would happen to features after transmission
    through the channel, so the receiver can be trained independently.
    
    Args:
        features: Clean features [batch_size, feature_dim]
        snr_db: Signal-to-noise ratio in dB
        comm_mode: Communication mode
        noise_type: Type of noise to add
        
    Returns:
        torch.Tensor: Features with simulated channel effects
    """
    # Convert SNR to noise variance
    snr_linear = 10 ** (snr_db / 10)
    noise_variance = 1.0 / snr_linear
    
    if comm_mode in [1, 2]:
        # Analog transmission - add Gaussian noise
        if noise_type == 'gaussian':
            noise = torch.randn_like(features) * torch.sqrt(torch.tensor(noise_variance))
        elif noise_type == 'uniform':
            noise = (torch.rand_like(features) - 0.5) * 2 * torch.sqrt(torch.tensor(3 * noise_variance))
        else:
            noise = torch.randn_like(features) * torch.sqrt(torch.tensor(noise_variance))
            
        received_features = features + noise
        
    elif comm_mode in [3, 4]:
        # Digital transmission - simulate quantization errors and bit errors
        # First add some quantization-like distortion
        quantization_noise = torch.randn_like(features) * 0.1
        
        # Then add channel noise
        channel_noise = torch.randn_like(features) * torch.sqrt(torch.tensor(noise_variance))
        
        # Simulate some bit errors by randomly flipping some feature values
        bit_error_prob = max(0.001, (1.0 / snr_linear) * 0.1)  # Approximate BER
        error_mask = torch.rand_like(features) < bit_error_prob
        error_values = torch.randn_like(features) * 0.5
        
        received_features = features + quantization_noise + channel_noise
        received_features = torch.where(error_mask, error_values, received_features)
        
    else:
        # Default: just add Gaussian noise
        noise = torch.randn_like(features) * torch.sqrt(torch.tensor(noise_variance))
        received_features = features + noise
    
    return received_features


def train_receiver(model, train_loader, device, args, test_loader=None):
    """
    Main receiver training function.
    
    Args:
        model: ReceiverModel instance
        train_loader: Training data loader (should contain clean features and labels)
        device: torch.device
        args: Training arguments
        test_loader: Optional test data loader
        
    Returns:
        dict: Training results and checkpoint paths
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    # Create training directory
    training_dir = create_receiver_training_directory(args.checkpoint_dir)
    
    # Training metrics history
    history = {
        'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [],
        'snr_values': [], 'learning_rates': []
    }
    
    best_acc = 0.0
    best_checkpoint = None
    
    print(f"Starting receiver training for {args.epochs} epochs...")
    print(f"SNR range: {args.snr_min} - {args.snr_max} dB")
    print(f"Communication mode: {args.communication_mode}")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        epoch_snrs = []
        
        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            # Generate random SNR for this batch (direct input as required)
            snr_db = np.random.uniform(args.snr_min, args.snr_max)
            epoch_snrs.append(snr_db)
            
            # Simulate channel effects on the clean features
            # In real deployment, this would be actual received features
            received_features = simulate_channel_effects(data, snr_db, args.communication_mode)
            
            optimizer.zero_grad()
            
            # Forward pass through receiver
            result = model(received_features, snr=snr_db, epoch=epoch)
            
            # Calculate loss
            loss = F.cross_entropy(result['classification_output'], target)
            
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            pred = result['classification_output'].argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%',
                'SNR': f'{snr_db:.1f}dB'
            })
        
        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct / total
        
        print(f'Train Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}')
        
        # Test evaluation (if test_loader provided)
        test_accuracy = 0.0
        test_loss = 0.0
        if test_loader is not None:
            model.eval()
            test_correct = 0
            test_total = 0
            total_test_loss = 0.0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    
                    # Use middle SNR for testing
                    test_snr = (args.snr_min + args.snr_max) / 2
                    
                    # Simulate channel effects
                    received_features = simulate_channel_effects(data, test_snr, args.communication_mode)
                    
                    result = model(received_features, snr=test_snr, epoch=epoch)
                    
                    loss = F.cross_entropy(result['classification_output'], target)
                    total_test_loss += loss.item()
                    
                    pred = result['classification_output'].argmax(dim=1, keepdim=True)
                    test_correct += pred.eq(target.view_as(pred)).sum().item()
                    test_total += target.size(0)
            
            test_accuracy = test_correct / test_total
            test_loss = total_test_loss / len(test_loader)
            print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        
        # Update learning rate
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # Save metrics
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_accuracy)
        history['snr_values'].extend(epoch_snrs)
        history['learning_rates'].append(current_lr)
        
        # Save checkpoint
        metrics = {
            'train_loss': avg_train_loss,
            'train_acc': train_accuracy,
            'test_loss': test_loss,
            'test_acc': test_accuracy
        }
        
        checkpoint_path = save_receiver_checkpoint(
            model, optimizer, epoch, metrics, training_dir
        )
        
        # Save best model
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            best_checkpoint = save_receiver_checkpoint(
                model, optimizer, epoch, metrics, training_dir, is_best=True
            )
            print(f"New best accuracy: {best_acc:.4f}")
        
        # Save metrics periodically
        if epoch % 10 == 0:
            save_receiver_metrics(history, training_dir)
            plot_receiver_training_progress(history, training_dir)
    
    # Final save
    save_receiver_metrics(history, training_dir)
    plot_receiver_training_progress(history, training_dir)
    
    print(f"\nReceiver training completed!")
    print(f"Best test accuracy: {best_acc:.4f}")
    print(f"Training directory: {training_dir}")
    
    return {
        'training_dir': training_dir,
        'best_checkpoint': best_checkpoint,
        'final_checkpoint': checkpoint_path,
        'best_accuracy': best_acc,
        'history': history
    }


if __name__ == "__main__":
    # This allows the training script to be run directly
    print("Receiver training module - use receiver_main.py to run training")