"""
Sender Training Module for DHSCC

This module implements independent training for the sender component,
including the quantizer as required by the refactoring specification.

Key features:
- Independent sender training with quantizer integration
- Direct SNR input during training
- Three-phase training strategy adapted for sender-only
- Model checkpointing and metrics tracking
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import time

from models.sender_model import SenderModel
from utils.file_io import ensure_dir
from utils.signal_processing import set_seed
import torch.nn.functional as F

def create_sender_training_directory(base_dir="sender_checkpoints"):
    """
    Create timestamped directory for sender training outputs.
    
    Args:
        base_dir: Base directory for checkpoints
        
    Returns:
        str: Path to created training directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_dir = os.path.join(base_dir, f"sender_training_{timestamp}")
    ensure_dir(training_dir)
    
    # Create subdirectories
    subdirs = ['models', 'metrics', 'plots', 'logs']
    for subdir in subdirs:
        ensure_dir(os.path.join(training_dir, subdir))
    
    print(f"📁 Created sender training directory: {training_dir}")
    return training_dir


def save_sender_metrics(history, save_dir):
    """Save training metrics to files."""
    metrics_file = os.path.join(save_dir, "metrics", "training_metrics.pkl")
    with open(metrics_file, 'wb') as f:
        pickle.dump(history, f)
    print(f"Metrics saved to {metrics_file}")


def plot_sender_training_progress(history, save_dir):
    """Plot and save training progress."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Sender Training Progress', fontsize=16)
    
    # Loss plot
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Total Loss')
    axes[0, 0].plot(epochs, history['classification_loss'], 'r-', label='Classification Loss')
    axes[0, 0].plot(epochs, history['commitment_loss'], 'g-', label='Commitment Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy plot
    if 'train_acc' in history and any(acc > 0 for acc in history['train_acc']):
        axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        if 'test_acc' in history:
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
    
    # Phase indicators
    axes[1, 1].plot(epochs, history['phases'], 'k-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Training Phase')
    axes[1, 1].set_title('Training Phases')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_file = os.path.join(save_dir, "plots", "sender_training_progress.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training plot saved to {plot_file}")


def save_sender_checkpoint(model, optimizer, epoch, phase, metrics, save_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'phase': phase,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': {
            'comm_mode': model.comm_mode,
            'hidden_channel': model.hidden_channel,
            'num_classes': model.num_classes
        }
    }
    
    # Save quantizer state if available
    if hasattr(model, 'quantizer') and model.quantizer is not None:
        checkpoint['quantizer_state'] = model.quantizer.state_dict()
    
    filename = f"sender_checkpoint_epoch_{epoch}.pth"
    if is_best:
        filename = f"sender_best_model.pth"
    
    filepath = os.path.join(save_dir, "models", filename)
    torch.save(checkpoint, filepath)
    print(f"Sender checkpoint saved: {filepath}")
    return filepath


def train_sender(model, train_loader, device, args, test_loader=None):
    """
    Main sender training function with three-phase strategy.
    
    Args:
        model: SenderModel instance
        train_loader: Training data loader
        device: torch.device
        args: Training arguments
        test_loader: Optional test data loader
        
    Returns:
        dict: Training results and checkpoint paths
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    # Create training directory
    training_dir = create_sender_training_directory(args.checkpoint_dir)
    
    # Training metrics history
    history = {
        'train_loss': [], 'classification_loss': [], 'commitment_loss': [],
        'train_acc': [], 'test_acc': [], 'snr_values': [], 'phases': []
    }
    
    best_acc = 0.0
    best_checkpoint = None
    
    print(f"Starting sender training for {args.epochs} epochs...")
    print(f"SNR range: {args.snr_min} - {args.snr_max} dB")
    print(f"Communication mode: {args.communication_mode}")
    
    for epoch in range(1, args.epochs + 1):
        # Determine training phase
        if epoch <= args.pretrain_epochs:
            phase = "pretrain"
            phase_num = 1
        elif epoch <= args.selector_epochs:
            phase = "selector"  # For sender, this focuses on quantizer optimization
            phase_num = 2
        else:
            phase = "final"
            phase_num = 3
            
        print(f"\nEpoch {epoch}/{args.epochs} - Phase: {phase}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_classification_loss = 0.0
        train_commitment_loss = 0.0
        correct = 0
        total = 0
        epoch_snrs = []
        
        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            # Generate random SNR for this batch (direct input as required)
            snr_db = np.random.uniform(args.snr_min, args.snr_max)
            epoch_snrs.append(snr_db)
            
            optimizer.zero_grad()
            
            # Forward pass through sender
            result = model(data, snr=snr_db, epoch=epoch)
            
            # Calculate losses
            classification_loss = F.cross_entropy(result['pre_task_logits'], target)
            commitment_loss = result['commitment_loss']
            
            # Phase-specific loss weighting
            if phase == "pretrain":
                # Focus on classification and basic feature learning
                total_loss = classification_loss + 0.1 * commitment_loss
            elif phase == "selector":
                # Focus on quantizer optimization (selector equivalent for sender)
                total_loss = classification_loss + 0.5 * commitment_loss
            else:  # final
                # Balanced training
                total_loss = classification_loss + 0.25 * commitment_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += total_loss.item()
            train_classification_loss += classification_loss.item()
            train_commitment_loss += commitment_loss.item()
            
            # Calculate accuracy from pre-task logits
            pred = result['pre_task_logits'].argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%',
                'SNR': f'{snr_db:.1f}dB'
            })
        
        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_classification_loss = train_classification_loss / len(train_loader)
        avg_commitment_loss = train_commitment_loss / len(train_loader)
        train_accuracy = correct / total
        
        print(f'Train Loss: {avg_train_loss:.4f}, Classification: {avg_classification_loss:.4f}, '
              f'Commitment: {avg_commitment_loss:.4f}, Accuracy: {train_accuracy:.4f}')
        
        # Test evaluation (if test_loader provided)
        test_accuracy = 0.0
        if test_loader is not None:
            model.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    
                    # Use middle SNR for testing
                    test_snr = (args.snr_min + args.snr_max) / 2
                    result = model(data, snr=test_snr, epoch=epoch)
                    
                    pred = result['pre_task_logits'].argmax(dim=1, keepdim=True)
                    test_correct += pred.eq(target.view_as(pred)).sum().item()
                    test_total += target.size(0)
            
            test_accuracy = test_correct / test_total
            print(f'Test Accuracy: {test_accuracy:.4f}')
        
        # Update learning rate
        scheduler.step()
        
        # Save metrics
        history['train_loss'].append(avg_train_loss)
        history['classification_loss'].append(avg_classification_loss)
        history['commitment_loss'].append(avg_commitment_loss)
        history['train_acc'].append(train_accuracy)
        history['test_acc'].append(test_accuracy)
        history['snr_values'].extend(epoch_snrs)
        history['phases'].append(phase_num)
        
        # Save checkpoint
        metrics = {
            'train_loss': avg_train_loss,
            'train_acc': train_accuracy,
            'test_acc': test_accuracy,
            'phase': phase
        }
        
        checkpoint_path = save_sender_checkpoint(
            model, optimizer, epoch, phase, metrics, training_dir
        )
        
        # Save best model
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            best_checkpoint = save_sender_checkpoint(
                model, optimizer, epoch, phase, metrics, training_dir, is_best=True
            )
            print(f"New best accuracy: {best_acc:.4f}")
        
        # Save metrics periodically
        if epoch % 10 == 0:
            save_sender_metrics(history, training_dir)
            plot_sender_training_progress(history, training_dir)
    
    # Final save
    save_sender_metrics(history, training_dir)
    plot_sender_training_progress(history, training_dir)
    
    print(f"\nSender training completed!")
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
    print("Sender training module - use sender_main.py to run training")