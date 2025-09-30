"""
End-to-End Training Module for DHSCC Split System

This module implements end-to-end training for the sender-receiver communication system.
It trains the complete pipeline including modulation, channel transmission, and demodulation.

Training phases (mimicking dhscc_train.py):
1. Pretrain: Train sender network without channel transmission
2. Selector: Joint training of sender and receiver with channel
3. Final: Fine-tuning of the complete system
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
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sender.sender_model import SenderNet
from receiver.receiver_model import ReceiverNet
from sender.quantizer import VibrationVectorQuantizer
from channel_modem.channel_modem import ChannelModem
from sender.modulation import Modulator
from receiver.demodulation import Demodulator
from sender.polar_encoding import PolarEncoder
from receiver.polar_decoding import PolarSCDecoder
from utils.signal_processing import psnr_to_noise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_training_directory(base_dir="checkpoints"):
    """
    Create timestamped directory for training outputs.

    Args:
        base_dir: Base directory for checkpoints

    Returns:
        str: Path to created training directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_dir = os.path.join(base_dir, f"end_to_end_training_{timestamp}")
    os.makedirs(training_dir, exist_ok=True)

    # Create subdirectories
    subdirs = ['pretrain', 'selector', 'final', 'plots', 'logs']
    for subdir in subdirs:
        os.makedirs(os.path.join(training_dir, subdir), exist_ok=True)

    print(f"📁 Created training directory: {training_dir}")
    return training_dir


def save_training_metrics(history, save_dir):
    """
    Save training metrics and create visualizations.

    Args:
        history: Dictionary containing training history
        save_dir: Directory to save plots and metrics
    """
    # Save raw metrics
    metrics_path = os.path.join(save_dir, 'logs', 'training_metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(history, f)

    # Convert tensors to CPU numpy arrays for plotting
    for key in history:
        if isinstance(history[key], list) and history[key] and hasattr(history[key][0], 'cpu'):
            history[key] = [x.detach().cpu().numpy() if hasattr(x, 'cpu') else x for x in history[key]]

    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot 1: Training and test loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    axes[0, 0].set_title('Training and Test Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Training and test accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['test_acc'], 'r-', label='Test Acc', linewidth=2)
    axes[0, 1].set_title('Training and Test Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: BER curves
    axes[0, 2].plot(epochs, history['train_ber'], 'b-', label='Train BER', linewidth=2)
    axes[0, 2].plot(epochs, history['test_ber'], 'r-', label='Test BER', linewidth=2)
    axes[0, 2].set_title('Bit Error Rate (BER)')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('BER')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_yscale('log')

    # Plot 4: Modulation loss
    if 'modulation_loss' in history:
        axes[1, 0].plot(epochs, history['modulation_loss'], 'g-', label='Modulation Loss', linewidth=2)
        axes[1, 0].set_title('Modulation Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Channel SNR
    if 'channel_snr' in history:
        axes[1, 1].plot(epochs, history['channel_snr'], 'orange', label='Channel SNR', linewidth=2)
        axes[1, 1].set_title('Channel SNR')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('SNR (dB)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Training phases
    phase_changes = []
    current_phase = history.get('phases', [0] * len(epochs))[0]
    for i, phase in enumerate(history.get('phases', [])):
        if phase != current_phase:
            phase_changes.append((i, current_phase, phase))
            current_phase = phase

    colors = {'pretrain': 'blue', 'selector': 'orange', 'final': 'green'}
    start = 0
    for end, phase, _ in phase_changes + [(len(epochs), current_phase, None)]:
        if phase in colors:
            axes[1, 2].axvspan(start + 1, end + 1, alpha=0.3, color=colors[phase], label=phase.replace('_', ' ').title())
        start = end

    axes[1, 2].plot(epochs, history['test_acc'], 'k-', linewidth=2, label='Test Accuracy')
    axes[1, 2].set_title('Training Phases')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Test Accuracy (%)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plots
    plot_path = os.path.join(save_dir, 'plots', 'end_to_end_training_progress.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()

    print(f"📈 Saved training plots to: {plot_path}")


def train_end_to_end(sender_model, receiver_model, quantizer, channel_modem, train_loader, device, args, test_loader=None):
    """
    End-to-end training function for sender-receiver system.

    Args:
        sender_model: SenderNet model
        receiver_model: ReceiverNet model
        channel_modem: ChannelModem for transmission
        train_loader: DataLoader for training
        device: torch.device
        args: argparse.Namespace with training parameters
        test_loader: DataLoader for testing (optional)

    Returns:
        dict: Training results and checkpoints
    """
    # Setup optimizers
    sender_optimizer = torch.optim.Adam(sender_model.parameters(), lr=args.lr)
    receiver_optimizer = torch.optim.Adam(receiver_model.parameters(), lr=args.lr)
    scheduler = StepLR(sender_optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # Create training directory
    training_dir = create_training_directory(args.checkpoint_dir)

    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_ber': [],
        'test_loss': [], 'test_acc': [], 'test_ber': [],
        'modulation_loss': [], 'channel_snr': [], 'phases': []
    }

    # Training phases configuration (mimicking dhscc_train.py)
    pretrain_epochs = getattr(args, 'pretrain_epochs', 50)
    selector_epochs = getattr(args, 'selector_epochs', 80) 
    final_epochs = getattr(args, 'final_epochs', 100)
    
    # Use epochs from args if specified, otherwise use final_epochs
    total_epochs = getattr(args, 'epochs', final_epochs)

    best_acc = 0
    best_checkpoint = None

    for epoch in range(total_epochs):
        # Determine current phase (mimicking dhscc_train.py)
        if epoch < pretrain_epochs:
            phase = 'pretrain'
            use_coding = False  # Disable polar coding in pretrain
            # Pretrain: train all components end-to-end without coding
            for param in sender_model.parameters():
                param.requires_grad = True
            for param in receiver_model.parameters():
                param.requires_grad = True
        elif epoch < selector_epochs:
            phase = 'selector'
            use_coding = True   # Enable polar coding in selector phase
            # Selector: freeze sender, train receiver with adaptive coding
            for param in sender_model.parameters():
                param.requires_grad = False
            for param in receiver_model.parameters():
                param.requires_grad = True
        else:
            phase = 'final'
            use_coding = True   # Enable polar coding in final phase
            # Final: unfreeze all, fine-tune the complete system
            for param in sender_model.parameters():
                param.requires_grad = True
            for param in receiver_model.parameters():
                param.requires_grad = True

        history['phases'].append(phase)

        # Training loop
        sender_model.train()
        receiver_model.train()

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_ber = 0.0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} ({phase})")):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            total_samples += batch_size

            # Random SNR selection for training (similar to dhscc_train.py)
            if hasattr(args, 'snr_min') and hasattr(args, 'snr_max'):
                snr_db = torch.rand(1).item() * (args.snr_max - args.snr_min) + args.snr_min
            else:
                snr_db = getattr(args, 'channel_snr', 3.0)
            
            # Forward pass through sender
            # channel_noise = torch.randn(batch_size, 1).to(device) * 0.1 + 1.0  # Old fixed noise
            # Use SNR-dependent noise level
            channel_noise = psnr_to_noise(torch.tensor(snr_db, device=device)).unsqueeze(0).expand(batch_size, -1)
            sender_output = sender_model(data, channel_noise)

            features, importance_weights, pre_task_logits, feature_shape, original_encoded_features = sender_output
            
            # Quantize features for transmission (all phases)
            quantized_features, quantizer_indices, vq_loss = quantizer.encode(features)
            
            # Annealing mechanism - gradually introduce transmit loss (same as dhscc_train.py)
            anneal_ratio = min(1.0, epoch / 20.0)
            
            # Use channel_modem for transmission with polar coding integration (all phases)
            if hasattr(channel_modem, 'transmit_with_polar_coding'):
                # Set group_to_use based on phase (adaptive selection simulation)
                if phase == 'pretrain':
                    group_to_use = 8  # Use all groups in pretrain
                elif phase == 'selector':
                    group_to_use = 6  # Reduce groups in selector phase
                    if batch_idx == 0:  # Only print for first batch of each epoch
                        print(f"🔄 SELECTOR PHASE: Using {group_to_use} groups out of {quantizer_indices.shape[1]} available")
                        print(f"   Quantizer indices shape: {quantizer_indices.shape}")
                        print(f"   SNR: {snr_db:.1f}dB, Use coding: {use_coding}")
                else:  # final
                    group_to_use = 4  # Minimal groups in final phase
                
                received_features, ber, transmit_mse = channel_modem.transmit_with_polar_coding(
                    quantized_features, quantizer_indices, 
                    snr_db=snr_db,
                    use_coding=use_coding,
                    group_to_use=group_to_use
                )
            else:
                # Fallback: simple transmission without proper coding
                received_features = channel_modem.transmit(quantized_features, snr_db)
                ber = torch.tensor(0.01, device=device)  # Placeholder BER
                transmit_mse = torch.tensor(0.0, device=device)
            
            # Receiver processing (all phases)
            receiver_output = receiver_model(received_features, channel_noise, feature_shape)

            # Calculate losses (same formula as dhscc_train.py for all phases)
            classification_loss = nn.CrossEntropyLoss()(receiver_output, target)
            vq_weight = getattr(args, 'vq_loss_weight', 0.1)
            total_loss = classification_loss + vq_weight * vq_loss + 0.1 * anneal_ratio * transmit_mse
            
            # Calculate metrics
            predictions = receiver_output.argmax(dim=1)
            acc = (predictions == target).float().mean().item() * 100

            # Backward pass (gradients computed for unfrozen parameters only)
            sender_optimizer.zero_grad()
            receiver_optimizer.zero_grad()
            total_loss.backward()
            sender_optimizer.step()
            receiver_optimizer.step()

            loss = total_loss.item()

            epoch_loss += loss * batch_size
            epoch_acc += acc * batch_size
            if isinstance(ber, torch.Tensor):
                ber = ber.item()
            epoch_ber += ber * batch_size

        # Average metrics
        epoch_loss /= total_samples
        epoch_acc /= total_samples
        epoch_ber /= total_samples

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['train_ber'].append(epoch_ber)
        history['modulation_loss'].append(0.0)  # Placeholder for modulation loss
        history['channel_snr'].append(args.channel_snr)

        # Validation
        if test_loader is not None:
            test_loss, test_acc, test_ber = validate_end_to_end(sender_model, receiver_model, quantizer, channel_modem, test_loader, device, args, use_coding=use_coding)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['test_ber'].append(test_ber)

            print(f"Epoch {epoch+1}/{total_epochs} ({phase}) - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%, BER: {test_ber:.6f}")

            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                best_checkpoint = save_checkpoint(sender_model, receiver_model, quantizer, sender_optimizer, receiver_optimizer,
                                                epoch, phase, {'acc': test_acc, 'loss': test_loss, 'ber': test_ber},
                                                training_dir, is_best=True)
        else:
            print(f"Epoch {epoch+1}/{total_epochs} ({phase}) - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, BER: {epoch_ber:.6f}")

        scheduler.step()

    # Save training metrics
    save_training_metrics(history, training_dir)
    
    # Save model components separately for independent deployment
    save_model_components(sender_model, receiver_model, quantizer, training_dir, prefix="final_")

    return {
        'training_dir': training_dir,
        'best_checkpoint': best_checkpoint,
        'final_metrics': {
            'train_acc': history['train_acc'][-1],
            'test_acc': history['test_acc'][-1] if test_loader else None,
            'ber': history['test_ber'][-1] if test_loader else history['train_ber'][-1]
        },
        'history': history
    }


def validate_end_to_end(sender_model, receiver_model, quantizer, channel_modem, test_loader, device, args, use_coding=True):
    """
    Validation function for end-to-end system with multiple SNR testing (similar to dhscc_train.py).

    Args:
        sender_model: SenderNet model
        receiver_model: ReceiverNet model
        quantizer: VibrationVectorQuantizer
        channel_modem: ChannelModem
        test_loader: DataLoader for testing
        device: torch.device
        args: Training arguments
        use_coding: Whether to use polar coding in validation

    Returns:
        tuple: (avg_loss, avg_accuracy, avg_ber)
    """
    sender_model.eval()
    receiver_model.eval()

    # Test multiple SNRs and average results (similar to dhscc_train.py)
    snr_levels = []
    if hasattr(args, 'snr_min') and hasattr(args, 'snr_max'):
        snr_levels = [args.snr_min, (args.snr_min + args.snr_max) / 2, args.snr_max]
    else:
        snr_levels = [getattr(args, 'channel_snr', 3.0)]

    all_losses = []
    all_accs = []
    all_bers = []

    for snr_db in snr_levels:
        total_loss = 0.0
        total_acc = 0.0
        total_ber = 0.0
        total_samples = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                batch_size = data.size(0)
                total_samples += batch_size

                # Forward pass
                # channel_noise = torch.randn(batch_size, 1).to(device) * 0.1 + 1.0  # Old fixed noise
                # Use SNR-dependent noise level for consistency
                channel_noise = psnr_to_noise(torch.tensor(snr_db, device=device)).unsqueeze(0).expand(batch_size, -1)
                sender_output = sender_model(data, channel_noise)
                features, importance_weights, pre_task_logits, feature_shape, original_encoded_features = sender_output
                
                # Quantize features for transmission
                quantized_features, quantizer_indices, vq_loss = quantizer.encode(features)
                
                # Use integrated polar coding transmission if available
                if hasattr(channel_modem, 'transmit_with_polar_coding'):
                    received_features, ber, transmit_mse = channel_modem.transmit_with_polar_coding(
                        quantized_features, quantizer_indices, 
                        snr_db=snr_db,
                        use_coding=use_coding,
                        group_to_use=8  # Default for validation
                    )
                else:
                    # Fallback: simple transmission
                    received_features = channel_modem.transmit(quantized_features, snr_db)
                    ber = 0.01  # Placeholder BER
                
                # Receiver processing
                receiver_output = receiver_model(received_features, channel_noise, feature_shape)

                # Calculate loss
                loss = nn.CrossEntropyLoss()(receiver_output, target)
                total_loss += loss.item() * batch_size

                # Calculate accuracy
                predictions = receiver_output.argmax(dim=1)
                acc = (predictions == target).float().mean().item() * 100
                total_acc += acc * batch_size

                # Add BER
                if isinstance(ber, torch.Tensor):
                    ber = ber.item()
                total_ber += ber * batch_size

        # Average for this SNR
        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        avg_ber = total_ber / total_samples

        all_losses.append(avg_loss)
        all_accs.append(avg_acc)
        all_bers.append(avg_ber)

    # Return averages across all SNR levels
    return sum(all_losses) / len(all_losses), sum(all_accs) / len(all_accs), sum(all_bers) / len(all_bers)


def save_checkpoint(sender_model, receiver_model, quantizer, sender_optimizer, receiver_optimizer, epoch, phase, metrics, save_dir, is_best=False):
    """
    Save model checkpoint.

    Args:
        sender_model: SenderNet model
        receiver_model: ReceiverNet model
        quantizer: VibrationVectorQuantizer
        sender_optimizer: Sender optimizer
        receiver_optimizer: Receiver optimizer
        epoch: Current epoch
        phase: Training phase
        metrics: Current metrics
        save_dir: Save directory
        is_best: Whether this is the best model

    Returns:
        str: Path to saved checkpoint
    """
    save_data = {
        'epoch': epoch,
        'phase': phase,
        'sender_state_dict': sender_model.state_dict(),
        'receiver_state_dict': receiver_model.state_dict(),
        'quantizer_state_dict': quantizer.state_dict(),
        'sender_optimizer_state_dict': sender_optimizer.state_dict(),
        'receiver_optimizer_state_dict': receiver_optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }

    filename = 'best_end_to_end_model.pth' if is_best else f'end_to_end_{phase}_epoch_{epoch}.pth'
    save_path = os.path.join(save_dir, phase, filename)
    torch.save(save_data, save_path)

    print(f"💾 Saved checkpoint: {filename}")
    return save_path


def save_model_components(sender_model, receiver_model, quantizer, save_dir, prefix=""):
    """
    Save model components separately for independent deployment.

    Args:
        sender_model: SenderNet model
        receiver_model: ReceiverNet model
        quantizer: VibrationVectorQuantizer
        save_dir: Directory to save components
        prefix: Prefix for filenames
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save sender model
    sender_path = os.path.join(save_dir, f"{prefix}sender_model.pth")
    torch.save({
        'state_dict': sender_model.state_dict(),
        'model_type': 'SenderNet',
        'timestamp': datetime.now().isoformat()
    }, sender_path)
    print(f"💾 Saved sender model: {sender_path}")
    
    # Save receiver model
    receiver_path = os.path.join(save_dir, f"{prefix}receiver_model.pth")
    torch.save({
        'state_dict': receiver_model.state_dict(),
        'model_type': 'ReceiverNet',
        'timestamp': datetime.now().isoformat()
    }, receiver_path)
    print(f"💾 Saved receiver model: {receiver_path}")
    
    # Save quantizer
    quantizer_path = os.path.join(save_dir, f"{prefix}quantizer.pth")
    torch.save({
        'state_dict': quantizer.state_dict(),
        'model_type': 'VibrationVectorQuantizer',
        'timestamp': datetime.now().isoformat()
    }, quantizer_path)
    print(f"💾 Saved quantizer: {quantizer_path}")


def load_model_components(sender_model, receiver_model, quantizer, load_dir, prefix=""):
    """
    Load model components separately.

    Args:
        sender_model: SenderNet model instance
        receiver_model: ReceiverNet model instance
        quantizer: VibrationVectorQuantizer instance
        load_dir: Directory to load components from
        prefix: Prefix for filenames
    """
    # Load sender model
    sender_path = os.path.join(load_dir, f"{prefix}sender_model.pth")
    if os.path.exists(sender_path):
        checkpoint = torch.load(sender_path, map_location=device)
        sender_model.load_state_dict(checkpoint['state_dict'])
        print(f"📁 Loaded sender model: {sender_path}")
    else:
        print(f"⚠️  Sender model not found: {sender_path}")
    
    # Load receiver model
    receiver_path = os.path.join(load_dir, f"{prefix}receiver_model.pth")
    if os.path.exists(receiver_path):
        checkpoint = torch.load(receiver_path, map_location=device)
        receiver_model.load_state_dict(checkpoint['state_dict'])
        print(f"📁 Loaded receiver model: {receiver_path}")
    else:
        print(f"⚠️  Receiver model not found: {receiver_path}")
    
    # Load quantizer
    quantizer_path = os.path.join(load_dir, f"{prefix}quantizer.pth")
    if os.path.exists(quantizer_path):
        checkpoint = torch.load(quantizer_path, map_location=device)
        quantizer.load_state_dict(checkpoint['state_dict'])
        print(f"📁 Loaded quantizer: {quantizer_path}")
    else:
        print(f"⚠️  Quantizer not found: {quantizer_path}")


if __name__ == "__main__":
    # Example usage
    print("End-to-End Training Module")
    print("Use this module by importing train_end_to_end function")