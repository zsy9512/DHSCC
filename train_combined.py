#!/usr/bin/env python3
"""
Combined Training Module for DHSCC Split System

This module implements two training approaches for the sender-receiver communication system:
1. Direct polar coding training: Direct training with polar coding + QPSK from the start
2. Two-stage training: First train without polar coding, then fine-tune with polar coding

Training approaches:
- Direct polar: Single phase with polar coding + QPSK for specified epochs
- Two-stage: Phase 1 (no polar coding) + Phase 2 (with polar coding)
- AWGN channel with SNR range 10-20dB
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
from channel.channel_modem import ChannelModem
from sender.modulation import WirelessTransmitter
from receiver.demodulation import WirelessReceiver
from sender.polar_encoding import PolarEncoder
from receiver.polar_decoding import PolarSCDecoder
from sender.utils_signal import psnr_to_noise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_training_directory(base_dir="checkpoints", prefix="training"):
    """
    Create timestamped directory for training outputs.

    Args:
        base_dir: Base directory for checkpoints
        prefix: Prefix for directory name

    Returns:
        str: Path to created training directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(training_dir, exist_ok=True)

    # Create subdirectories
    subdirs = ['plots', 'logs']
    for subdir in subdirs:
        os.makedirs(os.path.join(training_dir, subdir), exist_ok=True)

    print(f"📁 Created training directory: {training_dir}")
    return training_dir


def save_training_metrics(history, save_dir, title="Training Progress"):
    """
    Save training metrics and create visualizations.

    Args:
        history: Dictionary containing training history
        save_dir: Directory to save plots and metrics
        title: Title for the plot
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

    colors = {'stage1_no_polar': 'blue', 'stage2_with_polar': 'green', 'direct_polar': 'red'}
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

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    # Save plots
    plot_path = os.path.join(save_dir, 'plots', 'training_progress.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()

    print(f"📈 Saved training plots to: {plot_path}")


def validate_model(sender_model, receiver_model, quantizer, channel_modem, test_loader, device, args, use_polar_coding=True):
    """
    Validation function for the communication system.

    Args:
        sender_model: SenderNet model
        receiver_model: ReceiverNet model
        quantizer: Quantizer instance
        channel_modem: ChannelModem for transmission
        test_loader: DataLoader for testing
        device: torch.device
        args: arguments
        use_polar_coding: Whether to use polar coding in validation

    Returns:
        tuple: (loss, accuracy, ber)
    """
    sender_model.eval()
    receiver_model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_ber = 0.0
    total_samples = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            total_samples += batch_size

            # Random SNR selection for validation
            if hasattr(args, 'snr_min') and hasattr(args, 'snr_max'):
                snr_db = torch.rand(1).item() * (args.snr_max - args.snr_min) + args.snr_min
            else:
                snr_db = getattr(args, 'channel_snr', 15.0)

            # Forward pass through sender
            channel_noise = psnr_to_noise(torch.tensor(snr_db, device=device)).unsqueeze(0).expand(batch_size, -1)
            sender_output = sender_model(data, channel_noise)

            features, importance_weights, pre_task_logits, feature_shape, original_encoded_features = sender_output

            # Quantize features
            quantized_features, quantizer_indices, vq_loss = quantizer.encode(features)

            # Use channel transmission based on polar coding setting
            if use_polar_coding and hasattr(channel_modem, 'transmit_with_polar_coding'):
                group_to_use = getattr(args, 'max_groups', 8)
                received_features, ber, transmit_mse = channel_modem.transmit_with_polar_coding(
                    quantized_features, quantizer_indices,
                    snr_db=snr_db,
                    use_coding=True,
                    group_to_use=group_to_use,
                    quantizer=quantizer
                )
            else:
                # Digital transmission without polar coding
                received_features = channel_modem.transmit(quantized_features, snr_db)
                ber = torch.tensor(0.01, device=device)  # Placeholder BER for non-polar mode
                transmit_mse = torch.tensor(0.0, device=device)

            # Receiver processing
            receiver_output = receiver_model(received_features, channel_noise, feature_shape)

            # Calculate losses
            classification_loss = nn.CrossEntropyLoss()(receiver_output, target)
            vq_weight = getattr(args, 'vq_loss_weight', 0.1)
            total_loss_batch = classification_loss + vq_weight * vq_loss + 0.1 * transmit_mse

            # Calculate metrics
            predictions = receiver_output.argmax(dim=1)
            acc = (predictions == target).float().mean().item() * 100

            total_loss += total_loss_batch.item() * batch_size
            total_acc += acc * batch_size
            if isinstance(ber, torch.Tensor):
                ber = ber.item()
            total_ber += ber * batch_size

    # Average metrics
    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    avg_ber = total_ber / total_samples

    return avg_loss, avg_acc, avg_ber


def save_checkpoint(sender_model, receiver_model, quantizer, sender_optimizer, receiver_optimizer,
                   epoch, phase, metrics, training_dir, is_best=False):
    """
    Save model checkpoint.

    Args:
        sender_model: SenderNet model
        receiver_model: ReceiverNet model
        quantizer: Quantizer instance
        sender_optimizer: Sender optimizer
        receiver_optimizer: Receiver optimizer
        epoch: Current epoch
        phase: Training phase
        metrics: Performance metrics
        training_dir: Training directory
        is_best: Whether this is the best checkpoint

    Returns:
        str: Path to saved checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'phase': phase,
        'sender_model_state_dict': sender_model.state_dict(),
        'receiver_model_state_dict': receiver_model.state_dict(),
        'quantizer_state_dict': quantizer.state_dict(),
        'sender_optimizer_state_dict': sender_optimizer.state_dict(),
        'receiver_optimizer_state_dict': receiver_optimizer.state_dict(),
        'metrics': metrics
    }

    if is_best:
        checkpoint_path = os.path.join(training_dir, 'best_model.pth')
    else:
        checkpoint_path = os.path.join(training_dir, f'checkpoint_epoch_{epoch}.pth')

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def save_model_components(sender_model, receiver_model, quantizer, training_dir, prefix="final_"):
    """
    Save individual model components for independent deployment.

    Args:
        sender_model: SenderNet model
        receiver_model: ReceiverNet model
        quantizer: Quantizer instance
        training_dir: Training directory
        prefix: Prefix for saved files
    """
    # Save sender model
    sender_path = os.path.join(training_dir, f'{prefix}sender_model.pth')
    torch.save(sender_model.state_dict(), sender_path)
    print(f"💾 Saved sender model: {sender_path}")

    # Save receiver model
    receiver_path = os.path.join(training_dir, f'{prefix}receiver_model.pth')
    torch.save(receiver_model.state_dict(), receiver_path)
    print(f"💾 Saved receiver model: {receiver_path}")

    # Save quantizer
    quantizer_path = os.path.join(training_dir, f'{prefix}quantizer.pth')
    torch.save(quantizer.state_dict(), quantizer_path)
    print(f"💾 Saved quantizer: {quantizer_path}")


def train_direct_polar(sender_model, receiver_model, quantizer, channel_modem, train_loader, device, args, test_loader=None):
    """
    Direct training with polar coding from the start.

    Args:
        sender_model: SenderNet model
        receiver_model: ReceiverNet model
        quantizer: Quantizer instance
        channel_modem: ChannelModem for transmission
        train_loader: DataLoader for training
        device: torch.device
        args: argparse.Namespace with training parameters
        test_loader: DataLoader for testing (optional)

    Returns:
        dict: Training results and checkpoints
    """
    print("🚀 Starting Direct Polar Coding Training")

    # Setup optimizers
    sender_optimizer = torch.optim.Adam(sender_model.parameters(), lr=args.lr)
    receiver_optimizer = torch.optim.Adam(receiver_model.parameters(), lr=args.lr)
    scheduler = StepLR(sender_optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # Create training directory
    training_dir = create_training_directory(args.checkpoint_dir, "direct_polar_training")

    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_ber': [], 'train_transmit_mse': [],
        'test_loss': [], 'test_acc': [], 'test_ber': [],
        'modulation_loss': [], 'channel_snr': [], 'phases': []
    }

    # Training parameters
    total_epochs = getattr(args, 'epochs', 120)
    best_acc = 0
    best_checkpoint = None

    for epoch in range(total_epochs):
        phase = 'direct_polar'
        history['phases'].append(phase)

        # Training loop
        sender_model.train()
        receiver_model.train()

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_ber = 0.0
        epoch_transmit_mse = 0.0
        epoch_classification_loss = 0.0
        epoch_vq_loss = 0.0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} ({phase})")):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            total_samples += batch_size

            # Random SNR selection for training
            if hasattr(args, 'snr_min') and hasattr(args, 'snr_max'):
                snr_db = torch.rand(1).item() * (args.snr_max - args.snr_min) + args.snr_min
            else:
                snr_db = getattr(args, 'channel_snr', 15.0)

            # Forward pass through sender
            channel_noise = psnr_to_noise(torch.tensor(snr_db, device=device)).unsqueeze(0).expand(batch_size, -1)
            sender_output = sender_model(data, channel_noise)

            features, importance_weights, pre_task_logits, feature_shape, original_encoded_features = sender_output

            # Quantize features
            quantized_features, quantizer_indices, vq_loss = quantizer.encode(features)

            # Annealing mechanism - gradually introduce transmit loss
            anneal_ratio = min(1.0, epoch / 20.0)

            # Always use polar coding
            if hasattr(channel_modem, 'transmit_with_polar_coding'):
                group_to_use = getattr(args, 'max_groups', 8)
                received_features, ber, transmit_mse = channel_modem.transmit_with_polar_coding(
                    quantized_features, quantizer_indices,
                    snr_db=snr_db,
                    use_coding=True,
                    group_to_use=group_to_use,
                    quantizer=quantizer
                )
            else:
                received_features = channel_modem.transmit(quantized_features, snr_db)
                ber = torch.tensor(0.01, device=device)
                transmit_mse = torch.tensor(0.0, device=device)
           
            # Receiver processing
            received_features = quantized_features + (received_features - quantized_features).detach()
            receiver_output = receiver_model(received_features, channel_noise, feature_shape)
            # Calculate losses
            classification_loss = nn.CrossEntropyLoss()(receiver_output, target)
            transmit_mse = torch.nn.MSELoss()(received_features, quantized_features)
            vq_weight = getattr(args, 'vq_loss_weight', 0.1)
            #print(f"VQ Loss: {vq_loss.item():.6f}, Transmit MSE: {transmit_mse.item():.6f},classification_loss: {classification_loss.item():.6f}")
            total_loss = classification_loss + vq_weight *  anneal_ratio * (vq_loss +transmit_mse)

            # Calculate metrics
            predictions = receiver_output.argmax(dim=1)
            acc = (predictions == target).float().mean().item() * 100

            # Backward pass
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
            epoch_classification_loss += classification_loss.item() * batch_size
            epoch_vq_loss += vq_loss.item() * batch_size

        # Average metrics
        epoch_loss /= total_samples
        epoch_acc /= total_samples
        epoch_ber /= total_samples
        epoch_transmit_mse /= total_samples
        epoch_classification_loss /= total_samples
        epoch_vq_loss /= total_samples

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['train_ber'].append(epoch_ber)
        history['modulation_loss'].append(0.0)
        history['channel_snr'].append(args.channel_snr)

        # Validation
        if test_loader is not None:
            test_loss, test_acc, test_ber = validate_model(sender_model, receiver_model, quantizer, channel_modem, test_loader, device, args, use_polar_coding=True)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['test_ber'].append(test_ber)

            print(f"Epoch {epoch+1}/{total_epochs} ({phase}) - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%, BER: {test_ber:.6f}, Transmit MSE: {epoch_transmit_mse:.8f}, Class Loss: {epoch_classification_loss:.6f}, VQ Loss: {epoch_vq_loss:.6f}")

            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                best_checkpoint = save_checkpoint(sender_model, receiver_model, quantizer, sender_optimizer, receiver_optimizer,
                                                epoch, phase, {'acc': test_acc, 'loss': test_loss, 'ber': test_ber},
                                                training_dir, is_best=True)
        else:
            print(f"Epoch {epoch+1}/{total_epochs} ({phase}) - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, BER: {epoch_ber:.6f}, Transmit MSE: {epoch_transmit_mse:.6f}, Class Loss: {epoch_classification_loss:.6f}, VQ Loss: {epoch_vq_loss:.6f}")

        scheduler.step()

    # Save training metrics
    save_training_metrics(history, training_dir, "Direct Polar Coding Training Progress")

    # Save model components
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


def train_two_stage(sender_model, receiver_model, quantizer, channel_modem, train_loader, device, args, test_loader=None):
    """
    Two-stage training: Stage 1 without polar coding, Stage 2 with polar coding.

    Args:
        sender_model: SenderNet model
        receiver_model: ReceiverNet model
        quantizer: Quantizer instance
        channel_modem: ChannelModem for transmission
        train_loader: DataLoader for training
        device: torch.device
        args: argparse.Namespace with training parameters
        test_loader: DataLoader for testing (optional)

    Returns:
        dict: Training results and checkpoints
    """
    print("🚀 Starting Two-Stage Training")

    # Create training directory
    training_dir = create_training_directory(args.checkpoint_dir, "two_stage_training")

    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_ber': [], 'train_transmit_mse': [],
        'test_loss': [], 'test_acc': [], 'test_ber': [],
        'modulation_loss': [], 'channel_snr': [], 'phases': []
    }

    # Stage 1: Train without polar coding
    print("\n--- Stage 1: Training without polar coding ---")
    stage1_epochs = getattr(args, 'pretrain_epochs', 60)  # Default 60 epochs for stage 1

    sender_optimizer = torch.optim.Adam(sender_model.parameters(), lr=args.lr)
    receiver_optimizer = torch.optim.Adam(receiver_model.parameters(), lr=args.lr)
    scheduler = StepLR(sender_optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    best_acc_stage1 = 0
    best_checkpoint_stage1 = None

    for epoch in range(stage1_epochs):
        phase = 'stage1_no_polar'
        history['phases'].append(phase)

        # Training loop
        sender_model.train()
        receiver_model.train()

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_ber = 0.0
        epoch_transmit_mse = 0.0
        epoch_classification_loss = 0.0
        epoch_vq_loss = 0.0
        total_samples = 0
        transmit_mse = torch.tensor(0.0, device=device)
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Stage 1 - Epoch {epoch+1}/{stage1_epochs}")):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            total_samples += batch_size

            # Random SNR selection for training
            if hasattr(args, 'snr_min') and hasattr(args, 'snr_max'):
                snr_db = torch.rand(1).item() * (args.snr_max - args.snr_min) + args.snr_min
            else:
                snr_db = getattr(args, 'channel_snr', 15.0)

            # Forward pass through sender
            channel_noise = psnr_to_noise(torch.tensor(snr_db, device=device)).unsqueeze(0).expand(batch_size, -1)
            sender_output = sender_model(data, channel_noise)

            features, importance_weights, pre_task_logits, feature_shape, original_encoded_features = sender_output

            # Quantize features
            quantized_features, quantizer_indices, vq_loss = quantizer.encode(features)

            # Annealing mechanism - gradually introduce transmit loss
            anneal_ratio = min(1.0, epoch / 20.0)

            # Stage 1: NO polar coding
            received_features = channel_modem.transmit(quantized_features, snr_db)
            ber = torch.tensor(0.01, device=device)  # Placeholder BER for non-polar mode
            
            received_features = quantized_features + (received_features - quantized_features).detach()
            # Receiver processing
            receiver_output = receiver_model(received_features, channel_noise, feature_shape)
            
            transmit_mse = torch.nn.MSELoss()(received_features, quantized_features)
            
            # Calculate losses
            classification_loss = nn.CrossEntropyLoss()(receiver_output, target)
            vq_weight = getattr(args, 'vq_loss_weight', 0.1)
            #print(f"VQ Loss: {vq_loss.item():.6f}, Transmit MSE: {transmit_mse.item():.6f},classification_loss: {classification_loss.item():.6f}")
            total_loss = classification_loss + vq_weight *  anneal_ratio * (vq_loss +transmit_mse)

            # Calculate metrics
            predictions = receiver_output.argmax(dim=1)
            acc = (predictions == target).float().mean().item() * 100

            # Backward pass
            sender_optimizer.zero_grad()
            receiver_optimizer.zero_grad()
            total_loss.backward()
            sender_optimizer.step()
            receiver_optimizer.step()

            loss = total_loss.item()

            epoch_loss += loss * batch_size
            epoch_acc += acc * batch_size
            epoch_ber += ber.item() * batch_size
            epoch_transmit_mse += transmit_mse.item() * batch_size
            epoch_classification_loss += classification_loss.item() * batch_size
            epoch_vq_loss += vq_loss.item() * batch_size

        # Average metrics
        epoch_loss /= total_samples
        epoch_acc /= total_samples
        epoch_ber /= total_samples
        epoch_transmit_mse /= total_samples
        epoch_classification_loss /= total_samples
        epoch_vq_loss /= total_samples

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['train_ber'].append(epoch_ber)
        history['train_transmit_mse'].append(epoch_transmit_mse)
        history['modulation_loss'].append(0.0)
        history['channel_snr'].append(args.channel_snr)

        # Validation
        if test_loader is not None:
            test_loss, test_acc, test_ber = validate_model(sender_model, receiver_model, quantizer, channel_modem, test_loader, device, args, use_polar_coding=False)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['test_ber'].append(test_ber)

            print(f"Stage 1 - Epoch {epoch+1}/{stage1_epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%, BER: {test_ber:.6f}, Transmit MSE: {epoch_transmit_mse:.8f}, Class Loss: {epoch_classification_loss:.6f}, VQ Loss: {epoch_vq_loss:.6f}")

            # Save best model for stage 1
            if test_acc > best_acc_stage1:
                best_acc_stage1 = test_acc
                best_checkpoint_stage1 = save_checkpoint(sender_model, receiver_model, quantizer, sender_optimizer, receiver_optimizer,
                                                        epoch, phase, {'acc': test_acc, 'loss': test_loss, 'ber': test_ber},
                                                        training_dir, is_best=False)
                # Also save as stage1_best
                stage1_path = os.path.join(training_dir, 'stage1_best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'phase': phase,
                    'sender_model_state_dict': sender_model.state_dict(),
                    'receiver_model_state_dict': receiver_model.state_dict(),
                    'quantizer_state_dict': quantizer.state_dict(),
                    'sender_optimizer_state_dict': sender_optimizer.state_dict(),
                    'receiver_optimizer_state_dict': receiver_optimizer.state_dict(),
                    'metrics': {'acc': test_acc, 'loss': test_loss, 'ber': test_ber}
                }, stage1_path)
        else:
            print(f"Stage 1 - Epoch {epoch+1}/{stage1_epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, BER: {epoch_ber:.6f}, Transmit MSE: {epoch_transmit_mse:.8f}, Class Loss: {epoch_classification_loss:.6f}, VQ Loss: {epoch_vq_loss:.6f}")

        scheduler.step()

    print(f"Stage 1 completed. Best checkpoint: {stage1_path}")

    # Stage 2: Fine-tune with polar coding
    print("\n--- Stage 2: Fine-tuning with polar coding ---")
    stage2_epochs = getattr(args, 'final_epochs', 60)  # Default 60 epochs for stage 2

    # Load stage 1 checkpoint
    if os.path.exists(stage1_path):
        checkpoint = torch.load(stage1_path, map_location=device)
        sender_model.load_state_dict(checkpoint['sender_model_state_dict'])
        receiver_model.load_state_dict(checkpoint['receiver_model_state_dict'])
        quantizer.load_state_dict(checkpoint['quantizer_state_dict'])
        print(f"Loaded stage 1 checkpoint from epoch {checkpoint['epoch']}")

    # Setup optimizers for stage 2 (can use different learning rate)
    stage2_lr = args.lr * 0.1  # Lower learning rate for fine-tuning
    sender_optimizer = torch.optim.Adam(sender_model.parameters(), lr=stage2_lr)
    receiver_optimizer = torch.optim.Adam(receiver_model.parameters(), lr=stage2_lr)
    scheduler = StepLR(sender_optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    best_acc = 0
    best_checkpoint = None

    for epoch in range(stage1_epochs, stage1_epochs + stage2_epochs):
        phase = 'stage2_with_polar'
        history['phases'].append(phase)

        # Training loop
        sender_model.train()
        receiver_model.train()

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_ber = 0.0
        epoch_transmit_mse = 0.0
        epoch_classification_loss = 0.0
        epoch_vq_loss = 0.0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Stage 2 - Epoch {epoch+1}/{stage1_epochs + stage2_epochs}")):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            total_samples += batch_size

            # Random SNR selection for training
            if hasattr(args, 'snr_min') and hasattr(args, 'snr_max'):
                snr_db = torch.rand(1).item() * (args.snr_max - args.snr_min) + args.snr_min
            else:
                snr_db = getattr(args, 'channel_snr', 15.0)

            # Forward pass through sender
            channel_noise = psnr_to_noise(torch.tensor(snr_db, device=device)).unsqueeze(0).expand(batch_size, -1)
            sender_output = sender_model(data, channel_noise)

            features, importance_weights, pre_task_logits, feature_shape, original_encoded_features = sender_output

            # Quantize features
            quantized_features, quantizer_indices, vq_loss = quantizer.encode(features)

            # Annealing mechanism - gradually introduce transmit loss
            anneal_ratio = min(1.0, (epoch - stage1_epochs) / 20.0)

            # Stage 2: WITH polar coding
            if hasattr(channel_modem, 'transmit_with_polar_coding'):
                group_to_use = getattr(args, 'max_groups', 8)
                received_features, ber, transmit_mse = channel_modem.transmit_with_polar_coding(
                    quantized_features, quantizer_indices,
                    snr_db=snr_db,
                    use_coding=True,
                    group_to_use=group_to_use,
                    quantizer=quantizer
                )
            else:
                received_features = channel_modem.transmit(quantized_features, snr_db)
                ber = torch.tensor(0.01, device=device)
                transmit_mse = torch.nn.MSELoss()(received_features, quantized_features)
            received_features = quantized_features + (received_features - quantized_features).detach()
            epoch_transmit_mse += transmit_mse.item() * batch_size

            # Receiver processing
            receiver_output = receiver_model(received_features, channel_noise, feature_shape)

            # Calculate losses
            classification_loss = nn.CrossEntropyLoss()(receiver_output, target)
            vq_weight = getattr(args, 'vq_loss_weight', 0.1)
            total_loss = classification_loss + vq_weight *  anneal_ratio * (vq_loss +transmit_mse)

            # Calculate metrics
            predictions = receiver_output.argmax(dim=1)
            acc = (predictions == target).float().mean().item() * 100

            # Backward pass
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
            epoch_classification_loss += classification_loss.item() * batch_size
            epoch_vq_loss += vq_loss.item() * batch_size

        # Average metrics
        epoch_loss /= total_samples
        epoch_acc /= total_samples
        epoch_ber /= total_samples
        epoch_transmit_mse /= total_samples
        epoch_classification_loss /= total_samples
        epoch_vq_loss /= total_samples

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['train_ber'].append(epoch_ber)
        history['modulation_loss'].append(0.0)
        history['channel_snr'].append(args.channel_snr)

        # Validation
        if test_loader is not None:
            test_loss, test_acc, test_ber = validate_model(sender_model, receiver_model, quantizer, channel_modem, test_loader, device, args, use_polar_coding=True)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['test_ber'].append(test_ber)

            print(f"Stage 2 - Epoch {epoch+1}/{stage1_epochs + stage2_epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%, BER: {test_ber:.6f}, Transmit MSE: {epoch_transmit_mse:.6f}, Class Loss: {epoch_classification_loss:.6f}, VQ Loss: {epoch_vq_loss:.6f}")

            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                best_checkpoint = save_checkpoint(sender_model, receiver_model, quantizer, sender_optimizer, receiver_optimizer,
                                                epoch, phase, {'acc': test_acc, 'loss': test_loss, 'ber': test_ber},
                                                training_dir, is_best=True)
        else:
            print(f"Stage 2 - Epoch {epoch+1}/{stage1_epochs + stage2_epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, BER: {epoch_ber:.6f}, Transmit MSE: {epoch_transmit_mse:.6f}, Class Loss: {epoch_classification_loss:.6f}, VQ Loss: {epoch_vq_loss:.6f}")

        scheduler.step()

    # Save training metrics
    save_training_metrics(history, training_dir, "Two-Stage Training Progress")

    # Save model components
    save_model_components(sender_model, receiver_model, quantizer, training_dir, prefix="final_")

    return {
        'training_dir': training_dir,
        'best_checkpoint': best_checkpoint,
        'stage1_checkpoint': stage1_path,
        'final_metrics': {
            'train_acc': history['train_acc'][-1],
            'test_acc': history['test_acc'][-1] if test_loader else None,
            'ber': history['test_ber'][-1] if test_loader else history['train_ber'][-1]
        },
        'history': history
    }


if __name__ == "__main__":
    # Example usage
    print("Combined Training Module")
    print("Use train_direct_polar() or train_two_stage() functions")
