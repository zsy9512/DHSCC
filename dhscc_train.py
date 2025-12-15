"""
Training module for DHSCC vibration classification model.

This module implements the three-phase training strategy:
1. Pretrain: Train the base model without wireless channel
2. Selector: Train the adaptive group selector 
3. Final: Joint training of the full model with wireless channel

The training process is modular and supports checkpointing for resuming training.
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

from utils.file_io import ensure_dir
from utils.signal_processing import set_seed


def create_training_directory(base_dir="checkpoints"):
    """
    Create timestamped directory for training outputs.
    
    Args:
        base_dir: Base directory for checkpoints
        
    Returns:
        str: Path to created training directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_dir = os.path.join(base_dir, f"training_{timestamp}")
    ensure_dir(training_dir)
    
    # Create subdirectories
    subdirs = ['pretrain', 'selector', 'final', 'plots', 'logs']
    for subdir in subdirs:
        ensure_dir(os.path.join(training_dir, subdir))
    
    print(f"📁 Created training directory: {training_dir}")
    return training_dir


def save_training_metrics(history, save_dir):
    """
    Save training metrics and create visualizations.
    
    Args:
        history: Training history dictionary
        save_dir: Directory to save metrics and plots
    """
    # Save raw metrics
    metrics_path = os.path.join(save_dir, 'logs', 'training_metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"📊 Saved training metrics to: {metrics_path}")
    
    # Create visualizations
    plot_training_progress(history, save_dir)


def plot_training_progress(history, save_dir):
    """
    Create comprehensive training progress plots.
    
    Args:
        history: Training history dictionary
        save_dir: Directory to save plots
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('DHSCC Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['test_acc'], 'r-', label='Test Acc', linewidth=2)
    axes[0, 1].set_title('Classification Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Loss components
    axes[0, 2].plot(epochs, history['train_commit_loss'], 'g-', label='Commit Loss', linewidth=2)
    axes[0, 2].plot(epochs, history['train_transmit_loss'], 'orange', label='Transmit Loss', linewidth=2)
    axes[0, 2].set_title('Loss Components')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: BER curves
    axes[1, 0].plot(epochs, history['train_ber'], 'b-', label='Train BER', linewidth=2)
    axes[1, 0].set_title('Bit Error Rate (BER)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('BER')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Plot 5: Group usage
    axes[1, 1].plot(epochs, history['train_group'], 'b-', label='Train Group', linewidth=2)
    axes[1, 1].plot(epochs, history['test_group'], 'r-', label='Test Group', linewidth=2)
    axes[1, 1].set_title('Average Groups Used')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Number of Groups')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Training phase indicators
    pretrain_epochs = next((i for i, phase in enumerate(history.get('phases', [])) if phase != 'Pretrain'), len(epochs))
    selector_epochs = next((i for i, phase in enumerate(history.get('phases', [])) if phase == 'Final'), len(epochs))
    
    axes[1, 2].axvspan(1, pretrain_epochs, alpha=0.3, color='blue', label='Pretrain')
    axes[1, 2].axvspan(pretrain_epochs, selector_epochs, alpha=0.3, color='orange', label='Selector')
    axes[1, 2].axvspan(selector_epochs, len(epochs), alpha=0.3, color='green', label='Final')
    axes[1, 2].plot(epochs, history['test_acc'], 'k-', linewidth=2, label='Test Accuracy')
    axes[1, 2].set_title('Training Phases')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Test Accuracy (%)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    plot_path = os.path.join(save_dir, 'plots', 'training_progress.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"📈 Saved training plots to: {plot_path}")


def save_best_model(model, optimizer, epoch, phase, metrics, save_dir, is_best_acc=False):
    """
    Save model checkpoint with comprehensive information.
    
    Args:
        model: DHSCCNet model
        optimizer: Training optimizer
        epoch: Current epoch
        phase: Training phase ('pretrain', 'selector', 'final')
        metrics: Dictionary of current metrics
        save_dir: Training directory
        is_best_acc: Whether this is the best accuracy model
    """
    # Prepare save data
    save_data = {
        'epoch': epoch,
        'phase': phase,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save adaptive selector state if available
    if hasattr(model, 'save_adaptive_selector_state'):
        selector_state = model.save_adaptive_selector_state()
        if selector_state is not None:
            save_data['adaptive_selector_state'] = selector_state
    
    # Determine save path
    if is_best_acc:
        filename = f'best_{phase}_model.pth'
    else:
        filename = f'{phase}_epoch_{epoch}.pth'
    
    save_path = os.path.join(save_dir, phase, filename)
    torch.save(save_data, save_path)
    
    print(f"💾 Saved {phase} checkpoint: {filename}")
    print(f"   📊 Metrics: Acc={metrics.get('test_acc', 0):.2f}%, Loss={metrics.get('test_loss', 0):.4f}")
    
    return save_path


def train(model, train_loader, device, args, test_loader=None):
    """
    Unified training function for DHSCC model with phase control and checkpointing.
    
    Args:
        model: DHSCCNet model
        train_loader: DataLoader for training
        device: torch.device
        args: argparse.Namespace
        test_loader: DataLoader for testing (optional)
        
    Returns:
        dict: {'final': {'final_checkpoint': ...}, ...}
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    # Create timestamped training directory (replaces simple checkpoint_dir)
    training_dir = create_training_directory(args.checkpoint_dir)
    
    # 训练指标历史
    history = {
        'train_loss': [], 'train_acc': [], 'train_commit_loss': [], 'train_transmit_loss': [],
        'train_group': [], 'train_ber': [],
        'test_loss': [], 'test_acc': [], 'test_group': [], 'phases': []
    }
    
    # Track best models for each phase
    best_models = {
        'pretrain': {'acc': 0, 'epoch': 0, 'path': None},
        'selector': {'acc': 0, 'epoch': 0, 'path': None},
        'final': {'acc': 0, 'epoch': 0, 'path': None}
    }
    
    final_checkpoint = None
    prev_phase = None
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_class_loss = 0.0
        epoch_commit_loss = 0.0
        epoch_transmit_loss = 0.0
        epoch_acc = 0.0
        epoch_ber = 0.0
        epoch_group = 0.0
        total = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # 每个新batch开始时重置group选择（确保同一batch内使用相同group）
            if hasattr(model, 'reset_batch_group'):
                model.reset_batch_group()

            # Forward: outputs, commit_loss, ber, transmit_mse, zeros_count, channel_noise, group_to_use, debug_info
            outputs, commit_loss, ber, transmit_mse, zeros_count, channel_noise, group_to_use, debug_info = model(data, epoch, args.snr)
            eta = (1 - 2 * torch.log10(torch.Tensor([channel_noise],device="cpu") + 1e-6)) .to(device)
            # 分类损失 - CrossEntropyLoss
            criterion = nn.CrossEntropyLoss()
            classification_loss = criterion(outputs, target)
            
            # 退火机制 - 渐进式引入传输损失
            anneal_ratio = min(1.0, epoch / 20)
            
            # 总损失组成: classification_loss + 0.1 * commit_loss + 0.05 * anneal_ratio * transmit_mse
            total_loss = eta * classification_loss + 0.1 * eta *  commit_loss + 0.1 * anneal_ratio * transmit_mse*eta
           
            # 检查NaN值
            if torch.isnan(total_loss):
                print(f"NaN detected in loss at epoch {epoch}, batch {batch_idx}")
                print(f"  Classification: {classification_loss.item()}")
                print(f"  Commit: {commit_loss.item()}")
                print(f"  Transmit MSE: {transmit_mse.item()}")
                print(f"  Anneal ratio: {anneal_ratio}")
                continue

            loss = total_loss
            
            # 第二阶段不进行梯度下降
            # if epoch < args.pretrain_epochs or epoch >= args.selector_epochs:
            #     loss.backward()
            #     optimizer.step()
            loss.backward()
            optimizer.step()
            
            # 统计
            epoch_loss += loss.item()
            epoch_class_loss += classification_loss.item()
            epoch_commit_loss += commit_loss.item() if hasattr(commit_loss, 'item') else float(commit_loss)
            epoch_transmit_loss += transmit_mse.item() if hasattr(transmit_mse, 'item') else float(transmit_mse)
            epoch_ber += ber.item() if hasattr(ber, 'item') else float(ber)
            epoch_group += group_to_use if isinstance(group_to_use, (int, float)) else float(group_to_use)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 更新自适应组选择器性能（仅在选择器训练阶段）
            if hasattr(model, 'update_performance'):
                if hasattr(model, 'is_selector_training_phase') and model.is_selector_training_phase(epoch):
                    batch_accuracy = predicted.eq(target).float().mean().item()
                    model.update_performance(batch_accuracy, epoch)
        
        # 平均训练指标
        avg_loss = epoch_loss / len(train_loader)
        avg_class_loss = epoch_class_loss / len(train_loader)
        avg_commit_loss = epoch_commit_loss / len(train_loader)
        avg_transmit_loss = epoch_transmit_loss / len(train_loader)
        avg_acc = 100. * correct / total
        avg_ber = epoch_ber / len(train_loader)
        avg_group = epoch_group / len(train_loader)
        
        # 每轮epoch后进行简单测试
        if test_loader is not None:
            from dhscc_test import test
            test_loss_min, test_acc_min, test_group_min = test(model, test_loader, epoch, args.snr_min, device)
            test_loss, test_acc, test_group = test(model, test_loader, epoch, args.snr, device)
            test_loss_max, test_acc_max, test_group_max = test(model, test_loader, epoch, args.snr_max, device)
            test_loss= (test_loss_min + test_loss + test_loss_max) / 3
            test_acc = (test_acc_min + test_acc + test_acc_max) / 3
            test_group = (test_group_min + test_group + test_group_max) / 3
        else:
            test_loss, test_acc, test_group = 0.0, 0.0, 0.0
        
        # 记录历史
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(avg_acc)
        history['train_commit_loss'].append(avg_commit_loss)
        history['train_transmit_loss'].append(avg_transmit_loss)
        history['train_group'].append(avg_group)
        history['train_ber'].append(avg_ber)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_group'].append(test_group)
        
        # 确定当前训练阶段
        if epoch < args.pretrain_epochs:
            current_phase = "pretrain"
        elif epoch < args.selector_epochs:
            current_phase = "selector"
        else:
            current_phase = "final"

        # 阶段切换时加载前一阶段最优参数
        if prev_phase is not None and current_phase != prev_phase:
            best_path = best_models[prev_phase]['path']
            if best_path is not None and os.path.exists(best_path):
                print(f"\n🔄 Loading best {prev_phase} model for {current_phase} phase: {best_path}")
                checkpoint = torch.load(best_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"✓ Loaded best {prev_phase} model weights for {current_phase} phase.")
            else:
                print(f"\n⚠️ No best {prev_phase} model found to load for {current_phase} phase.")
        prev_phase = current_phase

        
        # 当前epoch的metrics
        current_metrics = {
            'train_loss': avg_loss,
            'train_acc': avg_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'train_commit_loss': avg_commit_loss,
            'train_transmit_loss': avg_transmit_loss,
            'train_ber': avg_ber,
            'train_group': avg_group,
            'test_group': test_group
        }
        
        # 打印详细的训练和测试日志
        print(f"\n=== Epoch {epoch+1}/{args.epochs} [{current_phase.capitalize()} Phase] ===")
        print(f"Training Metrics:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Classification Loss: {avg_class_loss:.4f}")
        print(f"  Commit Loss: {avg_commit_loss:.4f}")
        print(f"  Transmit Loss: {avg_transmit_loss:.4f}")
        print(f"  Accuracy: {avg_acc:.2f}%")
        print(f"  BER: {avg_ber:.6f}")
        print(f"  Avg Group Used: {avg_group:.1f}")
        
        print(f"Testing Metrics:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"  Test Group Used: {test_group:.1f}")
        print("=" * 50)
        
        # 阶段切换与权重冻结/解冻
        if epoch == args.pretrain_epochs:
            print(f"\n🔄 Switching to Selector Training Phase at epoch {epoch+1}")
            if hasattr(model, 'set_training_phase'):
                model.set_training_phase('selector')
                print("✓ Model phase set to 'selector'")
            # 确保自适应选择器处于训练模式
            if hasattr(model, 'adaptive_group_selector') and model.adaptive_group_selector is not None:
                model.adaptive_group_selector.set_inference_mode(False)
                print("✓ Adaptive selector set to training mode")

        
        if epoch == args.selector_epochs:
            print(f"\n🔄 Switching to Final Training Phase at epoch {epoch+1}")
            if hasattr(model, 'set_training_phase'):
                model.set_training_phase('final')
                print("✓ Model phase set to 'final'")
            # 调用最终化函数并切换到推理模式
            if hasattr(model, 'adaptive_group_selector') and model.adaptive_group_selector is not None:
                model.adaptive_group_selector.finalize_training()
                model.adaptive_group_selector.set_inference_mode(True)  # Final阶段使用固定策略
                print("✓ Adaptive selector finalized and set to inference mode")
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = args.lr/100
            #print(f"✓ Learning rate reset to {param_group['lr']} for final phase")
            # 保存最佳模型 (30 epoch后开始保存，每阶段只保存最优)
        if epoch >= 30 and test_loader is not None:
            if test_acc > best_models[current_phase]['acc']+0.1:  # 0.1%的精度提升才算新最佳
                # 更新最佳模型信息
                best_models[current_phase]['acc'] = test_acc
                best_models[current_phase]['epoch'] = epoch + 1
                
                # 保存最佳模型
                best_path = save_best_model(
                    model, optimizer, epoch + 1, current_phase, 
                    current_metrics, training_dir, is_best_acc=True
                )
                best_models[current_phase]['path'] = best_path
                
                print(f"🎯 New best {current_phase} accuracy: {test_acc:.2f}%")
        
        # 在selector阶段结束时输出选择器统计信息
        if epoch+1 == args.selector_epochs:
            if hasattr(model, 'adaptive_group_selector') and model.adaptive_group_selector is not None:
                # 输出选择器策略摘要
                if hasattr(model, 'get_group_strategy_summary'):
                    strategy_summary = model.get_group_strategy_summary()
                    print(f"📊 Group Strategy Summary (top 5 SNR levels):")
                    for i, (snr, group) in enumerate(strategy_summary[:5]):
                        print(f"  SNR {snr:.2f}dB → {group} groups")
                
                # 输出选择器统计信息
                if hasattr(model, 'get_adaptive_selector_statistics'):
                    stats = model.get_adaptive_selector_statistics()
                    print(f"📊 Selector Statistics: {stats.get('total_updates', 0)} updates, "
                          f"{stats.get('strategy_changes', 0)} changes")
        
        if epoch+1 == args.epochs:
            final_checkpoint = os.path.join(training_dir, 'final', 'final_model.pth')
            final_save_data = {
                'epoch': epoch+1, 
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': current_metrics,
                'history': history,
                'best_models': best_models,
                'timestamp': datetime.now().isoformat()
            }
            
            # 保存adaptive selector状态
            if hasattr(model, 'save_adaptive_selector_state'):
                selector_state = model.save_adaptive_selector_state()
                if selector_state is not None:
                    final_save_data['adaptive_selector_state'] = selector_state
                    
            torch.save(final_save_data, final_checkpoint)
            print(f"💾 Final checkpoint saved: {final_checkpoint}")
        
        scheduler.step()
    
    # 训练完成时保存最终metrics和可视化
    print(f"\n🎉 Training completed after {args.epochs} epochs!")
    print(f"📁 Training directory: {training_dir}")
    print("📊 Generating final metrics and visualizations...")
    
    save_training_metrics(history, training_dir)
    
    # 输出各阶段最佳结果
    for phase, info in best_models.items():
        if info['path'] is not None:
            print(f"📊 Best {phase} accuracy: {info['acc']:.2f}% (epoch {info['epoch']})")
    
    if final_checkpoint:
        print(f"📁 Final model saved to: {final_checkpoint}")
    
    return {
        'final': {'final_checkpoint': final_checkpoint}, 
        'history': history,
        'best_models': best_models,
        'training_dir': training_dir
    }


def resume_training(model, checkpoint_path, device, args):
    """
    Resume training from a checkpoint.
    
    Args:
        model: DHSCCNet model
        checkpoint_path: Path to checkpoint file
        device: Training device
        args: Training arguments
        
    Returns:
        dict: Loaded checkpoint information
    """
    print(f"🔄 Resuming training from checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load adaptive selector state if available
    if 'adaptive_selector_state' in checkpoint and hasattr(model, 'load_adaptive_selector_state'):
        model.load_adaptive_selector_state(checkpoint['adaptive_selector_state'])
        print("✓ Adaptive selector state restored")
    
    # Restore optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        print("ℹ️ Optimizer state found in checkpoint (create optimizer first to restore)")
    
    # Print checkpoint information
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        print(f"📊 Previous metrics:")
        if 'test_loss' in metrics:
            print(f"   Test Loss: {metrics['test_loss']:.4f}")
        if 'test_acc' in metrics:
            print(f"   Test Accuracy: {metrics['test_acc']:.2f}%")
    
    if 'phase' in checkpoint:
        print(f"📋 Training phase: {checkpoint['phase']}")
    
    if 'best_models' in checkpoint:
        print(f"📊 Best models from previous training:")
        for phase, info in checkpoint['best_models'].items():
            if info.get('acc', 0) > 0:
                print(f"   {phase}: {info['acc']:.2f}% (epoch {info['epoch']})")
    
    return checkpoint


def save_training_results(results, save_path):
    """
    Save training results to file.
    
    Args:
        results: Training results dictionary
        save_path: Path to save results
    """
    import pickle
    
    ensure_dir(os.path.dirname(save_path))
    
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Training results saved to: {save_path}")


def load_training_results(load_path):
    """
    Load training results from file.
    
    Args:
        load_path: Path to load results from
        
    Returns:
        dict: Loaded training results
    """
    import pickle
    
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Results file not found: {load_path}")
    
    with open(load_path, 'rb') as f:
        results = pickle.load(f)
    
    print(f"Training results loaded from: {load_path}")
    
    return results


def print_loss_components(phase, epoch, classification_loss, commit_loss, transmit_mse, anneal_ratio=None):
    """
    Print detailed loss component information for debugging.
    
    Args:
        phase: Training phase name
        epoch: Current epoch
        classification_loss: Classification loss value
        commit_loss: Commitment loss value
        transmit_mse: Transmission MSE loss value
        anneal_ratio: Annealing ratio (if applicable)
    """
    print(f"[{phase} Epoch {epoch}] Loss Components:")
    print(f"  Classification Loss: {classification_loss.item():.6f}")
    print(f"  Commitment Loss: {commit_loss.item():.6f}")
    print(f"  Transmission MSE: {transmit_mse.item():.6f}")
    if anneal_ratio is not None:
        print(f"  Annealing Ratio: {anneal_ratio:.4f}")
        print(f"  Weighted Transmission: {0.05 * anneal_ratio * transmit_mse.item():.6f}")
    total_loss = classification_loss + 0.1 * commit_loss
    if anneal_ratio is not None:
        total_loss += 0.05 * anneal_ratio * transmit_mse
    print(f"  Total Loss: {total_loss.item():.6f}")
    print("-" * 50)
