"""
Training Engine for DTJ-VIB System

Modified from original DTJ engine to work with vibration classification tasks.
Handles training and evaluation loops for DTJ-VIB networks.
"""

from datetime import datetime
from typing import Iterable
import torch 
from tensorboardX import SummaryWriter
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
# 现在可以正常导入
from accuracy import accuracy


def train_one_epoch(dataloader: Iterable, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, criterion: torch.nn.Module,
                    writer: SummaryWriter, epoch: int, mod, args):
    """
    Train the model for one epoch.
    
    Args:
        dataloader: Training data loader
        model: DTJ-VIB model
        optimizer: Optimizer
        criterion: Loss function
        writer: Tensorboard writer
        epoch: Current epoch number
        mod: Modulation module
        args: Training arguments
    """
    model.train()
    batches_start = datetime.now()
    
    for i_batch, (imgs, labs) in enumerate(dataloader):
        imgs = imgs.to(args.device)
        labs = labs.to(args.device)
        
        # Forward pass through DTJ-VIB network
        outs, dist,loss_commit = model(imgs, mod=mod)
        
        # Compute loss
        loss, loss_post, loss_entr = criterion(dist, outs, labs,loss_commit)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping if specified
        if args.maxnorm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.maxnorm)
        
        optimizer.step()

        # Compute accuracy
        acc = accuracy(outs, labs, (1, 3))
        
        # Log metrics
        writer.add_scalar('train/losses', loss.item(), args.n_iter)
        writer.add_scalar('train/losses_post', loss_post.item(), args.n_iter)
        writer.add_scalar('train/losses_entr', loss_entr.item(), args.n_iter)
        writer.add_scalar('train/acc1', acc[0].item(), args.n_iter)
        writer.add_scalar('train/acc3', acc[1].item(), args.n_iter)
        args.n_iter += 1
        
        # Print progress
        if i_batch % 80 == 0:
            batches_end = datetime.now()
            avg_time = (batches_end - batches_start) / 80
            print(f'\n \n average batch time for batch size of {imgs.shape[0]}: {avg_time}')
            batches_start = datetime.now()
            print(f'[{epoch}][{i_batch}/{len(dataloader)}]\t '
                  f'Losses: {loss.item():.4f}\t '
                  f'Loss_post: {loss_post.item():.4f}\t '
                  f'Loss_entr: {loss_entr.item():.4f}')
            print(f'acc1: {acc[0].item():.4f}\t acc3: {acc[1].item():.4f}')


def test(dataloader, model, criterion, writer, epoch, mod, args):
    """
    Evaluate the model on test/validation data.
    
    Args:
        dataloader: Test/validation data loader
        model: DTJ-VIB model
        criterion: Loss function
        writer: Tensorboard writer
        epoch: Current epoch number
        mod: Modulation module
        args: Training arguments
        
    Returns:
        Top-1 accuracy
    """
    losses, losses_post, losses_entr, acc1, acc3 = 0., 0., 0., 0., 0.
    
    with torch.no_grad():
        model.eval()
        print("Testing...")
        
        for imgs, labs in dataloader:
            imgs = imgs.to(args.device)
            labs = labs.to(args.device)
            
            # Forward pass
            outs, dist,commit_loss = model(imgs, mod=mod,snr=args.psnr)
            
            # Compute loss
            loss, loss_post, loss_entr = criterion(dist, outs, labs,commit_loss)
            
            # Compute accuracy
            acc = accuracy(outs, labs, (1, 3))
            acc1 += acc[0].item()
            acc3 += acc[1].item()
            
            # Log individual batch metrics
            writer.add_scalar('test/acc1-single', acc[0].item(), epoch)
            writer.add_scalar('test/acc3-single', acc[1].item(), epoch)
            
            # Accumulate losses
            losses += loss.item()
            losses_post += loss_post.item()
            losses_entr += loss_entr.item()
    
    # Compute average metrics
    avg_loss = losses / len(dataloader)
    avg_loss_post = losses_post / len(dataloader)
    avg_loss_entr = losses_entr / len(dataloader)
    avg_acc1 = acc1 / len(dataloader)
    avg_acc3 = acc3 / len(dataloader)
    
    # Log average metrics
    writer.add_scalar('test/losses', avg_loss, epoch)
    writer.add_scalar('test/losses_post', avg_loss_post, epoch)
    writer.add_scalar('test/losses_entr', avg_loss_entr, epoch)
    writer.add_scalar('test/acc1', avg_acc1, epoch)
    writer.add_scalar('test/acc3', avg_acc3, epoch)
    
    print(f'Test Results - Loss: {avg_loss:.4f}, '
          f'Acc1: {avg_acc1:.4f}, Acc3: {avg_acc3:.4f}')
    print('Done!')
    
    return avg_acc1


def evaluate_at_snrs(dataloader, model, criterion, snr_list, args):
    """
    Evaluate model performance across different SNR values.
    
    Args:
        dataloader: Test data loader
        model: DTJ-VIB model
        criterion: Loss function
        snr_list: List of SNR values to test
        args: Arguments containing modulation settings
        
    Returns:
        Dictionary with SNR-specific results
    """
    from utils.modulation import QAM, PSK
    
    results = {}
    
    model.eval()
    
    for snr in snr_list:
        print(f"Evaluating at SNR: {snr} dB")
        
        # Create modulation module for this SNR
        if args.mod == 'qam':
            mod = QAM(args.num_embeddings, snr)
        elif args.mod == 'psk':
            mod = PSK(args.num_embeddings, snr)
        else:
            raise ValueError(f"Unknown modulation: {args.mod}")
        
        losses, acc1, acc3 = 0., 0., 0.
        
        with torch.no_grad():
            for imgs, labs in dataloader:
                imgs = imgs.to(args.device)
                labs = labs.to(args.device)
                
                outs, dist = model(imgs, mod=mod)
                loss, _, _ = criterion(dist, outs, labs)
                
                acc = accuracy(outs, labs, (1, 3))
                acc1 += acc[0].item()
                acc3 += acc[1].item()
                losses += loss.item()
        
        # Store results for this SNR
        results[snr] = {
            'loss': losses / len(dataloader),
            'acc1': acc1 / len(dataloader),
            'acc3': acc3 / len(dataloader)
        }
        
        print(f"  SNR {snr} dB - Loss: {results[snr]['loss']:.4f}, "
              f"Acc1: {results[snr]['acc1']:.4f}, Acc3: {results[snr]['acc3']:.4f}")
    
    return results


def log_model_info(model, writer, epoch):
    """
    Log model information to tensorboard.
    
    Args:
        model: DTJ-VIB model
        writer: Tensorboard writer
        epoch: Current epoch
    """
    # Log model parameters count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    writer.add_scalar('model/total_params', total_params, epoch)
    writer.add_scalar('model/trainable_params', trainable_params, epoch)
    
    # Log gradient norms
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    
    writer.add_scalar('train/grad_norm', total_norm, epoch)


def save_checkpoint(model, optimizer, epoch, best_acc, save_path, is_best=False):
    """
    Save model checkpoint.
    
    Args:
        model: DTJ-VIB model
        optimizer: Optimizer
        epoch: Current epoch
        best_acc: Best accuracy achieved so far
        save_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_states': model.state_dict(),
        'optimizer_states': optimizer.state_dict(),
        'best_acc': best_acc
    }
    
    # Save regular checkpoint
    #torch.save(checkpoint, f'{save_path}/model_epoch_{epoch + 1}.pt')
    
    # Save best model separately if this is the best
    if is_best:
        torch.save(checkpoint, f'{save_path}/best.pt')
        print(f"New best model saved with accuracy: {best_acc:.4f}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Load model checkpoint.
    
    Args:
        model: DTJ-VIB model
        optimizer: Optimizer
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (current_epoch, best_acc)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_states'])
    optimizer.load_state_dict(checkpoint['optimizer_states'])
    
    current_epoch = checkpoint['epoch']
    best_acc = checkpoint.get('best_acc', 0.0)
    
    print(f"Loaded checkpoint from epoch {current_epoch + 1}, best acc: {best_acc:.4f}")
    
    return current_epoch, best_acc


if __name__ == '__main__':
    """
    Test the training engine functions.
    """
    print("DTJ-VIB training engine ready!")
