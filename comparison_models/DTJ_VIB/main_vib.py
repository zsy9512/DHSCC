"""
Main Training Script for DTJ-VIB System

This script trains the DTJ-VIB network for vibration classification using
joint source-channel coding with digital modulation.
"""

import os
import sys
import argparse
import multiprocessing as mp
import torch 
import torch.optim as optim 
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.DT_JSCC_VIB import DTJSCC_VIB, DTJSCC_VIB_Small
from model.losses import RIBLoss
from datasets.tactile_dataloader import get_data
from engine_vib import train_one_epoch, test, save_checkpoint, load_checkpoint
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from modulation import QAM, PSK


def main(args):
    """
    Main training function for DTJ-VIB.
    
    Args:
        args: Parsed command line arguments
    """
    print("=== DTJ-VIB Training Started ===")
    print(f"Dataset: {args.dataset}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Modulation: {args.mod}")
    print(f"PSNR: {args.psnr}")
    print(f"Lambda: {args.lam}")
    
    # Create model
    if args.model_type == 'standard':
        model = DTJSCC_VIB(
            in_channels=2,
            latent_channels=args.latent_d,
            out_classes=args.num_classes,
            num_embeddings=args.num_embeddings,
            snr=args.psnr
        )
    elif args.model_type == 'small':
        model = DTJSCC_VIB_Small(
            in_channels=2,
            latent_channels=args.latent_d,
            out_classes=args.num_classes,
            num_embeddings=args.num_embeddings,
            snr=args.psnr
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model.to(args.device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        params=model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = lr_scheduler.StepLR(
        optimizer, 
        step_size=args.scheduler_step, 
        gamma=args.scheduler_gamma
    )
    
    # Loss function
    criterion = RIBLoss(args.lam)
    criterion.train()
    
    # Data loaders
    print("Loading datasets...")
    dataloader_train = get_data(args.dataset, args.batch_size, n_worker=args.num_workers, train=True)
    dataloader_test = get_data(args.dataset, args.batch_size, n_worker=args.num_workers, train=False)
    
    print(f"Training batches: {len(dataloader_train)}")
    print(f"Test batches: {len(dataloader_test)}")
    
    # Tensorboard writer
    log_writer = SummaryWriter(f'./logs/{experiment_name}')
    
    # Initialize training state
    current_epoch = 0
    best_acc = 0.0
    args.n_iter = 0  # Global iteration counter
    
    # Load checkpoint if exists
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        current_epoch, best_acc = load_checkpoint(model, optimizer, checkpoint_path, args.device)
        print(f"Resumed from epoch {current_epoch + 1}, best acc: {best_acc:.4f}")
        
    # Training loop
    print("\n=== Starting Training ===")
    for epoch in range(current_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Create modulation module for this epoch
        if args.mod == 'qam':
            mod = QAM(args.num_embeddings, args.psnr)
        elif args.mod == 'psk':
            mod = PSK(args.num_embeddings, args.psnr)
        else:
            raise ValueError(f"Unknown modulation: {args.mod}")
        
        # Training
        train_one_epoch(
            dataloader_train, 
            model, 
            optimizer=optimizer, 
            criterion=criterion, 
            writer=log_writer, 
            epoch=epoch, 
            mod=mod, 
            args=args
        )
        
        # Step scheduler
        scheduler.step()
        
        # Evaluation (skip first few epochs for faster training)
        if epoch >= args.eval_start_epoch:
            acc1 = test(
                dataloader_test, 
                model, 
                criterion=criterion, 
                writer=log_writer, 
                epoch=epoch, 
                mod=mod, 
                args=args
            )
            
            print(f'Epoch {epoch + 1} - Test Acc1: {acc1:.4f}')
            print(f'Current best accuracy: {best_acc:.4f}')
            
            # Save best model
            is_best = acc1 > best_acc
            if is_best:
                best_acc = acc1
                print(f"New best accuracy: {best_acc:.4f}")
            
            # Save checkpoint
            save_checkpoint(
                model, optimizer, epoch, best_acc, 
                path_to_backup, is_best=is_best
            )
        
        # Log epoch info
        log_writer.add_scalar('train/epoch', epoch, epoch)
        log_writer.add_scalar('train/best_acc', best_acc, epoch)
        log_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
    
    print(f"\n=== Training Completed ===")
    print(f"Best accuracy achieved: {best_acc:.4f}")
    print(f"Model saved to: {path_to_backup}")
    
    # Close tensorboard writer
    log_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DTJ-VIB Training')
    
    # Dataset parameters
    parser.add_argument('-d', '--dataset', type=str, default='VIB', 
                       help='Dataset name (VIB for vibration data)')
    parser.add_argument('--dataset_root', type=str, 
                       default='/home/zsy/workplace/DHSCC/dataset/vibcls',
                       help='Path to dataset root directory')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='standard', 
                       choices=['standard', 'small'],
                       help='Model variant to use')
    parser.add_argument('--latent_d', type=int, default=256, 
                       help='Latent feature dimension')
    parser.add_argument('--num_classes', type=int, default=10, 
                       help='Number of output classes')
    parser.add_argument('--num_embeddings', type=int, default=16, 
                       help='Size of quantization codebook')
    
    # Training parameters
    parser.add_argument('-e', '--epochs', type=int, default=150, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, 
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, 
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                       help='Weight decay for AdamW')
    parser.add_argument('--maxnorm', type=float, default=1.0, 
                       help='Max norm for gradient clipping')
    
    # Scheduler parameters
    parser.add_argument('--scheduler_step', type=int, default=20, 
                       help='Step size for learning rate scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, 
                       help='Gamma for learning rate scheduler')
    
    # Communication parameters
    parser.add_argument('--mod', type=str, default='psk', choices=['psk', 'qam'],
                       help='Modulation scheme')
    parser.add_argument('--psnr', type=float, default=3.0, 
                       help='Channel PSNR in dB')
    parser.add_argument('--lam', type=float, default=1e-4, 
                       help='Entropy regularization weight')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda:0', 
                       help='Device for training')
    parser.add_argument('--num_workers', type=int, default=mp.cpu_count() - 1,
                       help='Number of workers for data loading')
    parser.add_argument('--eval_start_epoch', type=int, default=10,
                       help='Epoch to start evaluation (for faster initial training)')
    
    # Paths
    parser.add_argument('-r', '--root', type=str, default='./trained_models', 
                       help='Root directory for saving models')
    
    args = parser.parse_args()
    
    # Create experiment name
    experiment_name = (f"{args.dataset}-{args.model_type}-"
                      f"num_e{args.num_embeddings}-"
                      f"latent_d{args.latent_d}-"
                      f"mod{args.mod}-"
                      f"psnr{args.psnr}-"
                      f"lam{args.lam}")
    
    # Setup paths
    path_to_backup = os.path.join(args.root, experiment_name)
    if not os.path.exists(path_to_backup):
        print(f'Creating model directory: {path_to_backup}')
        os.makedirs(path_to_backup)
    
    checkpoint_path = os.path.join(path_to_backup, 'best.pt')
    
    if not os.path.exists('./logs'):
        print('Creating logs directory...')
        os.makedirs('./logs')
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    print(f'Using device: {device}')
    
    # Start training
    main(args)
