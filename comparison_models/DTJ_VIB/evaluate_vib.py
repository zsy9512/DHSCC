"""
Evaluation Script for DTJ-VIB System

This script evaluates trained DTJ-VIB models on vibration classification tasks
across different channel conditions (SNR values).
"""

import os
import sys
import argparse
import torch
import numpy as np

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.DT_JSCC_VIB import DTJSCC_VIB, DTJSCC_VIB_Small
from model.losses import RIBLoss
from datasets.tactile_dataloader import get_data
from engine_vib import evaluate_at_snrs

# Add utils path
utils_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils')
sys.path.append(utils_path)
from modulation import QAM, PSK
from accuracy import accuracy


def eval_test(dataloader, model, mod, snr,args):
    """
    Evaluate model on test data with given modulation.
    
    Args:
        dataloader: Test data loader
        model: DTJ-VIB model
        mod: Modulation module
        args: Evaluation arguments
        
    Returns:
        Tuple of (acc1, acc3, distribution)
    """
    acc1, acc3 = 0., 0.
    dist_result = None
    
    with torch.no_grad():
        model.eval()
        import time
        start=time.time()
        for imgs, labs in dataloader:
            imgs = imgs.to(args.device)
            labs = labs.to(args.device)
            
            outs, dist,_ = model(imgs, mod=mod,snr=snr)
            acc = accuracy(outs, labs, (1, 3))
            acc1 += acc[0].item()
            acc3 += acc[1].item()
            
            # Store one distribution for analysis
            if dist_result is None:
                dist_result = dist.cpu().clone()
        end = time.time()
        time_cost = (end - start)/ len(dataloader)
    avg_acc1 = acc1 / len(dataloader)
    avg_acc3 = acc3 / len(dataloader)
    
    print(f'Evaluation completed: Acc1={avg_acc1:.4f}, Acc3={avg_acc3:.4f},time_cost={time_cost:.2f}s')
    
    return avg_acc1, avg_acc3, dist_result


def evaluate_model_comprehensive(args):
    """
    Comprehensive evaluation of DTJ-VIB model.
    
    Args:
        args: Evaluation arguments
    """
    print("=== DTJ-VIB Model Evaluation ===")
    print(f"Model: {args.name}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    
    # Load model architecture
    if args.model_type == 'standard':
        model = DTJSCC_VIB(
            in_channels=2,
            latent_channels=args.latent_d,
            out_classes=args.num_classes,
            num_embeddings=args.num_embeddings
        )
    elif args.model_type == 'small':
        model = DTJSCC_VIB_Small(
            in_channels=2,
            latent_channels=args.latent_d,
            out_classes=args.num_classes,
            num_embeddings=args.num_embeddings
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Load checkpoint
    print(f"Loading model from: {path_to_backup}")
    checkpoint = torch.load(path_to_backup, map_location=args.device)
    model.load_state_dict(checkpoint['model_states'])
    model.to(args.device)
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'best_acc' in checkpoint:
        print(f"Training best accuracy: {checkpoint['best_acc']:.4f}")
    
    # Load test data
    print("Loading test dataset...")
    dataloader_test = get_data(args.dataset, 256, n_worker=8, train=False)
    print(f"Test batches: {len(dataloader_test)}")
    
    # Evaluate across different SNR values
    print("\n=== SNR Sweep Evaluation ===")
    SNR_list = np.linspace(0, 5, 11).tolist()  # 0, 0.5, ..., 5 dB
    print(f"Testing SNR values: {SNR_list}")
    
    results = {
        'snr_values': SNR_list,
        'acc1_results': [],
        'acc3_results': [],
        'loss_results': []
    }
    
    # Loss function for evaluation
    criterion = RIBLoss(args.lam)
    
    for snr in SNR_list:
        print(f"\nEvaluating at SNR: {snr} dB")
        
        # Create modulation for this SNR
        if args.mod == 'qam':
            mod = QAM(args.num_embeddings, snr)
        elif args.mod == 'psk':
            mod = PSK(args.num_embeddings, snr)
        else:
            raise ValueError(f"Unknown modulation: {args.mod}")
        
        # Evaluate
        acc1, acc3, dist = eval_test(dataloader_test, model, mod,snr, args)
        
        # Compute loss (optional)
        # We can approximate loss from the last batch
        loss_approx = 0.0  # You could compute this if needed
        
        # Store results
        results['acc1_results'].append(acc1)
        results['acc3_results'].append(acc3)
        results['loss_results'].append(loss_approx)
        
        print(f"  SNR {snr} dB: Acc1={acc1:.4f}, Acc3={acc3:.4f}")
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    print("SNR (dB)\tAcc1\t\tAcc3")
    print("-" * 35)
    for i, snr in enumerate(SNR_list):
        print(f"{snr:>7}\t\t{results['acc1_results'][i]:.4f}\t\t{results['acc3_results'][i]:.4f}")
    
    # Find best performance
    best_acc1_idx = np.argmax(results['acc1_results'])
    best_snr = SNR_list[best_acc1_idx]
    best_acc1 = results['acc1_results'][best_acc1_idx]
    
    print(f"\nBest performance: {best_acc1:.4f} at {best_snr} dB")
    
    # Save results
    save_path = f'{path_to_save}.pt'
    torch.save({
        'args': args,
        'snr_values': results['snr_values'],
        'acc1_results': results['acc1_results'],
        'acc3_results': results['acc3_results'], 
        'loss_results': results['loss_results'],
        'best_snr': best_snr,
        'best_acc1': best_acc1,
        'model_info': {
            'total_params': sum(p.numel() for p in model.parameters()),
            'model_type': args.model_type,
            'latent_d': args.latent_d,
            'num_embeddings': args.num_embeddings
        }
    }, save_path)
    
    print(f"Results saved to: {save_path}")
    
    # Optional: Save CSV for easy analysis
    if args.save_csv:
        import csv
        csv_path = f'{path_to_save}.csv'
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['SNR_dB', 'Acc1', 'Acc3'])
            for i, snr in enumerate(SNR_list):
                writer.writerow([snr, results['acc1_results'][i], results['acc3_results'][i]])
        print(f"CSV results saved to: {csv_path}")


def single_snr_evaluation(args):
    """
    Evaluate model at a single SNR value.
    
    Args:
        args: Evaluation arguments
    """
    print(f"=== Single SNR Evaluation at {args.psnr} dB ===")
    
    # Load model (similar to comprehensive evaluation)
    if args.model_type == 'standard':
        model = DTJSCC_VIB(
            in_channels=2,
            latent_channels=args.latent_d,
            out_classes=args.num_classes,
            num_embeddings=args.num_embeddings
        )
    elif args.model_type == 'small':
        model = DTJSCC_VIB_Small(
            in_channels=2,
            latent_channels=args.latent_d,
            out_classes=args.num_classes,
            num_embeddings=args.num_embeddings
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(path_to_backup, map_location=args.device)
    model.load_state_dict(checkpoint['model_states'])
    model.to(args.device)
    
    # Load test data
    dataloader_test = get_data(args.dataset, 256, n_worker=8, train=False)
    
    # Create modulation
    if args.mod == 'qam':
        mod = QAM(args.num_embeddings, args.psnr)
    elif args.mod == 'psk':
        mod = PSK(args.num_embeddings, args.psnr)
    else:
        raise ValueError(f"Unknown modulation: {args.mod}")
    
    # Evaluate
    acc1, acc3, dist = eval_test(dataloader_test, model, mod, args)
    
    print(f"Results at {args.psnr} dB SNR:")
    print(f"  Top-1 Accuracy: {acc1:.4f}")
    print(f"  Top-3 Accuracy: {acc3:.4f}")
    
    return acc1, acc3


def main(args):
    """
    Main evaluation function.
    
    Args:
        args: Evaluation arguments
    """
    # Default to comprehensive evaluation for SNR range 0-5dB
    args.comprehensive = True
    evaluate_model_comprehensive(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DTJ-VIB Evaluation')
    
    # Model and data parameters
    parser.add_argument('--dataset', type=str, default='VIB', 
                       help='Dataset name')
    parser.add_argument('--dataset_root', type=str, 
                       default='/home/zsy/workplace/DHSCC/dataset/vibcls',
                       help='Path to dataset root directory')
    parser.add_argument('--model_type', type=str, default='standard', 
                       choices=['standard', 'small'],
                       help='Model variant to evaluate')
    parser.add_argument('--latent_d', type=int, default=256, 
                       help='Latent feature dimension')
    parser.add_argument('--num_classes', type=int, default=10, 
                       help='Number of output classes')
    parser.add_argument('--num_embeddings', type=int, default=16, 
                       help='Size of quantization codebook')
    
    # Communication parameters
    parser.add_argument('--mod', type=str, default='psk', choices=['psk', 'qam'],
                       help='Modulation scheme')
    parser.add_argument('--psnr', type=float, default=5.0, 
                       help='Channel PSNR in dB (for single SNR evaluation)')
    parser.add_argument('--lam', type=float, default=1e-4, 
                       help='Entropy regularization weight')
    
    # Evaluation options
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive evaluation across multiple SNRs')
    parser.add_argument('--save_csv', action='store_true',
                       help='Save results in CSV format')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda:0', 
                       help='Device for evaluation')
    
    # Model paths
    parser.add_argument('--root', type=str, default='comparison_models/DTJ_VIB/trained_models', 
                       help='Root directory containing trained models')
    parser.add_argument('--name', type=str, default='VIB-standard-num_e16-latent_d256-modpsk-psnr10.0-lam0.001',
                       help='Name of the trained model (experiment name)')
    parser.add_argument('--save_root', type=str, default='./evaluation_results', 
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Setup paths
    path_to_backup = os.path.join(args.root, args.name, 'best.pt')
    
    if not os.path.exists(path_to_backup):
        print(f"Error: Model checkpoint not found at {path_to_backup}")
        print("Please check the model name and root directory.")
        sys.exit(1)
    
    # Create results directory
    if not os.path.exists(args.save_root):
        print(f'Creating results directory: {args.save_root}')
        os.makedirs(args.save_root)
    
    path_to_save = os.path.join(args.save_root, args.name)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    print(f'Using device: {device}')
    
    # Run evaluation
    main(args)
