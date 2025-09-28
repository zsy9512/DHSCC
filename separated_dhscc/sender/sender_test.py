"""
Sender Testing Module for DHSCC

This module implements comprehensive testing for the sender component,
including performance evaluation across different SNR levels and 
feature analysis capabilities.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from tqdm import tqdm

from models.sender_model import SenderModel
from utils.file_io import ensure_dir

def test_sender_performance(model, test_loader, device, snr_range, results_dir):
    """
    Test sender performance across different SNR levels.
    
    Args:
        model: Trained SenderModel
        test_loader: Test data loader
        device: Computing device
        snr_range: Range of SNR values to test
        results_dir: Directory to save results
        
    Returns:
        dict: Performance results
    """
    model.eval()
    results = {}
    
    print(f"Testing sender across {len(snr_range)} SNR levels...")
    
    for snr_db in tqdm(snr_range, desc="SNR levels"):
        correct = 0
        total = 0
        total_commitment_loss = 0.0
        total_classification_loss = 0.0
        
        snr_results = {
            'accuracy': 0.0,
            'commitment_loss': 0.0,
            'classification_loss': 0.0,
            'feature_stats': {
                'mean': 0.0,
                'std': 0.0,
                'sparsity': 0.0
            },
            'quantization_stats': {
                'unique_indices': 0,
                'entropy': 0.0
            } if model.comm_mode in [2, 3, 4] else None
        }
        
        all_features = []
        all_indices = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                
                # Get sender output
                result = model(data, snr=snr_db)
                
                # Classification accuracy
                pred = result['pre_task_logits'].argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Losses
                classification_loss = F.cross_entropy(result['pre_task_logits'], target)
                total_classification_loss += classification_loss.item()
                total_commitment_loss += result['commitment_loss'].item()
                
                # Collect features for analysis
                all_features.append(result['encoded_features'].cpu())
                if result['indices'] is not None:
                    all_indices.append(result['indices'].cpu())
        
        # Calculate metrics
        snr_results['accuracy'] = correct / total
        snr_results['commitment_loss'] = total_commitment_loss / len(test_loader)
        snr_results['classification_loss'] = total_classification_loss / len(test_loader)
        
        # Feature statistics
        all_features = torch.cat(all_features, dim=0)
        snr_results['feature_stats']['mean'] = all_features.mean().item()
        snr_results['feature_stats']['std'] = all_features.std().item()
        snr_results['feature_stats']['sparsity'] = (all_features.abs() < 0.01).float().mean().item()
        
        # Quantization statistics (if applicable)
        if all_indices and model.comm_mode in [2, 3, 4]:
            all_indices = torch.cat(all_indices, dim=0)
            unique_indices = torch.unique(all_indices).numel()
            snr_results['quantization_stats']['unique_indices'] = unique_indices
            
            # Calculate entropy
            indices_flat = all_indices.flatten()
            indices_hist = torch.histc(indices_flat.float(), bins=256, min=0, max=255)
            indices_prob = indices_hist / indices_flat.numel()
            indices_prob = indices_prob[indices_prob > 0]  # Remove zeros
            entropy = -(indices_prob * torch.log2(indices_prob)).sum().item()
            snr_results['quantization_stats']['entropy'] = entropy
        
        results[f'snr_{snr_db:.1f}'] = snr_results
        
        print(f"SNR {snr_db:.1f}dB: Acc={snr_results['accuracy']:.4f}, "
              f"Commit={snr_results['commitment_loss']:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"sender_test_results_{timestamp}.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {results_file}")
    return results


def plot_sender_results(results, results_dir):
    """Plot and save sender test results."""
    snr_values = []
    accuracies = []
    commitment_losses = []
    classification_losses = []
    sparsities = []
    
    for snr_key, snr_data in results.items():
        if snr_key.startswith('snr_'):
            snr_val = float(snr_key.split('_')[1])
            snr_values.append(snr_val)
            accuracies.append(snr_data['accuracy'])
            commitment_losses.append(snr_data['commitment_loss'])
            classification_losses.append(snr_data['classification_loss'])
            sparsities.append(snr_data['feature_stats']['sparsity'])
    
    # Sort by SNR
    sorted_data = sorted(zip(snr_values, accuracies, commitment_losses, 
                            classification_losses, sparsities))
    snr_values, accuracies, commitment_losses, classification_losses, sparsities = zip(*sorted_data)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Sender Performance Analysis', fontsize=16)
    
    # Accuracy plot
    axes[0, 0].plot(snr_values, accuracies, 'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('SNR (dB)')
    axes[0, 0].set_ylabel('Classification Accuracy')
    axes[0, 0].set_title('Classification Accuracy vs SNR')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # Loss plots
    axes[0, 1].plot(snr_values, commitment_losses, 'r-o', linewidth=2, markersize=6, label='Commitment')
    axes[0, 1].plot(snr_values, classification_losses, 'g-o', linewidth=2, markersize=6, label='Classification')
    axes[0, 1].set_xlabel('SNR (dB)')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Losses vs SNR')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Sparsity plot
    axes[1, 0].plot(snr_values, sparsities, 'purple', marker='o', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('SNR (dB)')
    axes[1, 0].set_ylabel('Feature Sparsity')
    axes[1, 0].set_title('Feature Sparsity vs SNR')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Summary statistics
    axes[1, 1].text(0.1, 0.8, f"Best Accuracy: {max(accuracies):.4f}", transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.7, f"Worst Accuracy: {min(accuracies):.4f}", transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.6, f"Mean Accuracy: {np.mean(accuracies):.4f}", transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.5, f"Std Accuracy: {np.std(accuracies):.4f}", transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.3, f"SNR Range: {min(snr_values):.1f} - {max(snr_values):.1f} dB", transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].set_title('Summary Statistics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(results_dir, f"sender_performance_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance plot saved to {plot_file}")


def analyze_sender_features(model, test_loader, device, snr_db, results_dir, num_samples=100):
    """
    Analyze sender feature representations.
    
    Args:
        model: Trained SenderModel
        test_loader: Test data loader
        device: Computing device
        snr_db: SNR level for analysis
        results_dir: Directory to save results
        num_samples: Number of samples to analyze
    """
    model.eval()
    
    print(f"Analyzing sender features at SNR {snr_db}dB...")
    
    all_original_features = []
    all_encoded_features = []
    all_importance_weights = []
    all_labels = []
    sample_count = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            if sample_count >= num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            
            result = model(data, snr=snr_db)
            
            # Collect data
            all_encoded_features.append(result['encoded_features'].cpu())
            all_importance_weights.append(result['importance_weights'].cpu())
            all_labels.append(target.cpu())
            
            sample_count += data.size(0)
    
    # Concatenate results
    all_encoded_features = torch.cat(all_encoded_features, dim=0)[:num_samples]
    all_importance_weights = torch.cat(all_importance_weights, dim=0)[:num_samples]
    all_labels = torch.cat(all_labels, dim=0)[:num_samples]
    
    # Create feature analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Sender Feature Analysis (SNR {snr_db}dB)', fontsize=16)
    
    # Feature distribution
    axes[0, 0].hist(all_encoded_features.flatten().numpy(), bins=50, alpha=0.7)
    axes[0, 0].set_xlabel('Feature Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Encoded Feature Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Importance weight distribution
    axes[0, 1].hist(all_importance_weights.flatten().numpy(), bins=50, alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Importance Weight')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Importance Weight Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Feature magnitude by class
    unique_labels = torch.unique(all_labels)
    class_magnitudes = []
    for label in unique_labels:
        mask = all_labels == label
        class_features = all_encoded_features[mask]
        class_magnitude = torch.norm(class_features, dim=1).mean().item()
        class_magnitudes.append(class_magnitude)
    
    axes[1, 0].bar(unique_labels.numpy(), class_magnitudes, alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('Mean Feature Magnitude')
    axes[1, 0].set_title('Feature Magnitude by Class')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Feature correlation matrix (first 32 dimensions)
    feature_subset = all_encoded_features[:, :32].numpy()
    correlation_matrix = np.corrcoef(feature_subset.T)
    im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 1].set_title('Feature Correlation Matrix (32x32)')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(results_dir, f"sender_feature_analysis_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature analysis plot saved to {plot_file}")


def run_sender_comprehensive_test(model, test_loader, device, args):
    """
    Run comprehensive testing of sender model.
    
    Args:
        model: Trained SenderModel
        test_loader: Test data loader
        device: Computing device
        args: Testing arguments
        
    Returns:
        dict: Complete test results
    """
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_dir, f"sender_test_{timestamp}")
    ensure_dir(results_dir)
    
    # Define SNR range for testing
    snr_range = np.linspace(args.snr_min, args.snr_max, 11)
    
    print("Starting comprehensive sender testing...")
    print(f"Results will be saved to: {results_dir}")
    
    # Performance testing
    performance_results = test_sender_performance(
        model, test_loader, device, snr_range, results_dir
    )
    
    # Plot performance results
    plot_sender_results(performance_results, results_dir)
    
    # Feature analysis at middle SNR
    middle_snr = (args.snr_min + args.snr_max) / 2
    analyze_sender_features(model, test_loader, device, middle_snr, results_dir)
    
    # Save comprehensive results
    comprehensive_results = {
        'performance': performance_results,
        'config': {
            'comm_mode': model.comm_mode,
            'snr_range': snr_range.tolist(),
            'num_test_samples': len(test_loader.dataset),
            'model_parameters': sum(p.numel() for p in model.parameters())
        },
        'timestamp': timestamp
    }
    
    results_file = os.path.join(results_dir, "comprehensive_results.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump(comprehensive_results, f)
    
    print(f"\nComprehensive testing completed!")
    print(f"Results directory: {results_dir}")
    
    return comprehensive_results