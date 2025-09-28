"""
Receiver Testing Module for DHSCC

This module implements comprehensive testing for the receiver component,
including performance evaluation across different SNR levels and 
robustness analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from tqdm import tqdm

from models.receiver_model import ReceiverModel
from utils.file_io import ensure_dir

def simulate_channel_effects_for_testing(features, snr_db, comm_mode, error_types=['gaussian']):
    """
    Simulate various channel effects for comprehensive receiver testing.
    
    Args:
        features: Clean features [batch_size, feature_dim]
        snr_db: Signal-to-noise ratio in dB
        comm_mode: Communication mode
        error_types: Types of errors to simulate
        
    Returns:
        dict: Features with different error types
    """
    snr_linear = 10 ** (snr_db / 10)
    noise_variance = 1.0 / snr_linear
    
    corrupted_features = {}
    
    for error_type in error_types:
        if error_type == 'gaussian':
            # Standard Gaussian noise
            noise = torch.randn_like(features) * torch.sqrt(torch.tensor(noise_variance))
            corrupted_features['gaussian'] = features + noise
            
        elif error_type == 'impulse':
            # Impulse noise (sparse but high magnitude)
            impulse_prob = 0.05  # 5% impulse noise
            impulse_mask = torch.rand_like(features) < impulse_prob
            impulse_noise = torch.randn_like(features) * 3.0  # High magnitude
            corrupted = features.clone()
            corrupted[impulse_mask] = impulse_noise[impulse_mask]
            corrupted_features['impulse'] = corrupted
            
        elif error_type == 'fading':
            # Simulate fading effects
            fading_coeff = torch.rand_like(features) * 0.5 + 0.5  # [0.5, 1.0]
            noise = torch.randn_like(features) * torch.sqrt(torch.tensor(noise_variance))
            corrupted_features['fading'] = features * fading_coeff + noise
            
        elif error_type == 'quantization':
            # Simulation of quantization errors
            if comm_mode in [2, 3, 4]:
                # More aggressive quantization simulation
                quantized = torch.round(features * 8) / 8  # 8-level quantization
                noise = torch.randn_like(features) * torch.sqrt(torch.tensor(noise_variance))
                corrupted_features['quantization'] = quantized + noise
            else:
                corrupted_features['quantization'] = corrupted_features.get('gaussian', features)
                
        elif error_type == 'bit_errors':
            # Simulate bit errors in digital transmission
            if comm_mode in [3, 4]:
                bit_error_prob = max(0.001, (1.0 / snr_linear) * 0.1)
                error_mask = torch.rand_like(features) < bit_error_prob
                error_values = torch.randn_like(features) * 1.0
                corrupted = features.clone()
                corrupted[error_mask] = error_values[error_mask]
                corrupted_features['bit_errors'] = corrupted
            else:
                corrupted_features['bit_errors'] = corrupted_features.get('gaussian', features)
    
    return corrupted_features


def test_receiver_performance(model, test_loader, device, snr_range, comm_mode, results_dir):
    """
    Test receiver performance across different SNR levels and error types.
    
    Args:
        model: Trained ReceiverModel
        test_loader: Test data loader
        device: Computing device
        snr_range: Range of SNR values to test
        comm_mode: Communication mode
        results_dir: Directory to save results
        
    Returns:
        dict: Performance results
    """
    model.eval()
    results = {}
    
    error_types = ['gaussian', 'impulse', 'fading', 'quantization', 'bit_errors']
    
    print(f"Testing receiver across {len(snr_range)} SNR levels with {len(error_types)} error types...")
    
    for snr_db in tqdm(snr_range, desc="SNR levels"):
        snr_results = {}
        
        for error_type in error_types:
            correct = 0
            total = 0
            total_loss = 0.0
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(test_loader):
                    data, target = data.to(device), target.to(device)
                    
                    # Simulate received features with specific error type
                    # First convert raw data to simulated encoded features
                    batch_size = data.size(0)
                    encoded_features = torch.randn(batch_size, 128, device=device)
                    
                    # Apply channel effects
                    corrupted_features_dict = simulate_channel_effects_for_testing(
                        encoded_features, snr_db, comm_mode, [error_type]
                    )
                    received_features = corrupted_features_dict[error_type]
                    
                    # Get receiver output
                    result = model(received_features, snr=snr_db)
                    
                    # Calculate loss and accuracy
                    loss = F.cross_entropy(result['classification_output'], target)
                    total_loss += loss.item()
                    
                    pred = result['classification_output'].argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
            
            accuracy = correct / total
            avg_loss = total_loss / len(test_loader)
            
            snr_results[error_type] = {
                'accuracy': accuracy,
                'loss': avg_loss
            }
        
        results[f'snr_{snr_db:.1f}'] = snr_results
        
        # Print summary for this SNR
        gaussian_acc = snr_results['gaussian']['accuracy']
        worst_acc = min([res['accuracy'] for res in snr_results.values()])
        print(f"SNR {snr_db:.1f}dB: Gaussian={gaussian_acc:.4f}, Worst={worst_acc:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"receiver_test_results_{timestamp}.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {results_file}")
    return results


def plot_receiver_results(results, results_dir):
    """Plot and save receiver test results."""
    snr_values = []
    error_types = ['gaussian', 'impulse', 'fading', 'quantization', 'bit_errors']
    
    # Organize data
    results_by_error = {error_type: {'snr': [], 'accuracy': [], 'loss': []} 
                       for error_type in error_types}
    
    for snr_key, snr_data in results.items():
        if snr_key.startswith('snr_'):
            snr_val = float(snr_key.split('_')[1])
            for error_type in error_types:
                if error_type in snr_data:
                    results_by_error[error_type]['snr'].append(snr_val)
                    results_by_error[error_type]['accuracy'].append(snr_data[error_type]['accuracy'])
                    results_by_error[error_type]['loss'].append(snr_data[error_type]['loss'])
    
    # Sort by SNR
    for error_type in error_types:
        sorted_data = sorted(zip(results_by_error[error_type]['snr'],
                               results_by_error[error_type]['accuracy'],
                               results_by_error[error_type]['loss']))
        if sorted_data:
            snr_vals, accs, losses = zip(*sorted_data)
            results_by_error[error_type]['snr'] = list(snr_vals)
            results_by_error[error_type]['accuracy'] = list(accs)
            results_by_error[error_type]['loss'] = list(losses)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Receiver Performance Analysis', fontsize=16)
    
    # Accuracy comparison across error types
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i, error_type in enumerate(error_types):
        if results_by_error[error_type]['snr']:
            axes[0, 0].plot(results_by_error[error_type]['snr'], 
                          results_by_error[error_type]['accuracy'], 
                          color=colors[i], marker='o', linewidth=2, 
                          markersize=4, label=error_type.capitalize())
    
    axes[0, 0].set_xlabel('SNR (dB)')
    axes[0, 0].set_ylabel('Classification Accuracy')
    axes[0, 0].set_title('Accuracy vs SNR (Different Error Types)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # Loss comparison
    for i, error_type in enumerate(error_types):
        if results_by_error[error_type]['snr']:
            axes[0, 1].plot(results_by_error[error_type]['snr'], 
                          results_by_error[error_type]['loss'], 
                          color=colors[i], marker='o', linewidth=2, 
                          markersize=4, label=error_type.capitalize())
    
    axes[0, 1].set_xlabel('SNR (dB)')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss vs SNR (Different Error Types)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Robustness analysis - performance degradation
    if results_by_error['gaussian']['snr']:
        gaussian_acc = np.array(results_by_error['gaussian']['accuracy'])
        degradations = {}
        
        for error_type in error_types[1:]:  # Skip gaussian
            if results_by_error[error_type]['snr']:
                error_acc = np.array(results_by_error[error_type]['accuracy'])
                degradation = gaussian_acc - error_acc
                degradations[error_type] = degradation
                
                axes[1, 0].plot(results_by_error[error_type]['snr'], degradation, 
                              marker='o', linewidth=2, markersize=4, 
                              label=error_type.capitalize())
        
        axes[1, 0].set_xlabel('SNR (dB)')
        axes[1, 0].set_ylabel('Accuracy Degradation')
        axes[1, 0].set_title('Robustness Analysis (vs Gaussian)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, None])
    
    # Summary statistics
    if results_by_error['gaussian']['accuracy']:
        best_acc = max(results_by_error['gaussian']['accuracy'])
        worst_acc = min(results_by_error['gaussian']['accuracy'])
        mean_acc = np.mean(results_by_error['gaussian']['accuracy'])
        
        axes[1, 1].text(0.1, 0.8, f"Best Accuracy (Gaussian): {best_acc:.4f}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.7, f"Worst Accuracy (Gaussian): {worst_acc:.4f}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.6, f"Mean Accuracy (Gaussian): {mean_acc:.4f}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        
        # Find most robust error type
        if len(results_by_error['impulse']['accuracy']) > 0:
            impulse_mean = np.mean(results_by_error['impulse']['accuracy'])
            axes[1, 1].text(0.1, 0.4, f"Mean Accuracy (Impulse): {impulse_mean:.4f}", 
                           transform=axes[1, 1].transAxes, fontsize=12)
    
    axes[1, 1].set_title('Summary Statistics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(results_dir, f"receiver_performance_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance plot saved to {plot_file}")


def analyze_receiver_robustness(model, test_loader, device, snr_db, comm_mode, results_dir):
    """
    Analyze receiver robustness to different types of errors.
    
    Args:
        model: Trained ReceiverModel
        test_loader: Test data loader
        device: Computing device
        snr_db: SNR level for analysis
        comm_mode: Communication mode
        results_dir: Directory to save results
    """
    model.eval()
    
    print(f"Analyzing receiver robustness at SNR {snr_db}dB...")
    
    error_types = ['gaussian', 'impulse', 'fading', 'quantization', 'bit_errors']
    num_samples = 500
    
    robustness_results = {}
    all_predictions = {}
    all_targets = []
    
    with torch.no_grad():
        sample_count = 0
        for data, target in test_loader:
            if sample_count >= num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            
            # Simulate encoded features
            encoded_features = torch.randn(batch_size, 128, device=device)
            
            # Test with different error types
            for error_type in error_types:
                if error_type not in all_predictions:
                    all_predictions[error_type] = []
                
                corrupted_features_dict = simulate_channel_effects_for_testing(
                    encoded_features, snr_db, comm_mode, [error_type]
                )
                received_features = corrupted_features_dict[error_type]
                
                result = model(received_features, snr=snr_db)
                predictions = result['classification_output'].cpu()
                all_predictions[error_type].append(predictions)
            
            all_targets.append(target.cpu())
            sample_count += batch_size
    
    # Concatenate results
    all_targets = torch.cat(all_targets, dim=0)[:num_samples]
    for error_type in error_types:
        all_predictions[error_type] = torch.cat(all_predictions[error_type], dim=0)[:num_samples]
    
    # Calculate robustness metrics
    for error_type in error_types:
        predictions = all_predictions[error_type]
        predicted_classes = predictions.argmax(dim=1)
        
        accuracy = (predicted_classes == all_targets).float().mean().item()
        confidence = F.softmax(predictions, dim=1).max(dim=1)[0].mean().item()
        
        robustness_results[error_type] = {
            'accuracy': accuracy,
            'confidence': confidence,
            'predictions': predicted_classes.numpy(),
            'probabilities': F.softmax(predictions, dim=1).numpy()
        }
    
    # Create robustness analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Receiver Robustness Analysis (SNR {snr_db}dB)', fontsize=16)
    
    # Accuracy comparison
    error_names = list(robustness_results.keys())
    accuracies = [robustness_results[et]['accuracy'] for et in error_names]
    confidences = [robustness_results[et]['confidence'] for et in error_names]
    
    axes[0, 0].bar(range(len(error_names)), accuracies, alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('Error Type')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy by Error Type')
    axes[0, 0].set_xticks(range(len(error_names)))
    axes[0, 0].set_xticklabels([name.capitalize() for name in error_names], rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # Confidence comparison
    axes[0, 1].bar(range(len(error_names)), confidences, alpha=0.7, color='lightcoral')
    axes[0, 1].set_xlabel('Error Type')
    axes[0, 1].set_ylabel('Mean Confidence')
    axes[0, 1].set_title('Confidence by Error Type')
    axes[0, 1].set_xticks(range(len(error_names)))
    axes[0, 1].set_xticklabels([name.capitalize() for name in error_names], rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Confusion between error types (using gaussian as reference)
    gaussian_preds = robustness_results['gaussian']['predictions']
    confusion_scores = []
    
    for error_type in error_names[1:]:  # Skip gaussian
        error_preds = robustness_results[error_type]['predictions']
        agreement = (gaussian_preds == error_preds).mean()
        confusion_scores.append(agreement)
    
    if confusion_scores:
        axes[1, 0].bar(range(len(confusion_scores)), confusion_scores, alpha=0.7, color='lightgreen')
        axes[1, 0].set_xlabel('Error Type')
        axes[1, 0].set_ylabel('Agreement with Gaussian')
        axes[1, 0].set_title('Prediction Consistency')
        axes[1, 0].set_xticks(range(len(confusion_scores)))
        axes[1, 0].set_xticklabels([name.capitalize() for name in error_names[1:]], rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
    
    # Summary statistics
    best_error_type = max(robustness_results.keys(), key=lambda x: robustness_results[x]['accuracy'])
    worst_error_type = min(robustness_results.keys(), key=lambda x: robustness_results[x]['accuracy'])
    
    axes[1, 1].text(0.1, 0.8, f"Best Error Type: {best_error_type.capitalize()}", 
                   transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.7, f"Best Accuracy: {robustness_results[best_error_type]['accuracy']:.4f}", 
                   transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.5, f"Worst Error Type: {worst_error_type.capitalize()}", 
                   transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.4, f"Worst Accuracy: {robustness_results[worst_error_type]['accuracy']:.4f}", 
                   transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.2, f"Robustness Gap: {robustness_results[best_error_type]['accuracy'] - robustness_results[worst_error_type]['accuracy']:.4f}", 
                   transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].set_title('Robustness Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(results_dir, f"receiver_robustness_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Robustness analysis plot saved to {plot_file}")
    
    return robustness_results


def run_receiver_comprehensive_test(model, test_loader, device, args):
    """
    Run comprehensive testing of receiver model.
    
    Args:
        model: Trained ReceiverModel
        test_loader: Test data loader
        device: Computing device
        args: Testing arguments
        
    Returns:
        dict: Complete test results
    """
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_dir, f"receiver_test_{timestamp}")
    ensure_dir(results_dir)
    
    # Define SNR range for testing
    snr_range = np.linspace(args.snr_min, args.snr_max, 11)
    
    print("Starting comprehensive receiver testing...")
    print(f"Results will be saved to: {results_dir}")
    
    # Performance testing
    performance_results = test_receiver_performance(
        model, test_loader, device, snr_range, args.communication_mode, results_dir
    )
    
    # Plot performance results
    plot_receiver_results(performance_results, results_dir)
    
    # Robustness analysis at middle SNR
    middle_snr = (args.snr_min + args.snr_max) / 2
    robustness_results = analyze_receiver_robustness(
        model, test_loader, device, middle_snr, args.communication_mode, results_dir
    )
    
    # Save comprehensive results
    comprehensive_results = {
        'performance': performance_results,
        'robustness': robustness_results,
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