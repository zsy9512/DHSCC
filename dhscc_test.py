"""
Testing module for DHSCC vibration classification model.

This module implements the two-stage testing strategy:
1. Data Collection: Collect test results across different SNR levels and communication modes
2. Analysis: Analyze and visualize the collected results

The testing process supports comprehensive evaluation across various channel conditions.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from collections import defaultdict

from utils.file_io import ensure_dir, save_json, save_pickle, load_pickle
from utils.signal_processing import set_seed


def collect_test_data(model, test_loader, device, args):
    """
    Stage 1: Collect test results across different SNR levels and communication modes.
    
    Args:
        model: Trained DHSCCNet model
        test_loader: Test data loader
        device: Testing device
        args: Testing arguments
        
    Returns:
        dict: Comprehensive test results with feature data
    """
    print("=" * 50)
    print("Stage 1: Data Collection")
    print("=" * 50)
    
    model.eval()
    
    # Test SNR levels
    snr_levels = args.test_snr_range
    
    # Communication modes to test
    comm_modes = [0, 1, 2, 3, 4] if hasattr(args, 'test_all_modes') and args.test_all_modes else [args.communication_mode]
    
    print(f"Testing across SNR levels: {snr_levels}")
    print(f"Testing communication modes: {comm_modes}")
    
    # Initialize results storage (simplified structure like test.py)
    results = {}
    
    class_acc_matrix = []
    snr_list = []
    total_infer_time = 0.0
    snr_count = 0
    for comm_mode in comm_modes:
        print(f"\nTesting Communication Mode {comm_mode}...")
        
        # Update model communication mode
        model.comm_mode = comm_mode
        if hasattr(model, 'wireless_channel'):
            model.wireless_channel.comm_mode = comm_mode
        
        for snr in snr_levels:
            print(f"\nTesting SNR {snr:.2f} dB...")
            
            # Collect results for this configuration
            config_results = test_single_config(model, test_loader, device, comm_mode, snr, args)
            if 'inference_time' in config_results:
                total_infer_time += config_results['inference_time']
                snr_count += 1            
            # Print results for this SNR
            print(f"SNR: {snr:.2f} dB | Accuracy: {config_results['accuracy']:.2f}% | "
                  f"BER: {config_results['ber']:.6f} | Groups: {config_results['groups_used']:.1f}")
            
            # Store results using SNR as key (similar to test.py structure)
            results[snr] = config_results

            # 收集类别识别率和推理时间
            if 'class_acc' in config_results:
                class_acc_matrix.append(config_results['class_acc'])
                snr_list.append(snr)

    # 保存类别识别率热力图数据
    if class_acc_matrix:
        class_acc_matrix = np.stack(class_acc_matrix, axis=0)
        np.save(os.path.join(args.results_dir, 'class_acc_matrix.npy'), class_acc_matrix)
        np.save(os.path.join(args.results_dir, 'snr_list.npy'), np.array(snr_list))
        print(f"Per-class accuracy matrix saved to {args.results_dir}/class_acc_matrix.npy")
    # 计算平均推理时间
    avg_infer_time = total_infer_time / snr_count if snr_count > 0 else 0.0
    print(f"\nAverage inference time across all SNRs: {avg_infer_time:.4f} seconds")
    
    print(f"\nData collection completed!")
    
    # Print summary
    print("\nResults summary:")
    for snr in sorted(results.keys()):
        if isinstance(snr, (int, float)):
            print(f"SNR: {snr:.2f} dB | Accuracy: {results[snr]['accuracy']:.2f}% | BER: {results[snr]['ber']:.6f}")
    
    return results


def test_single_config(model, test_loader, device, comm_mode, snr, args):
    """
    Test model on a single configuration (communication mode + SNR).
    
    Args:
        model: DHSCCNet model
        test_loader: Test data loader
        device: Testing device
        comm_mode: Communication mode
        snr: Signal-to-noise ratio
        args: Testing arguments
        
    Returns:
        dict: Test results for this configuration including feature data
    """
    model.eval()
    
    correct = 0
    total = 0
    total_ber = 0.0
    ber_count = 0
    groups_used = []
    
    # Collect feature data for analysis
    sent_features_batch = []
    encoded_features_batch = []
    original_features_batch = []
    attn_weights_batch = []
    
    # 统计每类识别率
    num_classes = args.num_classes if hasattr(args, 'num_classes') else 10  # 根据实际类别数调整
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    
    start_time = time.time()
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f'Mode {comm_mode}, SNR {snr}dB', leave=False)
        
        # 设置模型为推理模式，关闭探索
        if hasattr(model, 'set_inference_mode'):
            model.set_inference_mode(True)
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            # 每个新batch开始时重置group选择
            if hasattr(model, 'reset_batch_group'):
                model.reset_batch_group()
            
            # Forward pass - DHSCCNet returns multiple values
            # (classification_output, commit_loss, ber, transmit_mse, zeros_count, group_to_use, debug_info)
            outputs, commit_loss, ber, transmit_mse, zeros_count, group_to_use, debug_info = model(data, args.epochs, snr)
            # Get predictions
            _, predicted = outputs.max(1)
            
            # Update statistics
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Collect BER and group usage
            if isinstance(ber, torch.Tensor):
                total_ber += ber.item()
            else:
                total_ber += float(ber)
            ber_count += 1
            
            if isinstance(group_to_use, torch.Tensor):
                groups_used.append(group_to_use.item())
            else:
                groups_used.append(float(group_to_use))
            
            # Collect feature data from debug_info
            if 'sent_feature' in debug_info:
                sent_features_batch.append(debug_info['sent_feature'])
            if 'encoded_feature' in debug_info:
                encoded_features_batch.append(debug_info['encoded_feature'])
            if 'original_feature' in debug_info:
                original_features_batch.append(debug_info['original_feature'])
            if 'attn_weights' in debug_info:
                attn_weights_batch.append(debug_info['attn_weights'])
            
            # 统计每类识别率
            for i in range(target.size(0)):
                label = target[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
            
            # Update progress bar
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Acc': f'{current_acc:.2f}%',
                'BER': f'{total_ber/ber_count:.4f}',
                'Groups': f'{np.mean(groups_used):.1f}',
            })
    
    inference_time = time.time() - start_time
    
    # Calculate final metrics
    accuracy = 100. * correct / total
    avg_ber = total_ber / ber_count if ber_count > 0 else 0.0
    avg_groups_used = np.mean(groups_used) if groups_used else 0.0
    print(f"'inference_time': {inference_time/total}")
    # Combine feature data
    result = {
        'accuracy': accuracy,
        'ber': avg_ber,
        'groups_used': avg_groups_used,
        'inference_time': inference_time,
        'total_samples': total,
        'correct_predictions': correct,
    }
    
    # Add feature data if available
    if sent_features_batch:
        result['sent_features'] = torch.cat(sent_features_batch, dim=0).cpu().numpy()
    if encoded_features_batch:
        result['encoded_features'] = torch.cat(encoded_features_batch, dim=0).cpu().numpy()
    if original_features_batch:
        result['original_features'] = torch.cat(original_features_batch, dim=0).cpu().numpy()
    if attn_weights_batch:
        result['attn_weights'] = torch.cat(attn_weights_batch, dim=0).cpu().numpy()
    
    # 计算每类准确率
    class_acc = class_correct / np.maximum(class_total, 1)
    result['class_acc'] = class_acc
    
    return result


class ResultsAnalyzer:
    """
    分析测试结果的类，包括绘制各种图表和统计分析
    基于test.py中的ResultsAnalyzer类重构
    """
    
    def __init__(self, results, save_dir):
        """
        初始化分析器
        
        Args:
            results: collect_test_data返回的结果字典
            save_dir: 保存分析结果的目录
        """
        self.results = results
        self.save_dir = save_dir
        self.snr_levels = sorted([k for k in results.keys() if isinstance(k, (int, float))])
        ensure_dir(save_dir)
        
        # 打印不同SNR下的测试结果
        self.print_snr_results()
    
    def print_snr_results(self):
        """打印不同SNR下的测试结果（acc, ber, group_to_use）"""
        print("\n" + "=" * 60)
        print("DETAILED SNR RESULTS")
        print("=" * 60)
        print(f"{'SNR (dB)':<10} {'Accuracy (%)':<15} {'BER':<12} {'Groups Used':<12}")
        print("-" * 60)
        
        for snr in self.snr_levels:
            result = self.results[snr]
            print(f"{snr:<10.2f} {result['accuracy']:<15.2f} {result['ber']:<12.6f} {result['groups_used']:<12.1f}")
        
        print("=" * 60)
    
    def plot_performance_curves(self):
        """绘制精度、BER和组使用数量曲线"""
        plt.figure(figsize=(18, 5))
        
        # 提取数据
        accuracies = [self.results[snr]['accuracy'] for snr in self.snr_levels]
        bers = [self.results[snr]['ber'] for snr in self.snr_levels]
        groups_used = [self.results[snr]['groups_used'] for snr in self.snr_levels]
        
        # 精度曲线
        plt.subplot(1, 3, 1)
        plt.plot(self.snr_levels, accuracies, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('SNR (dB)')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy vs. SNR Level')
        plt.grid(True, alpha=0.3)
        
        # BER曲线
        plt.subplot(1, 3, 2)
        plt.plot(self.snr_levels, bers, 'r-s', linewidth=2, markersize=6)
        plt.xlabel('SNR (dB)')
        plt.ylabel('Bit Error Rate (BER)')
        plt.title('BER vs. SNR Level')
        if max(bers) > 0:
            plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Groups Used曲线
        plt.subplot(1, 3, 3)
        plt.plot(self.snr_levels, groups_used, 'g-^', linewidth=2, markersize=6)
        plt.xlabel('SNR (dB)')
        plt.ylabel('Groups Used')
        plt.title('Groups Used vs. SNR Level')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 8.5)  # 限制y轴范围，因为最大组数是8
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'performance_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance curves saved to {self.save_dir}/performance_curves.png")
    
    def plot_feature_heatmaps(self):
        """绘制三种特征数据在不同SNR下的热力图，顺序为original->encoded->sent"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # 检查是否有特征数据
        if 'original_features' not in self.results[self.snr_levels[0]]:
            print("Warning: No feature data found for heatmap analysis")
            return
        
        # 1. Original Features热力图
        original_data = []
        encoded_data = []
        sent_data = []
        
        for snr in self.snr_levels:
            if 'original_features' in self.results[snr]:
                original_data.append(self.results[snr]['original_features'])
            if 'encoded_features' in self.results[snr]:
                encoded_data.append(self.results[snr]['encoded_features'])
            if 'sent_features' in self.results[snr]:
                sent_data.append(self.results[snr]['sent_features'])
        
        if original_data:
            original_mean = np.array([data.mean(axis=0) for data in original_data])
            im1 = axes[0].imshow(original_mean, aspect='auto', cmap='viridis', interpolation='nearest')
            axes[0].set_title('Original Features Heatmap Across SNR Levels')
            axes[0].set_xlabel('Feature Dimension')
            axes[0].set_ylabel('SNR Level Index')
            axes[0].set_yticks(range(len(self.snr_levels)))
            axes[0].set_yticklabels([f'{snr:.2f}' for snr in self.snr_levels])
            plt.colorbar(im1, ax=axes[0], label='Feature Value')
        
        # 2. Encoded Features热力图
        if encoded_data:
            encoded_mean = np.array([data.mean(axis=0) for data in encoded_data])
            im2 = axes[1].imshow(encoded_mean, aspect='auto', cmap='plasma', interpolation='nearest')
            axes[1].set_title('Encoded Features Heatmap Across SNR Levels')
            axes[1].set_xlabel('Feature Dimension')
            axes[1].set_ylabel('SNR Level Index')
            axes[1].set_yticks(range(len(self.snr_levels)))
            axes[1].set_yticklabels([f'{snr:.2f}' for snr in self.snr_levels])
            plt.colorbar(im2, ax=axes[1], label='Feature Value')
        
        # 3. Sent Features热力图
        if sent_data:
            sent_mean = np.array([data.mean(axis=0) for data in sent_data])
            im3 = axes[2].imshow(sent_mean, aspect='auto', cmap='inferno', interpolation='nearest')
            axes[2].set_title('Sent Features Heatmap Across SNR Levels')
            axes[2].set_xlabel('Feature Dimension')
            axes[2].set_ylabel('SNR Level Index')
            axes[2].set_yticks(range(len(self.snr_levels)))
            axes[2].set_yticklabels([f'{snr:.2f}' for snr in self.snr_levels])
            plt.colorbar(im3, ax=axes[2], label='Feature Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'feature_heatmaps.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature heatmaps saved to {self.save_dir}/feature_heatmaps.png")
    
    def plot_attention_analysis(self):
        """单独绘制注意力权重的分析图，只包含热力图和统计曲线"""
        # 检查是否有注意力权重数据
        if 'attn_weights' not in self.results[self.snr_levels[0]]:
            print("Warning: No attention weights data found for analysis")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 收集注意力权重数据
        attn_data = []
        for snr in self.snr_levels:
            if 'attn_weights' in self.results[snr]:
                attn_data.append(self.results[snr]['attn_weights'])
        
        if not attn_data:
            print("Warning: No attention weights data available")
            return
        
        # 1. 注意力权重热力图
        attn_mean = np.array([data.mean(axis=0) for data in attn_data])
        
        im1 = axes[0].imshow(attn_mean, aspect='auto', cmap='RdYlBu_r', interpolation='nearest', 
                    vmin=attn_mean.min(), vmax=attn_mean.max())
        axes[0].set_title('Attention Weights Heatmap Across SNR Levels')
        axes[0].set_xlabel('Attention Dimension')
        axes[0].set_ylabel('SNR Level Index')
        axes[0].set_yticks(range(len(self.snr_levels)))
        axes[0].set_yticklabels([f'{snr:.2f}' for snr in self.snr_levels])
        plt.colorbar(im1, ax=axes[0], label='Attention Weight')
        
        # 2. 注意力权重统计曲线
        attn_means = []
        attn_stds = []
        attn_maxs = []
        attn_mins = []
        
        for snr in self.snr_levels:
            if 'attn_weights' in self.results[snr]:
                data = self.results[snr]['attn_weights'].flatten()
                attn_means.append(np.mean(data))
                attn_stds.append(np.std(data))
                attn_maxs.append(np.max(data))
                attn_mins.append(np.min(data))
        
        attn_means = np.array(attn_means)
        attn_stds = np.array(attn_stds)
        
        axes[1].plot(self.snr_levels, attn_means, 'purple', linewidth=2, marker='o', label='Mean')
        axes[1].fill_between(self.snr_levels, attn_means - attn_stds, attn_means + attn_stds, 
                        color='purple', alpha=0.3, label='±1 Std')
        axes[1].fill_between(self.snr_levels, attn_mins, attn_maxs, 
                        color='purple', alpha=0.1, label='Min-Max Range')
        axes[1].set_title('Attention Weights Statistics vs. SNR')
        axes[1].set_xlabel('SNR (dB)')
        axes[1].set_ylabel('Attention Weight')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'attention_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Attention analysis saved to {self.save_dir}/attention_analysis.png")
    
    def plot_class_acc_heatmap(self):
        """绘制每个SNR下每个类别的平均识别率热力图"""
        class_acc_path = os.path.join(self.save_dir, '../class_acc_matrix.npy')
        snr_path = os.path.join(self.save_dir, '../snr_list.npy')
        if not os.path.exists(class_acc_path) or not os.path.exists(snr_path):
            print("No per-class accuracy data found for heatmap.")
            return
        class_acc_matrix = np.load(class_acc_path)
        snr_list = np.load(snr_path)
        plt.figure(figsize=(10, 6))
        im = plt.imshow(class_acc_matrix, aspect='auto', cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
        plt.colorbar(im, label='Accuracy')
        plt.xlabel('Class Index')
        plt.ylabel('SNR (dB)')
        plt.yticks(range(len(snr_list)), [f'{snr:.2f}' for snr in snr_list])
        plt.title('Per-Class Accuracy Heatmap Across SNR Levels')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'class_acc_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Per-class accuracy heatmap saved to {self.save_dir}/class_acc_heatmap.png")
    
    def analyze_all(self):
        """执行所有分析"""
        print("\nStarting comprehensive analysis...")
        
        # 1. 绘制性能曲线（包括精度、BER、groups_used）
        print("1. Plotting performance curves...")
        self.plot_performance_curves()
        
        # 2. 绘制特征热力图（original->encoded->sent）
        print("2. Plotting feature heatmaps...")
        self.plot_feature_heatmaps()
        
        # 3. 绘制注意力权重分析
        print("3. Plotting attention analysis...")
        self.plot_attention_analysis()
        
        # 4. 绘制每个类别在不同SNR下的准确率热力图
        print("4. Plotting per-class accuracy heatmap...")
        self.plot_class_acc_heatmap()
        
        print(f"\nAll analysis completed. Results saved to {self.save_dir}")
        return True


def analyze_results(results, args):
    """
    Stage 2: Analyze and visualize the collected test results.
    
    Args:
        results: Test results from data collection stage
        args: Testing arguments
        
    Returns:
        dict: Analysis results and visualization paths
    """
    print("=" * 50)
    print("Stage 2: Results Analysis")
    print("=" * 50)
    
    # Create analysis directory
    analysis_dir = os.path.join(args.results_dir, 'analysis')
    ensure_dir(analysis_dir)
    
    # 使用ResultsAnalyzer进行分析
    analyzer = ResultsAnalyzer(results, analysis_dir)
    analyzer.analyze_all()
    
    return {
        'analysis_completed': True,
        'analysis_dir': analysis_dir
    }


def prepare_model_for_testing(model):
    """
    准备模型进行测试，确保自适应组选择器处于正确状态
    
    Args:
        model: DHSCCNet model
    """
    print("\n📋 Preparing model for testing...")
    
    # 确保模型处于评估模式
    model.eval()
    
    if hasattr(model, 'adaptive_group_selector') and model.adaptive_group_selector is not None:
        print("🔧 Found adaptive group selector, preparing for testing...")
        
        # 检查是否已经完成最终化
        is_finalized = hasattr(model.adaptive_group_selector, '_finalized') and model.adaptive_group_selector._finalized
        
        if not is_finalized:
            print("⚠️  Adaptive selector not finalized, applying monotonicity constraints...")
            model.adaptive_group_selector.finalize_training()
            model.adaptive_group_selector._finalized = True
        else:
            print("✓ Adaptive selector already finalized")
            # 即使已经最终化，也要确保单调性约束被应用
            print("🔧 Re-applying monotonicity constraints to ensure correctness...")
            model.adaptive_group_selector._enforce_monotonicity()
            model.adaptive_group_selector._validate_group_choices()
        
        # 设置为推理模式
        model.adaptive_group_selector.set_inference_mode(True)
        
        # 验证策略的单调性
        strategy = model.adaptive_group_selector.get_strategy_summary()
        groups = [g for _, g in strategy]
        
        # 检查单调性
        is_monotonic = all(groups[i] <= groups[i+1] for i in range(len(groups)-1))
        
        if is_monotonic:
            print("✓ Adaptive selector strategy is monotonic")
        else:
            print("⚠️  Strategy still not monotonic, enforcing constraints again...")
            model.adaptive_group_selector._enforce_monotonicity()
            model.adaptive_group_selector._validate_group_choices()
            
            # 再次检查
            strategy = model.adaptive_group_selector.get_strategy_summary()
            groups = [g for _, g in strategy]
            is_monotonic = all(groups[i] <= groups[i+1] for i in range(len(groups)-1))
            print(f"✓ Monotonicity check after re-enforcement: {is_monotonic}")
        
        print("✓ Adaptive selector prepared for testing")
        
        # 打印当前策略摘要
        strategy_summary = model.adaptive_group_selector.get_strategy_summary()
        print("\nCurrent strategy for testing:")
        for snr, group in strategy_summary[::4]:  # 只显示部分策略避免输出过长
            print(f"  SNR {snr:.1f}dB -> {group} groups")
        print("  ...")
    else:
        print("✓ No adaptive selector found, using fixed group selection")
    
    print("📋 Model preparation completed\n")


def run_full_test(model, test_loader, device, args):
    """
    Run the complete two-stage testing process.
    
    Args:
        model: Trained DHSCCNet model
        test_loader: Test data loader
        device: Testing device
        args: Testing arguments
        
    Returns:
        dict: Complete test and analysis results
    """
    print("Starting comprehensive two-stage testing...")
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Ensure results directory exists
    ensure_dir(args.results_dir)
    
    # 确保自适应组选择器处于正确的测试状态
    prepare_model_for_testing(model)
    
    start_time = time.time()
    
    try:
        # Stage 1: Data Collection
        test_results = collect_test_data(model, test_loader, device, args)
        
        # Save raw test results
        raw_results_path = os.path.join(args.results_dir, 'raw_test_results.pkl')
        save_pickle(test_results, raw_results_path)
        print(f"Raw test results saved to: {raw_results_path}")
        
        # Stage 2: Analysis
        analysis_results = analyze_results(test_results, args)
        
        # Save analysis results
        analysis_results_path = os.path.join(args.results_dir, 'analysis_results.pkl')
        save_pickle(analysis_results, analysis_results_path)
        print(f"Analysis results saved to: {analysis_results_path}")
        
        # Combine all results
        complete_results = {
            'test_results': test_results,
            'analysis_results': analysis_results,
            'total_test_time': time.time() - start_time,
            'test_args': vars(args)
        }
        
        # Save complete results
        complete_results_path = os.path.join(args.results_dir, 'complete_results.pkl')
        save_pickle(complete_results, complete_results_path)
        
        print("=" * 50)
        print("Testing completed successfully!")
        print(f"Total testing time: {complete_results['total_test_time']:.2f} seconds")
        print(f"Results saved in: {args.results_dir}")
        print("=" * 50)
        
        return complete_results
        
    except Exception as e:
        print(f"Testing failed with error: {str(e)}")
        raise e


def test(model, test_loader, epoch, snr, device):
    """
    简单测试函数，用于训练期间的快速验证
    
    Args:
        model: DHSCCNet model
        test_loader: Test data loader
        epoch: Current training epoch
        snr: Signal-to-noise ratio for testing
        device: Computing device
        
    Returns:
        tuple: (test_loss, test_accuracy, avg_group_to_use)
    """
    model.eval()
    
    # 设置测试模式
    original_test_flag = getattr(model, 'test', 0)
    model.test = 1
    
    # 如果是final阶段，确保自适应选择器已经最终化
    if hasattr(model, 'is_final_phase') and model.is_final_phase(epoch):
        if hasattr(model, 'adaptive_group_selector') and model.adaptive_group_selector is not None:
            if not hasattr(model.adaptive_group_selector, '_finalized') or not model.adaptive_group_selector._finalized:
                print("🔧 Finalizing adaptive selector for final phase testing...")
                model.adaptive_group_selector.finalize_training()
                model.adaptive_group_selector.set_inference_mode(True)
    
    total_loss = 0.0
    correct = 0
    total = 0
    group_sum = 0.0
    batch_count = 0
    
    criterion = nn.CrossEntropyLoss()
    
    try:
        with torch.no_grad():
            # 设置模型为推理模式，关闭探索
            if hasattr(model, 'set_inference_mode'):
                model.set_inference_mode(True)
            
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                # 每个新batch开始时重置group选择
                if hasattr(model, 'reset_batch_group'):
                    model.reset_batch_group()
                
                # Forward pass
                outputs, commit_loss, ber, transmit_mse, zeros_count, group_to_use, debug_info = model(data, epoch, snr)
                
                # 计算损失 (与训练时相同的损失函数)
                classification_loss = criterion(outputs, target)
                anneal_ratio = min(1.0, epoch / 20)
                loss = classification_loss + 0.1 * commit_loss + 0.05 * anneal_ratio * transmit_mse
                
                total_loss += loss.item()
                
                # 计算准确率
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # 统计group使用情况
                if isinstance(group_to_use, torch.Tensor):
                    group_sum += group_to_use.item()
                elif isinstance(group_to_use, (int, float)):
                    group_sum += float(group_to_use)
                else:
                    group_sum += 0.0
                batch_count += 1
    
    except Exception as e:
        print(f"Warning: Error during testing: {e}")
        return 0.0, 0.0, 0.0
    
    finally:
        # 恢复训练模式标志和推理模式
        model.test = original_test_flag
        if hasattr(model, 'set_inference_mode'):
            model.set_inference_mode(False)
    
    test_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    test_accuracy = 100. * correct / total if total > 0 else 0.0
    avg_group_to_use = group_sum / batch_count if batch_count > 0 else 0.0
    
    return test_loss, test_accuracy, avg_group_to_use
