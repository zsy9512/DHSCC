#!/usr/bin/env python3
"""
Variable Code Rate Digital Transmission Test
============================================

测试mode4在不同group_to_use下的性能：
1. group_to_use范围: 2-8 (对应不同信息位数: 32-128)
2. 码长固定为256位，码率从32/256到128/256
3. 4QAM调制，SC译码
4. SNR扫描范围: 0-5dB，11个点
5. 分析不同码率下的BER性能

测试参数：
- 索引形状: [batch_size, group_to_use, 4] 
- group_to_use范围: 2-8
- 索引范围: 0-15 (4 bits per index)
- 信息位数: group_to_use * 4 * 4 (32-128 bits)
- 码长: n=256 (固定)
- 码率: k/n (0.125-0.5)
- 调制方式: 4QAM (2 bits per symbol)
- 译码算法: SC译码
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
import sys
import os
import pickle

# 添加路径以导入wireless_comm模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wireless_comm.interface import WirelessInterface


class VariableCodeRateTester:
    """可变码率数字传输测试器"""
    
    def __init__(self, batch_size: int = 256, device: str = "cuda"):
        """
        初始化测试器
        
        Args:
            batch_size: 批量大小
            device: 计算设备
        """
        self.batch_size = batch_size
        self.device = device
        self.group_size = 4  # 每组的大小固定为4
        self.bits_per_index = 4  # 每个索引4比特
        self.n = 256  # 码长固定为256
        
        print(f"=== 可变码率数字传输测试器初始化 ===")
        print(f"批量大小: {batch_size}")
        print(f"设备: {device}")
        print(f"码长: {self.n}")
        print(f"索引形状: [{batch_size}, group_to_use, {self.group_size}]")
        print()
        
    def _generate_test_indices(self, group_to_use: int) -> torch.Tensor:
        """生成指定group_to_use的测试索引数据"""
        torch.manual_seed(42)  # 固定随机种子以便重现
        indices = torch.randint(
            0, 16,  # 索引范围 0-15
            (self.batch_size, group_to_use, self.group_size),
            device=self.device
        )
        return indices
        
    def _setup_wireless_interface(self, group_to_use: int) -> WirelessInterface:
        """为指定的group_to_use设置mode4无线传输接口"""
        k = group_to_use * self.group_size * self.bits_per_index  # 信息位数
        code_rate = k / self.n
        
        print(f"设置group_to_use={group_to_use}: k={k}, n={self.n}, 码率={code_rate:.3f}")
        
        # 模式4: 编码数字传输（极化码）
        interface_mode4 = WirelessInterface(
            comm_mode=4,
            k=k,  # 信息位数
            n=self.n,  # 码长固定
            channel_model='awgn',
            constellation_type='qam',  # 4QAM调制
            bits_per_symbol=2,
            decoder_type='SC',  # SC译码
            list_size=8,
            device=self.device
        )
        
        return interface_mode4
        
    def test_single_config(self, group_to_use: int, snr_db: float) -> Dict:
        """测试单一配置下的性能"""
        # 生成测试数据
        indices = self._generate_test_indices(group_to_use)
        
        # 设置接口
        interface = self._setup_wireless_interface(group_to_use)
        
        # 执行传输
        start_time = time.time()
        result = interface.transmit_mode4(
            indices=indices,
            snr_db=snr_db,
            bits_per_index=self.bits_per_index,
            group_to_use=group_to_use
        )
        transmission_time = time.time() - start_time
        
        k = group_to_use * self.group_size * self.bits_per_index
        code_rate = k / self.n
        
        return {
            'group_to_use': group_to_use,
            'k': k,
            'n': self.n,
            'code_rate': code_rate,
            'snr_db': snr_db,
            'bit_error_rate': result['bit_error_rate'],
            'index_error_rate': result['index_error_rate'],
            'transmission_time': transmission_time,
            'received_indices': result['received_indices'],
            'original_indices': indices
        }
        
    def run_group_snr_sweep(self, group_range: List[int], snr_range: List[float]) -> Dict:
        """
        运行group和SNR的双重扫描测试
        
        Args:
            group_range: group_to_use范围
            snr_range: SNR范围
            
        Returns:
            完整测试结果
        """
        print("=" * 80)
        print("开始可变码率SNR扫描测试")
        print(f"Group范围: {group_range}")
        print(f"SNR范围: {snr_range} dB")
        print("=" * 80)
        
        results = {
            'group_range': group_range,
            'snr_range': snr_range,
            'test_results': [],  # 存储所有测试结果
            'ber_matrix': np.zeros((len(group_range), len(snr_range))),  # BER矩阵
            'ier_matrix': np.zeros((len(group_range), len(snr_range))),  # IER矩阵
            'code_rates': []  # 码率列表
        }
        
        total_tests = len(group_range) * len(snr_range)
        test_count = 0
        
        for i, group_to_use in enumerate(group_range):
            k = group_to_use * self.group_size * self.bits_per_index
            code_rate = k / self.n
            results['code_rates'].append(code_rate)
            
            print(f"\n>>> 测试Group {group_to_use} (k={k}, 码率={code_rate:.3f}) <<<")
            
            for j, snr_db in enumerate(snr_range):
                test_count += 1
                print(f"进度: {test_count}/{total_tests} - Group {group_to_use}, SNR {snr_db}dB", end=" ... ")
                
                try:
                    # 执行单一配置测试
                    result = self.test_single_config(group_to_use, snr_db)
                    results['test_results'].append(result)
                    
                    # 存储到矩阵中
                    results['ber_matrix'][i, j] = float(result['bit_error_rate'])
                    results['ier_matrix'][i, j] = float(result['index_error_rate'])
                    
                    print(f"BER: {result['bit_error_rate']:.2e}, IER: {result['index_error_rate']:.2e}")
                    
                except Exception as e:
                    print(f"失败: {e}")
                    results['ber_matrix'][i, j] = np.nan
                    results['ier_matrix'][i, j] = np.nan
                    
        return results
        
    def analyze_code_rate_performance(self, results: Dict):
        """分析不同码率下的性能"""
        print("\n" + "=" * 80)
        print("可变码率性能分析")
        print("=" * 80)
        
        group_range = results['group_range']
        snr_range = results['snr_range']
        ber_matrix = results['ber_matrix']
        ier_matrix = results['ier_matrix']
        code_rates = results['code_rates']
        
        # 1. 码率与性能汇总表
        print("\n=== 码率性能汇总 ===")
        print(f"{'Group':<6} {'k':<4} {'码率':<8} {'平均BER':<12} {'最佳BER':<12} {'平均IER':<12} {'最佳IER':<12}")
        print("-" * 75)
        
        for i, group_to_use in enumerate(group_range):
            k = group_to_use * 4 * 4
            code_rate = code_rates[i]
            
            # 计算该码率下的平均和最佳性能
            valid_ber = ber_matrix[i, ~np.isnan(ber_matrix[i, :])]
            valid_ier = ier_matrix[i, ~np.isnan(ier_matrix[i, :])]
            
            if len(valid_ber) > 0:
                avg_ber = np.mean(valid_ber)
                best_ber = np.min(valid_ber)
                avg_ier = np.mean(valid_ier)
                best_ier = np.min(valid_ier)
                
                print(f"{group_to_use:<6} {k:<4} {code_rate:<8.3f} {avg_ber:<12.2e} {best_ber:<12.2e} {avg_ier:<12.2e} {best_ier:<12.2e}")
            else:
                print(f"{group_to_use:<6} {k:<4} {code_rate:<8.3f} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
        
        # 2. 特定SNR点的码率比较
        print(f"\n=== 特定SNR点的码率性能比较 ===")
        mid_snr_idx = len(snr_range) // 2
        mid_snr = snr_range[mid_snr_idx]
        
        print(f"SNR = {mid_snr}dB 时的性能:")
        print(f"{'码率':<8} {'BER':<12} {'IER':<12} {'相对BER':<12}")
        print("-" * 50)
        
        baseline_ber = ber_matrix[0, mid_snr_idx]  # 以最低码率为基准
        for i, code_rate in enumerate(code_rates):
            ber = ber_matrix[i, mid_snr_idx]
            ier = ier_matrix[i, mid_snr_idx]
            relative_ber = ber / baseline_ber if not np.isnan(ber) and baseline_ber > 0 else np.nan
            
            if not np.isnan(ber):
                print(f"{code_rate:<8.3f} {ber:<12.2e} {ier:<12.2e} {relative_ber:<12.2f}")
            else:
                print(f"{code_rate:<8.3f} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
        
        # 3. 码率与容量分析
        print(f"\n=== 码率与理论容量分析 ===")
        print("理论容量计算 (4QAM, AWGN信道):")
        
        for snr_db in [0, 1, 2, 3, 4, 5]:
            if snr_db in snr_range:
                snr_linear = 10**(snr_db/10)
                # 4QAM的理论容量 (bits per channel use)
                capacity = 2 * np.log2(1 + snr_linear)  # 2 is for 4QAM
                print(f"  SNR {snr_db}dB: 理论容量 = {capacity:.3f} bits/symbol")
                
                # 找到可以工作的最大码率
                snr_idx = snr_range.index(snr_db)
                working_rates = []
                for i, code_rate in enumerate(code_rates):
                    if ber_matrix[i, snr_idx] < 1e-3:  # BER < 0.1% 认为可以工作
                        working_rates.append(code_rate)
                
                if working_rates:
                    max_working_rate = max(working_rates)
                    efficiency = max_working_rate / capacity
                    print(f"           最大可工作码率 = {max_working_rate:.3f}, 效率 = {efficiency:.1%}")
                else:
                    print(f"           无可工作的码率")
        
        # 4. 绘制性能图表
        self.plot_code_rate_results(results)
        
    def plot_code_rate_results(self, results: Dict):
        """绘制码率性能结果"""
        group_range = results['group_range']
        snr_range = results['snr_range']
        ber_matrix = results['ber_matrix']
        ier_matrix = results['ier_matrix']
        code_rates = results['code_rates']
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. BER vs SNR (不同码率)
        ax1 = plt.subplot(2, 3, 1)
        for i, (group, code_rate) in enumerate(zip(group_range, code_rates)):
            valid_mask = ~np.isnan(ber_matrix[i, :])
            if np.any(valid_mask):
                plt.semilogy(np.array(snr_range)[valid_mask], ber_matrix[i, valid_mask], 
                           marker='o', linewidth=2, label=f'Rate {code_rate:.3f} (Group {group})')
        
        plt.xlabel('SNR (dB)')
        plt.ylabel('Bit Error Rate (BER)')
        plt.title('BER vs SNR for Different Code Rates')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 2. IER vs SNR (不同码率)
        ax2 = plt.subplot(2, 3, 2)
        for i, (group, code_rate) in enumerate(zip(group_range, code_rates)):
            valid_mask = ~np.isnan(ier_matrix[i, :])
            if np.any(valid_mask):
                plt.semilogy(np.array(snr_range)[valid_mask], ier_matrix[i, valid_mask], 
                           marker='s', linewidth=2, label=f'Rate {code_rate:.3f} (Group {group})')
        
        plt.xlabel('SNR (dB)')
        plt.ylabel('Index Error Rate (IER)')
        plt.title('IER vs SNR for Different Code Rates')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 3. BER热力图
        ax3 = plt.subplot(2, 3, 3)
        im1 = plt.imshow(np.log10(ber_matrix + 1e-10), aspect='auto', cmap='viridis')
        plt.colorbar(im1, label='log10(BER)')
        plt.xlabel('SNR Index')
        plt.ylabel('Group Index')
        plt.title('BER Heatmap (log scale)')
        plt.xticks(range(len(snr_range)), [f'{s:.1f}' for s in snr_range])
        plt.yticks(range(len(group_range)), [f'{g}' for g in group_range])
        
        # 4. 码率 vs 性能 (固定SNR)
        ax4 = plt.subplot(2, 3, 4)
        mid_snr_idx = len(snr_range) // 2
        mid_snr = snr_range[mid_snr_idx]
        
        valid_mask = ~np.isnan(ber_matrix[:, mid_snr_idx])
        if np.any(valid_mask):
            plt.semilogy(np.array(code_rates)[valid_mask], ber_matrix[valid_mask, mid_snr_idx], 
                        'ro-', linewidth=2, markersize=8)
        
        plt.xlabel('Code Rate (k/n)')
        plt.ylabel('Bit Error Rate (BER)')
        plt.title(f'BER vs Code Rate (SNR = {mid_snr}dB)')
        plt.grid(True, alpha=0.3)
        
        # 5. 信息位数 vs 性能
        ax5 = plt.subplot(2, 3, 5)
        k_values = [g * 4 * 4 for g in group_range]
        
        valid_mask = ~np.isnan(ber_matrix[:, mid_snr_idx])
        if np.any(valid_mask):
            plt.semilogy(np.array(k_values)[valid_mask], ber_matrix[valid_mask, mid_snr_idx], 
                        'bo-', linewidth=2, markersize=8)
        
        plt.xlabel('Information Bits (k)')
        plt.ylabel('Bit Error Rate (BER)')
        plt.title(f'BER vs Information Bits (SNR = {mid_snr}dB)')
        plt.grid(True, alpha=0.3)
        
        # 6. 编码增益分析
        ax6 = plt.subplot(2, 3, 6)
        
        # 计算理论容量和实际码率的比较
        capacity_rates = []
        working_rates = []
        
        for snr_db in snr_range:
            snr_linear = 10**(snr_db/10)
            capacity = 2 * np.log2(1 + snr_linear)  # 4QAM capacity
            capacity_rates.append(capacity)
            
            # 找到该SNR下可工作的最大码率
            snr_idx = snr_range.index(snr_db)
            max_rate = 0
            for i, code_rate in enumerate(code_rates):
                if not np.isnan(ber_matrix[i, snr_idx]) and ber_matrix[i, snr_idx] < 1e-3:
                    max_rate = max(max_rate, code_rate)
            working_rates.append(max_rate)
        
        plt.plot(snr_range, capacity_rates, 'k--', linewidth=2, label='Theoretical Capacity')
        plt.plot(snr_range, working_rates, 'r-o', linewidth=2, label='Max Working Rate')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Rate (bits/symbol)')
        plt.title('Capacity vs Achievable Rate')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs('/home/zsy/workplace/DHSCC/results', exist_ok=True)
        plt.savefig('/home/zsy/workplace/DHSCC/results/variable_code_rate_results.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        print("\nCode rate analysis figure saved to: /home/zsy/workplace/DHSCC/results/variable_code_rate_results.png")
        
    def save_results(self, results: Dict):
        """保存测试结果"""
        os.makedirs('/home/zsy/workplace/DHSCC/results', exist_ok=True)
        
        results_file = "/home/zsy/workplace/DHSCC/results/variable_code_rate_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Variable code rate test results saved to: {results_file}")
        
        # 同时保存一个文本摘要
        summary_file = "/home/zsy/workplace/DHSCC/results/variable_code_rate_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Variable Code Rate Test Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Test Parameters:\n")
            f.write(f"- Group range: {results['group_range']}\n")
            f.write(f"- SNR range: {results['snr_range']} dB\n")
            f.write(f"- Code rates: {results['code_rates']}\n")
            f.write(f"- Batch size: {self.batch_size}\n")
            f.write(f"- Fixed code length: {self.n}\n\n")
            
            f.write("BER Matrix (rows: groups, cols: SNR):\n")
            f.write(str(results['ber_matrix']) + "\n\n")
            
            f.write("IER Matrix (rows: groups, cols: SNR):\n")
            f.write(str(results['ier_matrix']) + "\n")
            
        print(f"Summary saved to: {summary_file}")


def main():
    """主测试函数"""
    print("=" * 80)
    print("                可变码率数字传输系统测试")
    print("=" * 80)
    
    # 检查CUDA可用性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    if device == "cuda":
        print(f"GPU信息: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 创建测试器
    tester = VariableCodeRateTester(batch_size=1024, device=device)
    
    # 测试参数
    group_range = list(range(2, 9))  # [2, 3, 4, 5, 6, 7, 8]
    snr_range = np.linspace(0, 5, 11).tolist()  # 0-5dB, 11个点
    
    print(f"测试group_to_use范围: {group_range}")
    print(f"对应信息位数: {[g*4*4 for g in group_range]}")
    print(f"对应码率: {[g*4*4/256 for g in group_range]}")
    print(f"SNR测试点: {snr_range}")
    
    # 运行完整测试
    results = tester.run_group_snr_sweep(group_range, snr_range)
    
    # 分析结果
    tester.analyze_code_rate_performance(results)
    
    # 保存结果
    tester.save_results(results)
    
    print("\n" + "="*80)
    print("                可变码率测试完成!")
    print("="*80)


if __name__ == "__main__":
    main()
