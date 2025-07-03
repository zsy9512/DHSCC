#!/usr/bin/env python3
"""
Wireless Communication Digital Transmission Test
=================        # 模式4: 编码数字传输（极化码）
        self.interface_mode4 = WirelessInterface(
            comm_mode=4,
            k=128,  # 信息位数（等于总比特数）
            n=256,  # 码长
            channel_model='awgn',
            constellation_type='bpsk',  # 改为BPSK以降低SNR要求
            bits_per_symbol=1,
            decoder_type='SCL',  # 注意大写
            list_size=8,
            device=self.device
        )=================

测试数字传输的完整流程：
1. 索引转二进制比特
2. 模式3传输（无极化码）
3. 模式4传输（有极化码）
4. 误码率和误索引率分析

测试参数：
- 索引形状: [batch_size, 8, 4] (group_to_use=8)
- 索引范围: 0-15 (4 bits per index)
- 总比特数: 8 * 4 * 4 = 128 bits
- 极化码: k=128, n=256
- 调制方式: 4QAM (2 bits per symbol)
- SNR范围: 0-5 dB (6 steps)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
import sys
import os

# 添加路径以导入wireless_comm模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wireless_comm.interface import WirelessInterface, create_wireless_interface
from wireless_comm.system_integration import WirelessCommunicationSystem
from wireless_comm.config import get_config_by_scenario


class DigitalTransmissionTester:
    """数字传输测试器"""
    
    def __init__(self, batch_size: int = 4, device: str = "cuda"):
        """
        初始化测试器
        
        Args:
            batch_size: 批量大小
            device: 计算设备
        """
        self.batch_size = batch_size
        self.device = device
        self.group_to_use = 8
        self.group_size = 4
        self.bits_per_index = 4
        self.total_bits = self.group_to_use * self.group_size * self.bits_per_index  # 128 bits
        
        print(f"=== 数字传输测试器初始化 ===")
        print(f"批量大小: {batch_size}")
        print(f"设备: {device}")
        print(f"索引形状: [{batch_size}, {self.group_to_use}, {self.group_size}]")
        print(f"总比特数: {self.total_bits}")
        print()
        
        # 创建测试数据
        self.indices = self._generate_test_indices()
        
        # 创建无线接口
        self._setup_wireless_interfaces()
        
    def _generate_test_indices(self) -> torch.Tensor:
        """生成测试索引数据"""
        torch.manual_seed(42)  # 固定随机种子以便重现
        indices = torch.randint(
            0, 16,  # 索引范围 0-15
            (self.batch_size, self.group_to_use, self.group_size),
            device=self.device
        )
        print(f"生成测试索引: shape={indices.shape}, range=[{indices.min()}, {indices.max()}]")
        print(f"示例索引 (第一个batch):\n{indices[0]}")
        print()
        return indices
        
    def _setup_wireless_interfaces(self):
        """设置无线传输接口"""
        print("=== 设置无线传输接口 ===")
        
        # 模式3: 无编码数字传输
        self.interface_mode3 = WirelessInterface(
            comm_mode=3,
            channel_model='awgn',
            constellation_type='qam',  # 4QAM调制
            bits_per_symbol=2,
            device=self.device
        )
        print(f"模式3接口创建成功: 无编码数字传输, 4QAM调制")
        
        # 模式4: 编码数字传输（极化码）
        self.interface_mode4 = WirelessInterface(
            comm_mode=4,
            k=128,  # 信息位数（等于总比特数）
            n=256,  # 码长
            channel_model='awgn',
            constellation_type='qam',  # 4QAM调制
            bits_per_symbol=2,
            decoder_type='SC',
            list_size=8,
            device=self.device
        )
        print(f"模式4接口创建成功: 极化码 k={self.interface_mode4.k}, n={self.interface_mode4.n}, 4QAM调制")
        print()
        
    def test_indices_binary_conversion(self):
        """测试索引与二进制转换"""
        print("=== 测试索引与二进制转换 ===")
        
        # 索引转二进制
        binary_data = self.interface_mode3.indices_to_binary(
            self.indices, 
            bits_per_index=self.bits_per_index
        )
        
        print(f"输入索引: shape={self.indices.shape}")
        print(f"输出二进制: shape={binary_data.shape}, dtype={binary_data.dtype}")
        print(f"预期比特数: {self.total_bits}, 实际比特数: {binary_data.shape[-1]}")
        
        # 验证转换正确性（检查第一个样本的第一个索引）
        first_index = self.indices[0, 0, 0].item()
        first_binary = binary_data[0, :4]  # 前4位
        reconstructed_index = int(''.join(map(str, first_binary.int().tolist())), 2)
        print(f"验证转换: 索引 {first_index} -> 二进制 {first_binary.int().tolist()} -> 重建索引 {reconstructed_index}")
        
        # 二进制转回索引
        reconstructed_indices = self.interface_mode3.binary_to_indices(
            binary_data, 
            self.indices.shape
        )
        
        print(f"重建索引: shape={reconstructed_indices.shape}")
        
        # 验证转换的双向性
        conversion_error = (self.indices != reconstructed_indices).float().mean()
        print(f"转换误差率: {conversion_error:.6f}")
        
        if conversion_error == 0:
            print("✓ 索引-二进制转换测试通过")
        else:
            print("✗ 索引-二进制转换测试失败")
            
        print()
        return binary_data, reconstructed_indices
        
    def test_mode3_transmission(self, snr_db: float) -> Dict:
        """
        测试模式3传输（无编码）
        
        Args:
            snr_db: 信噪比
            
        Returns:
            测试结果字典
        """
        print(f"--- 测试模式3传输 (SNR={snr_db}dB) ---")
        
        # 1. 索引转二进制
        binary_input = self.interface_mode3.indices_to_binary(
            self.indices, 
            bits_per_index=self.bits_per_index
        )
        print(f"1. 索引转二进制: {self.indices.shape} -> {binary_input.shape}")
        
        # 2. 数字传输
        start_time = time.time()
        result = self.interface_mode3.transmit_mode3(
            indices=self.indices,
            snr_db=snr_db,
            bits_per_index=self.bits_per_index,
            group_to_use=self.group_to_use
        )
        transmission_time = time.time() - start_time
        
        print(f"2. 数字传输完成: 耗时 {transmission_time:.4f}s")
        print(f"   - BER: {result['bit_error_rate']:.6f}")
        print(f"   - 索引误差率: {result['index_error_rate']:.6f}")
        
        # 3. 获取中间结果进行详细分析
        received_indices = result['received_indices']
        
        print(f"3. 接收结果: shape={received_indices.shape}")
        print(f"   - 原始索引范围: [{self.indices.min()}, {self.indices.max()}]")
        print(f"   - 接收索引范围: [{received_indices.min()}, {received_indices.max()}]")
        
        # 4. 计算详细误差统计
        index_errors = (self.indices != received_indices).float()
        total_indices = self.indices.numel()
        error_indices = index_errors.sum().item()
        
        print(f"4. 误差统计:")
        print(f"   - 总索引数: {total_indices}")
        print(f"   - 错误索引数: {error_indices}")
        print(f"   - 索引误差率: {error_indices/total_indices:.6f}")
        
        result.update({
            'transmission_time': transmission_time,
            'total_indices': total_indices,
            'error_indices': error_indices,
            'original_indices': self.indices.clone(),
            'received_indices': received_indices
        })
        
        print()
        return result
        
    def test_mode4_transmission(self, snr_db: float) -> Dict:
        """
        测试模式4传输（极化码编码）
        
        Args:
            snr_db: 信噪比
            
        Returns:
            测试结果字典
        """
        print(f"--- 测试模式4传输 (SNR={snr_db}dB) ---")
        
        # 1. 索引转二进制
        binary_input = self.interface_mode4.indices_to_binary(
            self.indices, 
            bits_per_index=self.bits_per_index
        )
        print(f"1. 索引转二进制: {self.indices.shape} -> {binary_input.shape}")
        
        # 2. 极化编码传输
        start_time = time.time()
        result = self.interface_mode4.transmit_mode4(
            indices=self.indices,
            snr_db=snr_db,
            bits_per_index=self.bits_per_index,
            group_to_use=self.group_to_use
        )
        transmission_time = time.time() - start_time
        
        print(f"2. 极化编码传输完成: 耗时 {transmission_time:.4f}s")
        print(f"   - BER: {result['bit_error_rate']:.6f}")
        print(f"   - 索引误差率: {result['index_error_rate']:.6f}")
        
        # 3. 获取中间结果
        received_indices = result['received_indices']
        
        print(f"3. 接收结果: shape={received_indices.shape}")
        print(f"   - 原始索引范围: [{self.indices.min()}, {self.indices.max()}]")
        print(f"   - 接收索引范围: [{received_indices.min()}, {received_indices.max()}]")
        
        # 4. 极化码编码增益分析
        index_errors = (self.indices != received_indices).float()
        total_indices = self.indices.numel()
        error_indices = index_errors.sum().item()
        
        print(f"4. 极化码性能:")
        print(f"   - 码率: {self.interface_mode4.k}/{self.interface_mode4.n} = {self.interface_mode4.k/self.interface_mode4.n:.3f}")
        print(f"   - 总索引数: {total_indices}")
        print(f"   - 错误索引数: {error_indices}")
        print(f"   - 索引误差率: {error_indices/total_indices:.6f}")
        
        result.update({
            'transmission_time': transmission_time,
            'total_indices': total_indices,
            'error_indices': error_indices,
            'original_indices': self.indices.clone(),
            'received_indices': received_indices,
            'code_rate': self.interface_mode4.k / self.interface_mode4.n
        })
        
        print()
        return result
        
    def run_snr_sweep_test(self, snr_range: List[float]) -> Dict:
        """
        运行SNR扫描测试
        
        Args:
            snr_range: SNR范围列表
            
        Returns:
            完整测试结果
        """
        print("=" * 60)
        print("开始SNR扫描测试")
        print(f"SNR范围: {snr_range} dB")
        print("=" * 60)
        
        results = {
            'snr_range': snr_range,
            'mode3_results': [],
            'mode4_results': [],
            'mode3_bers': [],
            'mode4_bers': [],
            'mode3_index_errors': [],
            'mode4_index_errors': []
        }
        
        for i, snr_db in enumerate(snr_range):
            print(f"\n>>> 测试进度: {i+1}/{len(snr_range)} (SNR={snr_db}dB) <<<")
            
            # 测试模式3
            mode3_result = self.test_mode3_transmission(snr_db)
            results['mode3_results'].append(mode3_result)
            results['mode3_bers'].append(mode3_result['bit_error_rate'])
            results['mode3_index_errors'].append(mode3_result['index_error_rate'])
            
            # 测试模式4
            mode4_result = self.test_mode4_transmission(snr_db)
            results['mode4_results'].append(mode4_result)
            results['mode4_bers'].append(mode4_result['bit_error_rate'])
            results['mode4_index_errors'].append(mode4_result['index_error_rate'])
            
            # 当前SNR点的对比
            ber_improvement = mode3_result['bit_error_rate'] / max(mode4_result['bit_error_rate'], 1e-10)
            index_improvement = mode3_result['index_error_rate'] / max(mode4_result['index_error_rate'], 1e-10)
            
            print(f">>> SNR={snr_db}dB 性能对比:")
            print(f"    BER改善: {ber_improvement:.2f}x")
            print(f"    索引误差率改善: {index_improvement:.2f}x")
            print(f"    模式3 BER: {mode3_result['bit_error_rate']:.2e}")
            print(f"    模式4 BER: {mode4_result['bit_error_rate']:.2e}")
            
        return results
        
    def analyze_results(self, results: Dict):
        """分析和可视化测试结果"""
        print("\n" + "=" * 60)
        print("测试结果分析")
        print("=" * 60)
        
        snr_range = results['snr_range']
        mode3_bers = results['mode3_bers']
        mode4_bers = results['mode4_bers']
        mode3_index_errors = results['mode3_index_errors']
        mode4_index_errors = results['mode4_index_errors']
        
        # 1. 性能汇总表
        print("\n=== 性能汇总表 ===")
        print(f"{'SNR(dB)':<8} {'Mode3 BER':<12} {'Mode4 BER':<12} {'BER改善':<10} {'Mode3 IER':<12} {'Mode4 IER':<12} {'IER改善':<10}")
        print("-" * 80)
        
        for i, snr in enumerate(snr_range):
            ber_improve = float(mode3_bers[i]) / max(float(mode4_bers[i]), 1e-10)
            ier_improve = float(mode3_index_errors[i]) / max(float(mode4_index_errors[i]), 1e-10)
            
            print(f"{snr:<8.1f} {float(mode3_bers[i]):<12.2e} {float(mode4_bers[i]):<12.2e} {ber_improve:<10.2f} "
                  f"{float(mode3_index_errors[i]):<12.2e} {float(mode4_index_errors[i]):<12.2e} {ier_improve:<10.2f}")
        
        # 2. 计算平均改善
        avg_ber_improvement = np.mean([float(mode3_bers[i]) / max(float(mode4_bers[i]), 1e-10) for i in range(len(snr_range))])
        avg_ier_improvement = np.mean([float(mode3_index_errors[i]) / max(float(mode4_index_errors[i]), 1e-10) for i in range(len(snr_range))])
        
        print(f"\n=== 平均性能改善 ===")
        print(f"平均BER改善: {avg_ber_improvement:.2f}x")
        print(f"平均索引误差率改善: {avg_ier_improvement:.2f}x")
        
        # 3. 极化码编码增益
        print(f"\n=== 极化码编码分析 ===")
        print(f"码率: {results['mode4_results'][0]['code_rate']:.3f}")
        print(f"信息位: {self.interface_mode4.k}")
        print(f"码长: {self.interface_mode4.n}")
        print(f"编码开销: {(self.interface_mode4.n - self.interface_mode4.k) / self.interface_mode4.k * 100:.1f}%")
        
        # 4. 绘制性能曲线
        self.plot_performance_curves(results)
        
    def plot_performance_curves(self, results: Dict):
        """绘制性能曲线"""
        snr_range = results['snr_range']
        mode3_bers = [float(x) for x in results['mode3_bers']]
        mode4_bers = [float(x) for x in results['mode4_bers']]
        mode3_index_errors = [float(x) for x in results['mode3_index_errors']]
        mode4_index_errors = [float(x) for x in results['mode4_index_errors']]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. BER性能对比
        ax1.semilogy(snr_range, mode3_bers, 'b-o', label='Mode 3 (Uncoded)', linewidth=2, markersize=6)
        ax1.semilogy(snr_range, mode4_bers, 'r-s', label='Mode 4 (Polar Code)', linewidth=2, markersize=6)
        ax1.set_xlabel('SNR (dB)')
        ax1.set_ylabel('Bit Error Rate (BER)')
        ax1.set_title('BER Performance Comparison')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 索引误差率对比
        ax2.semilogy(snr_range, mode3_index_errors, 'b-o', label='Mode 3 (Uncoded)', linewidth=2, markersize=6)
        ax2.semilogy(snr_range, mode4_index_errors, 'r-s', label='Mode 4 (Polar Code)', linewidth=2, markersize=6)
        ax2.set_xlabel('SNR (dB)')
        ax2.set_ylabel('Index Error Rate (IER)')
        ax2.set_title('Index Error Rate Comparison')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. BER改善倍数
        ber_improvements = [float(mode3_bers[i]) / max(float(mode4_bers[i]), 1e-10) for i in range(len(snr_range))]
        ax3.plot(snr_range, ber_improvements, 'g-^', linewidth=2, markersize=6)
        ax3.set_xlabel('SNR (dB)')
        ax3.set_ylabel('BER Improvement Factor')
        ax3.set_title('Polar Code BER Improvement')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # 4. 传输时延对比
        mode3_times = [float(r['transmission_time']) for r in results['mode3_results']]
        mode4_times = [float(r['transmission_time']) for r in results['mode4_results']]
        
        x_pos = np.arange(len(snr_range))
        width = 0.35
        
        ax4.bar(x_pos - width/2, mode3_times, width, label='Mode 3 (Uncoded)', alpha=0.7)
        ax4.bar(x_pos + width/2, mode4_times, width, label='Mode 4 (Polar Code)', alpha=0.7)
        ax4.set_xlabel('SNR (dB)')
        ax4.set_ylabel('Transmission Time (s)')
        ax4.set_title('Transmission Time Comparison')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'{snr:.1f}' for snr in snr_range])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 确保results文件夹存在
        import os
        os.makedirs('/home/zsy/workplace/DHSCC/results', exist_ok=True)
        
        plt.savefig('/home/zsy/workplace/DHSCC/results/test_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\nFigure saved to: /home/zsy/workplace/DHSCC/results/test_results.png")


def main():
    """主测试函数"""
    print("=" * 80)
    print("                  数字传输系统完整测试")
    print("=" * 80)
    
    # 检查CUDA可用性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    if device == "cuda":
        print(f"GPU信息: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 创建测试器
    tester = DigitalTransmissionTester(batch_size=4, device=device)
    
    # 1. 基础转换测试
    print("\n" + "="*50)
    print("第一步: 测试索引-二进制转换")
    print("="*50)
    binary_data, reconstructed_indices = tester.test_indices_binary_conversion()
    
    # 2. SNR扫描测试
    print("="*50)
    print("第二步: SNR扫描测试")
    print("="*50)
    
    # SNR范围: 0-5dB, 6个点 (适合4QAM测试)
    snr_range = np.linspace(0, 5, 6).tolist()
    print(f"SNR测试点: {snr_range}")
    
    # 运行完整测试
    results = tester.run_snr_sweep_test(snr_range)
    
    # 3. 结果分析
    print("="*50)
    print("第三步: 结果分析与可视化")
    print("="*50)
    tester.analyze_results(results)
    
    print("\n" + "="*80)
    print("                        测试完成!")
    print("="*80)
    
    # 保存结果
    import os
    os.makedirs('/home/zsy/workplace/DHSCC/results', exist_ok=True)
    
    results_file = "/home/zsy/workplace/DHSCC/results/test_results.pkl"
    import pickle
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Detailed results saved to: {results_file}")


if __name__ == "__main__":
    main()
