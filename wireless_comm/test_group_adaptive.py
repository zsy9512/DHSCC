#!/usr/bin/env python3
"""
Group-Adaptive Digital Transmission Test
========================================

测试不同group_to_use下的数字传输系统：
1. 测试group_to_use = 2,3,4,5,6,7,8的情况
2. 检查数据转换的正确性
3. 验证不同码长对应的编码系统自动切换
4. 分析调制映射和传输性能

测试参数：
- 索引形状: [batch_size, group_to_use, 4] 
- group_to_use范围: 2-8
- 索引范围: 0-15 (4 bits per index)
- 极化码适应性: k = group_to_use * 4 * 4, n=256
- 调制方式: 4QAM (2 bits per symbol)
- 固定SNR: 3dB (中等信噪比测试)
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

from wireless_comm.interface import WirelessInterface
from wireless_comm.system_integration import WirelessCommunicationSystem


class GroupAdaptiveTransmissionTester:
    """Group自适应数字传输测试器"""
    
    def __init__(self, batch_size: int = 4, device: str = "cuda", test_snr: float = 3.0):
        """
        初始化测试器
        
        Args:
            batch_size: 批量大小
            device: 计算设备
            test_snr: 测试信噪比
        """
        self.batch_size = batch_size
        self.device = device
        self.test_snr = test_snr
        self.group_size = 4  # 每组的大小固定为4
        self.bits_per_index = 4  # 每个索引4比特
        
        print(f"=== Group自适应数字传输测试器初始化 ===")
        print(f"批量大小: {batch_size}")
        print(f"设备: {device}")
        print(f"测试SNR: {test_snr} dB")
        print(f"索引形状: [{batch_size}, group_to_use, {self.group_size}]")
        print()
        
        # 将在测试中根据group_to_use动态创建接口
        
    def _generate_test_indices(self, group_to_use: int) -> torch.Tensor:
        """生成指定group_to_use的测试索引数据"""
        torch.manual_seed(42)  # 固定随机种子以便重现
        indices = torch.randint(
            0, 16,  # 索引范围 0-15
            (self.batch_size, group_to_use, self.group_size),
            device=self.device
        )
        return indices
        
    def _setup_wireless_interfaces(self, group_to_use: int) -> Tuple[WirelessInterface, WirelessInterface]:
        """为指定的group_to_use设置无线传输接口"""
        total_bits = group_to_use * self.group_size * self.bits_per_index
        
        print(f"--- 设置group_to_use={group_to_use}的接口 ---")
        print(f"总比特数: {total_bits}")
        
        # 模式3: 无编码数字传输
        interface_mode3 = WirelessInterface(
            comm_mode=3,
            channel_model='awgn',
            constellation_type='qam',
            bits_per_symbol=2,
            device=self.device,
            verbose=True  # 启用详细输出以观察系统行为
        )
        
        # 模式4: 编码数字传输（极化码）
        # 关键：k应该等于total_bits，以便容纳所有信息
        interface_mode4 = WirelessInterface(
            comm_mode=4,
            k=total_bits,  # 信息位数等于总比特数
            n=256,  # 码长固定
            channel_model='awgn',
            constellation_type='qam',
            bits_per_symbol=2,
            decoder_type='SC',
            list_size=8,
            device=self.device,
            verbose=True
        )
        
        code_rate = total_bits / 256
        print(f"模式3接口: 无编码数字传输, 4QAM调制")
        print(f"模式4接口: 极化码 k={total_bits}, n=256, 码率={code_rate:.3f}")
        print()
        
        return interface_mode3, interface_mode4
        
    def test_data_conversion(self, group_to_use: int) -> Dict:
        """测试数据转换的正确性"""
        print(f"=== 测试group_to_use={group_to_use}的数据转换 ===")
        
        # 生成测试数据
        indices = self._generate_test_indices(group_to_use)
        interface_mode3, interface_mode4 = self._setup_wireless_interfaces(group_to_use)
        
        total_bits = group_to_use * self.group_size * self.bits_per_index
        
        print(f"原始索引: shape={indices.shape}")
        print(f"示例索引 (第一个batch):\n{indices[0]}")
        
        # 测试模式3的数据转换
        binary_data_mode3 = interface_mode3.indices_to_binary(
            indices, 
            bits_per_index=self.bits_per_index
        )
        
        # 测试模式4的数据转换
        binary_data_mode4 = interface_mode4.indices_to_binary(
            indices, 
            bits_per_index=self.bits_per_index
        )
        
        print(f"模式3二进制: shape={binary_data_mode3.shape}")
        print(f"模式4二进制: shape={binary_data_mode4.shape}")
        print(f"预期比特数: {total_bits}")
        
        # 验证两种模式的转换结果是否一致
        conversion_consistent = torch.equal(binary_data_mode3, binary_data_mode4)
        print(f"两种模式转换一致性: {conversion_consistent}")
        
        # 测试逆向转换
        reconstructed_indices_mode3 = interface_mode3.binary_to_indices(
            binary_data_mode3, indices.shape
        )
        
        reconstructed_indices_mode4 = interface_mode4.binary_to_indices(
            binary_data_mode4, indices.shape
        )
        
        # 验证转换的双向性
        mode3_conversion_error = (indices != reconstructed_indices_mode3).float().mean()
        mode4_conversion_error = (indices != reconstructed_indices_mode4).float().mean()
        
        print(f"模式3转换误差率: {mode3_conversion_error:.6f}")
        print(f"模式4转换误差率: {mode4_conversion_error:.6f}")
        
        conversion_success = (mode3_conversion_error == 0) and (mode4_conversion_error == 0)
        
        if conversion_success:
            print("✓ 数据转换测试通过")
        else:
            print("✗ 数据转换测试失败")
        
        print()
        
        return {
            'group_to_use': group_to_use,
            'total_bits': total_bits,
            'indices_shape': indices.shape,
            'binary_shape': binary_data_mode3.shape,
            'conversion_consistent': conversion_consistent,
            'mode3_conversion_error': mode3_conversion_error.item(),
            'mode4_conversion_error': mode4_conversion_error.item(),
            'conversion_success': conversion_success,
            'interface_mode3': interface_mode3,
            'interface_mode4': interface_mode4,
            'test_indices': indices
        }
        
    def test_transmission_performance(self, group_to_use: int, conversion_result: Dict) -> Dict:
        """测试传输性能"""
        print(f"=== 测试group_to_use={group_to_use}的传输性能 ===")
        
        indices = conversion_result['test_indices']
        interface_mode3 = conversion_result['interface_mode3']
        interface_mode4 = conversion_result['interface_mode4']
        
        # 测试模式3传输
        print(f"--- 模式3传输 (SNR={self.test_snr}dB) ---")
        start_time = time.time()
        result_mode3 = interface_mode3.transmit_mode3(
            indices=indices,
            snr_db=self.test_snr,
            bits_per_index=self.bits_per_index,
            group_to_use=group_to_use
        )
        mode3_time = time.time() - start_time
        
        print(f"模式3传输完成: 耗时 {mode3_time:.4f}s")
        print(f"BER: {result_mode3['bit_error_rate']:.6f}")
        print(f"索引误差率: {result_mode3['index_error_rate']:.6f}")
        
        # 测试模式4传输
        print(f"--- 模式4传输 (SNR={self.test_snr}dB) ---")
        start_time = time.time()
        result_mode4 = interface_mode4.transmit_mode4(
            indices=indices,
            snr_db=self.test_snr,
            bits_per_index=self.bits_per_index,
            group_to_use=group_to_use
        )
        mode4_time = time.time() - start_time
        
        print(f"模式4传输完成: 耗时 {mode4_time:.4f}s")
        print(f"BER: {result_mode4['bit_error_rate']:.6f}")
        print(f"索引误差率: {result_mode4['index_error_rate']:.6f}")
        
        # 性能对比
        ber_improvement = result_mode3['bit_error_rate'] / max(result_mode4['bit_error_rate'], 1e-10)
        index_improvement = result_mode3['index_error_rate'] / max(result_mode4['index_error_rate'], 1e-10)
        
        print(f"性能改善:")
        print(f"  BER改善: {ber_improvement:.2f}x")
        print(f"  索引误差率改善: {index_improvement:.2f}x")
        print(f"  传输时间比: {mode4_time/mode3_time:.2f}x")
        
        # 检查接收数据的完整性
        mode3_received = result_mode3['received_indices']
        mode4_received = result_mode4['received_indices']
        
        print(f"接收数据检查:")
        print(f"  原始索引范围: [{indices.min()}, {indices.max()}]")
        print(f"  模式3接收范围: [{mode3_received.min()}, {mode4_received.max()}]")
        print(f"  模式4接收范围: [{mode4_received.min()}, {mode4_received.max()}]")
        
        print()
        
        return {
            'group_to_use': group_to_use,
            'mode3_result': result_mode3,
            'mode4_result': result_mode4,
            'mode3_time': mode3_time,
            'mode4_time': mode4_time,
            'ber_improvement': ber_improvement,
            'index_improvement': index_improvement,
            'transmission_success': True
        }
        
    def run_comprehensive_group_test(self, group_range: List[int]) -> Dict:
        """运行综合的group测试"""
        print("=" * 80)
        print(f"开始综合Group测试 (group_to_use范围: {group_range})")
        print("=" * 80)
        
        results = {
            'group_range': group_range,
            'conversion_results': [],
            'transmission_results': [],
            'summary': {}
        }
        
        for i, group_to_use in enumerate(group_range):
            print(f"\n>>> 测试进度: {i+1}/{len(group_range)} (group_to_use={group_to_use}) <<<")
            
            try:
                # 1. 数据转换测试
                conversion_result = self.test_data_conversion(group_to_use)
                results['conversion_results'].append(conversion_result)
                
                # 2. 传输性能测试
                if conversion_result['conversion_success']:
                    transmission_result = self.test_transmission_performance(group_to_use, conversion_result)
                    results['transmission_results'].append(transmission_result)
                else:
                    print(f"跳过group_to_use={group_to_use}的传输测试（数据转换失败）")
                    results['transmission_results'].append({
                        'group_to_use': group_to_use,
                        'transmission_success': False,
                        'error': 'Data conversion failed'
                    })
                    
            except Exception as e:
                print(f"group_to_use={group_to_use}测试失败: {e}")
                results['transmission_results'].append({
                    'group_to_use': group_to_use,
                    'transmission_success': False,
                    'error': str(e)
                })
                
        return results
        
    def analyze_group_results(self, results: Dict):
        """分析group测试结果"""
        print("\n" + "=" * 80)
        print("Group测试结果分析")
        print("=" * 80)
        
        group_range = results['group_range']
        conversion_results = results['conversion_results']
        transmission_results = results['transmission_results']
        
        # 1. 数据转换成功率统计
        print("\n=== 数据转换分析 ===")
        print(f"{'Group':<6} {'Total Bits':<12} {'Code Rate':<12} {'Conv Success':<12} {'Mode3 Error':<12} {'Mode4 Error':<12}")
        print("-" * 75)
        
        successful_conversions = 0
        for result in conversion_results:
            group = result['group_to_use']
            total_bits = result['total_bits']
            code_rate = total_bits / 256
            conv_success = "✓" if result['conversion_success'] else "✗"
            mode3_error = result['mode3_conversion_error']
            mode4_error = result['mode4_conversion_error']
            
            print(f"{group:<6} {total_bits:<12} {code_rate:<12.3f} {conv_success:<12} {mode3_error:<12.6f} {mode4_error:<12.6f}")
            
            if result['conversion_success']:
                successful_conversions += 1
                
        print(f"\n数据转换成功率: {successful_conversions}/{len(conversion_results)} ({successful_conversions/len(conversion_results)*100:.1f}%)")
        
        # 2. 传输性能分析
        print("\n=== 传输性能分析 ===")
        print(f"{'Group':<6} {'Mode3 BER':<12} {'Mode4 BER':<12} {'BER Improve':<12} {'Mode3 IER':<12} {'Mode4 IER':<12} {'IER Improve':<12}")
        print("-" * 85)
        
        successful_transmissions = 0
        ber_improvements = []
        ier_improvements = []
        
        for result in transmission_results:
            if result.get('transmission_success', False):
                group = result['group_to_use']
                mode3_ber = float(result['mode3_result']['bit_error_rate'])
                mode4_ber = float(result['mode4_result']['bit_error_rate'])
                ber_improve = result['ber_improvement']
                mode3_ier = float(result['mode3_result']['index_error_rate'])
                mode4_ier = float(result['mode4_result']['index_error_rate'])
                ier_improve = result['index_improvement']
                
                print(f"{group:<6} {mode3_ber:<12.2e} {mode4_ber:<12.2e} {ber_improve:<12.2f} {mode3_ier:<12.2e} {mode4_ier:<12.2e} {ier_improve:<12.2f}")
                
                successful_transmissions += 1
                ber_improvements.append(ber_improve)
                ier_improvements.append(ier_improve)
            else:
                group = result['group_to_use']
                error = result.get('error', 'Unknown error')
                print(f"{group:<6} {'FAILED':<12} {'FAILED':<12} {'N/A':<12} {'FAILED':<12} {'FAILED':<12} {'N/A':<12}")
                
        print(f"\n传输成功率: {successful_transmissions}/{len(transmission_results)} ({successful_transmissions/len(transmission_results)*100:.1f}%)")
        
        if ber_improvements:
            # Convert cuda tensors to cpu before numpy operations
            ber_improvements_cpu = [float(x) if isinstance(x, torch.Tensor) else x for x in ber_improvements]
            ier_improvements_cpu = [float(x) if isinstance(x, torch.Tensor) else x for x in ier_improvements]
            
            avg_ber_improvement = np.mean(ber_improvements_cpu)
            avg_ier_improvement = np.mean(ier_improvements_cpu)
            print(f"平均BER改善: {avg_ber_improvement:.2f}x")
            print(f"平均索引误差率改善: {avg_ier_improvement:.2f}x")
        
        # 3. 码率影响分析
        print("\n=== 码率影响分析 ===")
        print("不同group_to_use对应的码率和性能:")
        
        for i, result in enumerate(transmission_results):
            if result.get('transmission_success', False):
                group = result['group_to_use']
                total_bits = group * 4 * 4
                code_rate = total_bits / 256
                mode4_ber = float(result['mode4_result']['bit_error_rate'])
                
                print(f"Group {group}: 码率 {code_rate:.3f}, BER {mode4_ber:.2e}")
                
        # 4. 绘制结果
        self.plot_group_results(results)
        
    def plot_group_results(self, results: Dict):
        """绘制group测试结果"""
        successful_results = [r for r in results['transmission_results'] if r.get('transmission_success', False)]
        
        if not successful_results:
            print("没有成功的传输结果可绘制")
            return
            
        groups = [r['group_to_use'] for r in successful_results]
        mode3_bers = [float(r['mode3_result']['bit_error_rate']) for r in successful_results]
        mode4_bers = [float(r['mode4_result']['bit_error_rate']) for r in successful_results]
        mode3_iers = [float(r['mode3_result']['index_error_rate']) for r in successful_results]
        mode4_iers = [float(r['mode4_result']['index_error_rate']) for r in successful_results]
        ber_improvements = [float(r['ber_improvement']) for r in successful_results]
        code_rates = [g * 4 * 4 / 256 for g in groups]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. BER vs Group Size
        ax1.semilogy(groups, mode3_bers, 'b-o', label='Mode 3 (Uncoded)', linewidth=2, markersize=6)
        ax1.semilogy(groups, mode4_bers, 'r-s', label='Mode 4 (Polar Code)', linewidth=2, markersize=6)
        ax1.set_xlabel('Group to Use')
        ax1.set_ylabel('Bit Error Rate (BER)')
        ax1.set_title(f'BER vs Group Size (SNR={self.test_snr}dB)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xticks(groups)
        
        # 2. Index Error Rate vs Group Size
        ax2.semilogy(groups, mode3_iers, 'b-o', label='Mode 3 (Uncoded)', linewidth=2, markersize=6)
        ax2.semilogy(groups, mode4_iers, 'r-s', label='Mode 4 (Polar Code)', linewidth=2, markersize=6)
        ax2.set_xlabel('Group to Use')
        ax2.set_ylabel('Index Error Rate (IER)')
        ax2.set_title(f'Index Error Rate vs Group Size (SNR={self.test_snr}dB)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xticks(groups)
        
        # 3. BER Improvement vs Code Rate
        ax3.plot(code_rates, ber_improvements, 'g-^', linewidth=2, markersize=6)
        ax3.set_xlabel('Code Rate (k/n)')
        ax3.set_ylabel('BER Improvement Factor')
        ax3.set_title('Polar Code Gain vs Code Rate')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # 4. Total Bits vs Performance
        total_bits = [g * 4 * 4 for g in groups]
        ax4.bar([str(g) for g in groups], total_bits, alpha=0.7, color='skyblue')
        ax4.set_xlabel('Group to Use')
        ax4.set_ylabel('Total Information Bits')
        ax4.set_title('Information Bits vs Group Size')
        ax4.grid(True, alpha=0.3)
        
        # 添加第二个y轴显示码率
        ax4_twin = ax4.twinx()
        ax4_twin.plot(groups, code_rates, 'ro-', linewidth=2, markersize=6)
        ax4_twin.set_ylabel('Code Rate', color='red')
        ax4_twin.tick_params(axis='y', labelcolor='red')
        
        plt.tight_layout()
        
        # 确保results文件夹存在
        os.makedirs('/home/zsy/workplace/DHSCC/results', exist_ok=True)
        
        plt.savefig('/home/zsy/workplace/DHSCC/results/group_adaptive_test_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\nGroup test figure saved to: /home/zsy/workplace/DHSCC/results/group_adaptive_test_results.png")


def main():
    """主测试函数"""
    print("=" * 80)
    print("                Group自适应数字传输系统测试")
    print("=" * 80)
    
    # 检查CUDA可用性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    if device == "cuda":
        print(f"GPU信息: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 创建测试器
    test_snr = 3.0  # 中等信噪比
    tester = GroupAdaptiveTransmissionTester(batch_size=4, device=device, test_snr=test_snr)
    
    # 测试group_to_use范围: 2-8
    group_range = list(range(2, 9))  # [2, 3, 4, 5, 6, 7, 8]
    print(f"测试group_to_use范围: {group_range}")
    
    # 运行综合测试
    results = tester.run_comprehensive_group_test(group_range)
    
    # 分析结果
    tester.analyze_group_results(results)
    
    print("\n" + "="*80)
    print("                    Group自适应测试完成!")
    print("="*80)
    
    # 保存结果
    os.makedirs('/home/zsy/workplace/DHSCC/results', exist_ok=True)
    
    results_file = "/home/zsy/workplace/DHSCC/results/group_adaptive_test_results.pkl"
    import pickle
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Group adaptive test results saved to: {results_file}")


if __name__ == "__main__":
    main()
