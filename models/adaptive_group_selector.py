"""
自适应组选择器模块

该模块实现了基于性能反馈的自适应组选择算法，用于根据信道条件动态选择最优的group_to_use。
算法的核心思想是在保证分类精度的前提下，尽可能使用更多的组来充分利用带宽。
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

class AdaptiveGroupSelector:
    """
    启发式算法用于根据信道条件自适应选择最优的group_to_use
    
    核心思想：
    1. 维护每个SNR级别的性能历史记录  
    2. 基于性能反馈动态调整group_to_use
    3. 保证单调性约束：SNR越高，group_to_use越大
    4. 在精度不降低超过0.5%的情况下，倾向于选择更大的group_to_use
    """
    
    def __init__(self, snr_min: float = -2.0, snr_max: float = 8.0, snr_bins: int = 21):
        """
        初始化自适应组选择器
        
        Args:
            snr_min: SNR最小值
            snr_max: SNR最大值  
            snr_bins: SNR离散化的bin数量
        """
        # 将SNR范围离散化为bins
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.snr_bins = snr_bins
        self.snr_step = (snr_max - snr_min) / (snr_bins - 1)
        
        # 为每个SNR bin维护group选择策略
        self.group_choices = [2] * snr_bins  # 初始化为最小值2
        
        # 性能历史记录：每个(snr_bin, group)组合的准确率历史
        self.accuracy_history: Dict[Tuple[int, int], List[float]] = {}
        self.update_count: Dict[Tuple[int, int], int] = {}
        
        # 算法参数
        self.accuracy_threshold = 0.005  # 1%的精度阈值，放宽一些
        self.min_samples = 8  # 减少最少样本数，加快学习速度
        self.exploration_prob = 0.3  # 探索概率
        
        # 统计信息
        self.total_updates = 0
        self.strategy_changes = 0
        
        # 推理模式控制 - 用于Final Training阶段
        self.inference_mode = False
    
    def snr_to_bin(self, snr_value: float) -> int:
        """将SNR值映射到对应的bin索引"""
        if isinstance(snr_value, torch.Tensor):
            snr_value = snr_value.item()
        
        bin_idx = int((snr_value - self.snr_min) / self.snr_step)
        return max(0, min(bin_idx, self.snr_bins - 1))
    
    def get_group_to_use(self, snr_value: float, is_training: bool = True) -> int:
        """
        根据SNR值获取最优的group_to_use
        
        Args:
            snr_value: 当前SNR值
            is_training: 是否在训练模式
            
        Returns:
            int: 推荐的group_to_use值
        """
        bin_idx = self.snr_to_bin(snr_value)
        
        # 推理模式：无论训练还是测试，都直接返回学习到的策略，不再探索
        if self.inference_mode:
            return int(self.group_choices[bin_idx])  # 确保返回整数
        
        if not is_training:
            # 测试模式：直接返回学习到的最优值（已应用单调性约束）
            return int(self.group_choices[bin_idx])  # 确保返回整数
        
        # 训练模式：自由探索，不受单调性约束限制
        if np.random.random() < self.exploration_prob:
            # 探索：在全范围内随机选择，允许违反单调性约束
            min_group = 2
            max_group = 8
            return np.random.choice(range(min_group, max_group + 1))
        else:
            # 利用：使用当前最优选择
            return int(self.group_choices[bin_idx])  # 确保返回整数
    
    def update_performance(self, snr_value: float, group_used: int, accuracy: float):
        """
        更新性能记录并调整策略
        
        Args:
            snr_value: SNR值
            group_used: 使用的group数量
            accuracy: 获得的准确率
        """
        # 推理模式下跳过性能更新和策略调整
        if self.inference_mode:
            return
            
        bin_idx = self.snr_to_bin(snr_value)
        key = (bin_idx, group_used)
        
        # 更新历史记录
        if key not in self.accuracy_history:
            self.accuracy_history[key] = []
            self.update_count[key] = 0
        
        self.accuracy_history[key].append(accuracy)
        self.update_count[key] += 1
        self.total_updates += 1
        

        if self.update_count[key] >= self.min_samples:
            self._update_strategy(bin_idx)
    
    def get_exploration_summary(self) -> Dict:
        """获取探索情况的统计摘要"""
        explored_combinations = set()
        for (bin_idx, group), count in self.update_count.items():
            if count >= self.min_samples:
                explored_combinations.add((bin_idx, group))
        
        # 统计每个bin有多少个group被充分探索
        bin_exploration = {}
        for bin_idx in range(self.snr_bins):
            explored_groups = [group for (b, group) in explored_combinations if b == bin_idx]
            bin_exploration[bin_idx] = len(explored_groups)
        
        return {
            'total_explored_combinations': len(explored_combinations),
            'bins_with_sufficient_exploration': sum(1 for count in bin_exploration.values() if count >= 3),
            'bin_exploration_details': bin_exploration,
            'exploration_coverage': len(explored_combinations) / (self.snr_bins * 7)  # 7个可能的group值
        }
    
    def _update_strategy(self, bin_idx: int):
        """更新指定SNR bin的group选择策略 - 训练期间不强制单调性约束"""
        current_group = self.group_choices[bin_idx]
        original_group = current_group
        
        # 获取当前group的平均性能
        current_key = (bin_idx, current_group)
        if current_key not in self.accuracy_history:
            return
        
        current_acc = np.mean(self.accuracy_history[current_key])
        best_group = current_group
        best_acc = current_acc
        
        # 尝试所有可能的group值（2-8），不受单调性约束限制
        for try_group in range(2, 9):
            if try_group == current_group:
                continue
                
            try_key = (bin_idx, try_group)
            if try_key in self.accuracy_history and len(self.accuracy_history[try_key]) >= self.min_samples:
                try_acc = np.mean(self.accuracy_history[try_key])
                
                # 选择性能最好的group，允许适度的精度下降来获得更高的带宽利用率
                if try_acc > best_acc or (try_acc >= best_acc - self.accuracy_threshold and try_group > best_group):
                    best_group = try_group
                    best_acc = try_acc
        
        # 更新策略（暂时不考虑单调性约束）
        self.group_choices[bin_idx] = int(best_group)  # 确保是整数
        
        # 统计策略变更
        if self.group_choices[bin_idx] != original_group:
            self.strategy_changes += 1
            #print(f"Strategy updated for SNR bin {bin_idx}: {original_group} -> {best_group} (acc: {best_acc:.4f})")
    
    def _is_monotonic_feasible(self, bin_idx: int, group_value: int) -> bool:
        """检查设置group_value是否违反单调性约束"""
        # 检查左邻居
        if bin_idx > 0 and self.group_choices[bin_idx - 1] > group_value:
            return False
        
        # 检查右邻居  
        if bin_idx < self.snr_bins - 1 and self.group_choices[bin_idx + 1] < group_value:
            return False
            
        return True
    
    def _get_min_allowed_group(self, bin_idx: int) -> int:
        """获取在单调性约束下的最小允许group值"""
        min_group = 2
        if bin_idx > 0:
            min_group = max(min_group, self.group_choices[bin_idx - 1])
        return min_group
    
    def _get_max_allowed_group(self, bin_idx: int) -> int:
        """获取在单调性约束下的最大允许group值"""
        max_group = 8
        if bin_idx < self.snr_bins - 1:
            max_group = min(max_group, self.group_choices[bin_idx + 1])
        return max_group
    
    def _enforce_monotonicity(self):
        """
        强制执行单调性约束：SNR越高，group_to_use越大
        按照SNR从小到大依次检查，以较小SNR对应的组数为基准进行增加或者不变
        """
        # 从左到右(SNR从小到大)扫描，确保序列单调非递减
        for i in range(1, self.snr_bins):
            # 如果当前值小于前一个值，说明违反了单调性
            if self.group_choices[i] < self.group_choices[i-1]:
                # 将当前值调整为前一个值（以较小SNR对应的组数为基准，只能增加或不变）
                self.group_choices[i] = int(self.group_choices[i-1])  # 确保是整数
    
    def get_strategy_summary(self) -> List[Tuple[float, int]]:
        """获取当前策略的摘要信息"""
        snr_values = [self.snr_min + i * self.snr_step for i in range(self.snr_bins)]
        return list(zip(snr_values, self.group_choices))
    
    def get_statistics(self) -> Dict:
        """获取算法统计信息"""
        return {
            'total_updates': self.total_updates,
            'strategy_changes': self.strategy_changes,
            'unique_combinations': len(self.accuracy_history),
            'current_strategy': self.group_choices.copy(),
            'snr_range': (self.snr_min, self.snr_max),
            'exploration_prob': self.exploration_prob
        }
    
    def reset_statistics(self):
        """重置统计信息（保留学习到的策略）"""
        self.total_updates = 0
        self.strategy_changes = 0
    
    def save_state(self) -> Dict:
        """保存当前状态"""
        return {
            'group_choices': self.group_choices.copy(),
            'accuracy_history': {str(k): v.copy() for k, v in self.accuracy_history.items()},
            'update_count': {str(k): v for k, v in self.update_count.items()},
            'total_updates': self.total_updates,
            'strategy_changes': self.strategy_changes,
            'snr_min': self.snr_min,
            'snr_max': self.snr_max,
            'snr_bins': self.snr_bins,
            '_finalized': getattr(self, '_finalized', False)
        }
    
    def load_state(self, state: Dict):
        """加载状态"""
        self.group_choices = state['group_choices']
        self.accuracy_history = {eval(k): v for k, v in state['accuracy_history'].items()}
        self.update_count = {eval(k): v for k, v in state['update_count'].items()}
        self.total_updates = state['total_updates']
        self.strategy_changes = state['strategy_changes']
        self.snr_min = state['snr_min']
        self.snr_max = state['snr_max']
        self.snr_bins = state['snr_bins']
        self.snr_step = (self.snr_max - self.snr_min) / (self.snr_bins - 1)
        self._finalized = state.get('_finalized', False)
    
    def _validate_group_choices(self):
        """验证并修正group_choices中的值，确保都是2-8范围内的整数"""
        for i in range(len(self.group_choices)):
            # 确保是整数
            if not isinstance(self.group_choices[i], int):
                self.group_choices[i] = int(round(self.group_choices[i]))
            
            # 确保在有效范围内
            self.group_choices[i] = max(2, min(8, self.group_choices[i]))
    
    def finalize_training(self):
        """
        训练完成后的最终策略调整：应用单调性约束并优化整体策略
        """
        print("Finalizing adaptive group selector training...")
        print(f"Strategy before monotonicity enforcement: {self.group_choices}")
        
        # 首先验证并修正所有group选择
        self._validate_group_choices()
        
        # 应用单调性约束
        self._enforce_monotonicity()
        
        # 再次验证以确保结果正确
        self._validate_group_choices()
        
        print(f"Final strategy after monotonicity enforcement: {self.group_choices}")
        
        # 降低探索概率，在测试时更多使用学习到的策略
        self.exploration_prob = 0.05
        
        # 设置最终化标记
        self._finalized = True
        
        # 打印最终策略摘要
        strategy_summary = self.get_strategy_summary()
        print("\nFinal strategy summary:")
        for snr, group in strategy_summary:
            print(f"SNR {snr:.1f}dB -> {group} groups")
        
        return strategy_summary
    
    def set_inference_mode(self, inference_mode: bool):
        """
        设置推理模式
        
        Args:
            inference_mode: True表示推理模式，False表示训练模式
        """
        self.inference_mode = inference_mode
        if inference_mode:
            print("Adaptive Group Selector: Switched to INFERENCE MODE")
            print("  - No more exploration or strategy updates")
            print("  - Using fixed learned strategy")
            print("  - Optimal for Final Training phase")
        else:
            print("Adaptive Group Selector: Switched to TRAINING MODE")
            print("  - Exploration and strategy updates enabled")
    
    def is_inference_mode(self) -> bool:
        """
        检查是否处于推理模式
        
        Returns:
            bool: 是否在推理模式
        """
        return self.inference_mode