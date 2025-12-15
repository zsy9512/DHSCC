"""
自适应组选择器模块 - 优化版
保持原始API接口不变，仅优化内部核心逻辑：
1. 探索策略：从固定概率改为动态衰减 (Epsilon-Greedy Decay)
2. 组选择逻辑：在精度阈值范围内，强制优先选择更大的 Group (带宽利用率最大化)
3. 最终化逻辑：强制执行物理层单调性约束
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

class AdaptiveGroupSelector:
    """
    启发式算法用于根据信道条件自适应选择最优的group_to_use
    (API 兼容优化版)
    """
    
    def __init__(self, snr_min: float = -2.0, snr_max: float = 8.0, snr_bins: int = 21):
        """
        初始化自适应组选择器 (API不变)
        """
        # --- 原始参数保持不变 ---
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.snr_bins = snr_bins
        self.snr_step = (snr_max - snr_min) / (snr_bins - 1)
        
        # 初始化策略，默认为最小值2
        self.group_choices = [2] * snr_bins
        
        # 性能历史记录
        self.accuracy_history: Dict[Tuple[int, int], List[float]] = {}
        self.update_count: Dict[Tuple[int, int], int] = {}
        
        # --- 算法参数 (内部优化) ---
        self.accuracy_threshold = 0.001  # 精度容忍阈值
        self.min_samples = 5             # 降低最小样本数要求，加快收敛
        
        # [修改点]：exploration_prob 现在是动态变化的
        self.exploration_prob = 0.5      # 初始探索概率
        self.min_exploration = 0.05      # 最小探索概率
        self.exploration_decay = 0.9995  # 每次更新后的衰减率
        
        # 统计信息
        self.total_updates = 0
        self.strategy_changes = 0
        
        # 推理模式控制
        self.inference_mode = False
        self._finalized = False
    
    def snr_to_bin(self, snr_value: float) -> int:
        """将SNR值映射到对应的bin索引 (API不变)"""
        if isinstance(snr_value, torch.Tensor):
            snr_value = snr_value.item()
        
        # 增加边界保护
        if snr_value <= self.snr_min:
            return 0
        if snr_value >= self.snr_max:
            return self.snr_bins - 1
            
        bin_idx = int((snr_value - self.snr_min) / self.snr_step + 0.5) # 四舍五入更准确
        return max(0, min(bin_idx, self.snr_bins - 1))
    
    def get_group_to_use(self, snr_value: float, is_training: bool = True) -> int:
        """
        根据SNR值获取最优的group_to_use (API不变)
        """
        bin_idx = self.snr_to_bin(snr_value)
        
        # 1. 推理模式或最终化后：直接查表
        if self.inference_mode or self._finalized:
            return int(self.group_choices[bin_idx])
        
        # 2. 测试阶段（非训练）：直接利用当前最优
        if not is_training:
            return int(self.group_choices[bin_idx])
        
        # 3. 训练阶段：Epsilon-Greedy 探索
        # 注意：使用 self.exploration_prob，它现在会随时间衰减
        if np.random.random() < self.exploration_prob:
            # 探索：在 [2, 8] 范围内随机选择
            # 优化：可以倾向于在当前最优值附近探索，但全范围随机更不容易陷入局部最优
            return int(np.random.choice(range(2, 9)))
        else:
            # 利用：使用当前学习到的最优策略
            return int(self.group_choices[bin_idx])
    
    def update_performance(self, snr_value: float, group_used: int, accuracy: float):
        """
        更新性能记录并调整策略 (API不变)
        """
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
        
        # [修改点]：应用探索率衰减
        # 每次更新都让探索概率微弱下降，直到达到最小值
        if self.exploration_prob > self.min_exploration:
            self.exploration_prob *= self.exploration_decay

        # 只有样本数足够时才更新策略
        if self.update_count[key] >= self.min_samples:
            self._update_strategy(bin_idx)
    
    def get_exploration_summary(self) -> Dict:
        """获取探索情况的统计摘要 (API不变)"""
        explored_combinations = set()
        for (bin_idx, group), count in self.update_count.items():
            if count >= self.min_samples:
                explored_combinations.add((bin_idx, group))
        
        bin_exploration = {}
        for bin_idx in range(self.snr_bins):
            explored_groups = [group for (b, group) in explored_combinations if b == bin_idx]
            bin_exploration[bin_idx] = len(explored_groups)
        
        return {
            'total_explored_combinations': len(explored_combinations),
            'bins_with_sufficient_exploration': sum(1 for count in bin_exploration.values() if count >= 3),
            'bin_exploration_details': bin_exploration,
            'exploration_coverage': len(explored_combinations) / (self.snr_bins * 7),
            'current_exploration_prob': self.exploration_prob  # 新增统计项
        }
    
    def _update_strategy(self, bin_idx: int):
        """
        [核心修改] 更新指定SNR bin的group选择策略
        逻辑：找出所有已探索的组，在精度满足 [MaxAcc - Threshold] 的范围内，选择 Group 最大的那个。
        """
        current_choice = self.group_choices[bin_idx]
        
        # 1. 收集该 bin 下所有有效数据的 (group, accuracy)
        candidates = []
        for group in range(2, 9):
            key = (bin_idx, group)
            # 只有样本数足够才纳入考量
            if key in self.accuracy_history and len(self.accuracy_history[key]) >= self.min_samples:
                avg_acc = np.mean(self.accuracy_history[key])
                candidates.append((group, avg_acc))
        
        if not candidates:
            return

        # 2. 找到该 bin 下能达到的最高准确率
        max_acc = max(c[1] for c in candidates)
        
        # 3. 筛选出 "可接受" 的组 (准确率在最高值的容忍范围内)
        # 这就是 "在保证分类精度的前提下"
        acceptable_candidates = [
            (g, acc) for g, acc in candidates 
            if acc >= (max_acc - self.accuracy_threshold)
        ]
        
        # 4. 在可接受的组中，选择 Group 值最大的
        # 这就是 "尽可能使用更多的组"
        acceptable_candidates.sort(key=lambda x: x[0], reverse=True)
        best_group = acceptable_candidates[0][0]
        
        # 更新策略
        if best_group != current_choice:
            self.group_choices[bin_idx] = int(best_group)
            self.strategy_changes += 1
    
    def _is_monotonic_feasible(self, bin_idx: int, group_value: int) -> bool:
        """检查设置group_value是否违反单调性约束 (保留辅助函数)"""
        if bin_idx > 0 and self.group_choices[bin_idx - 1] > group_value:
            return False
        if bin_idx < self.snr_bins - 1 and self.group_choices[bin_idx + 1] < group_value:
            return False
        return True
    
    def _get_min_allowed_group(self, bin_idx: int) -> int:
        """API不变"""
        min_group = 2
        if bin_idx > 0:
            min_group = max(min_group, self.group_choices[bin_idx - 1])
        return min_group
    
    def _get_max_allowed_group(self, bin_idx: int) -> int:
        """API不变"""
        max_group = 8
        if bin_idx < self.snr_bins - 1:
            max_group = min(max_group, self.group_choices[bin_idx + 1])
        return max_group
    
    def _enforce_monotonicity(self):
        """
        [修改] 强制执行单调性约束：确保 SNR 越高，Group 数不降低。
        使用前向扫描 (Forward Pass) 策略。
        """
        # 从左到右扫描 (SNR -2 -> 8)
        # 物理逻辑：如果 SNR=-1dB 时能用 4 个组，那 SNR=0dB 时信道更好，至少也能用 4 个组。
        for i in range(1, self.snr_bins):
            if self.group_choices[i] < self.group_choices[i-1]:
                self.group_choices[i] = self.group_choices[i-1]
    
    def get_strategy_summary(self) -> List[Tuple[float, int]]:
        """API不变"""
        snr_values = [self.snr_min + i * self.snr_step for i in range(self.snr_bins)]
        return list(zip(snr_values, self.group_choices))
    
    def get_statistics(self) -> Dict:
        """API不变，增加部分内部状态返回"""
        return {
            'total_updates': self.total_updates,
            'strategy_changes': self.strategy_changes,
            'unique_combinations': len(self.accuracy_history),
            'current_strategy': self.group_choices.copy(),
            'snr_range': (self.snr_min, self.snr_max),
            'exploration_prob': self.exploration_prob # 返回当前的探索率
        }
    
    def reset_statistics(self):
        """API不变"""
        self.total_updates = 0
        self.strategy_changes = 0
        # 重置探索率? 通常不想重置衰减，保持当前状态
        # 如果需要完全重置训练，可以手动设回 0.5，这里遵循原逻辑只重置计数
    
    def save_state(self) -> Dict:
        """保存当前状态 (API不变)"""
        return {
            'group_choices': self.group_choices.copy(),
            # 兼容 key 为 tuple 的转存
            'accuracy_history': {str(k): v.copy() for k, v in self.accuracy_history.items()},
            'update_count': {str(k): v for k, v in self.update_count.items()},
            'total_updates': self.total_updates,
            'strategy_changes': self.strategy_changes,
            'snr_min': self.snr_min,
            'snr_max': self.snr_max,
            'snr_bins': self.snr_bins,
            '_finalized': getattr(self, '_finalized', False),
            # [新增] 保存当前的探索状态，以便断点续训
            'exploration_prob': self.exploration_prob
        }
    
    def load_state(self, state: Dict):
        """加载状态 (API不变，增加兼容性处理)"""
        self.group_choices = state['group_choices']
        # 使用 eval 恢复 tuple key
        self.accuracy_history = {eval(k): v for k, v in state['accuracy_history'].items()}
        self.update_count = {eval(k): v for k, v in state['update_count'].items()}
        self.total_updates = state['total_updates']
        self.strategy_changes = state['strategy_changes']
        self.snr_min = state['snr_min']
        self.snr_max = state['snr_max']
        self.snr_bins = state['snr_bins']
        self.snr_step = (self.snr_max - self.snr_min) / (self.snr_bins - 1)
        self._finalized = state.get('_finalized', False)
        
        # [新增] 尝试加载探索率，如果旧存档没有，默认为 min_exploration (假设旧存档是训练好的)
        self.exploration_prob = state.get('exploration_prob', 0.05)
    
    def _validate_group_choices(self):
        """API不变"""
        for i in range(len(self.group_choices)):
            if not isinstance(self.group_choices[i], int):
                self.group_choices[i] = int(round(self.group_choices[i]))
            self.group_choices[i] = max(2, min(8, self.group_choices[i]))
    
    def finalize_training(self):
        """
        训练完成后的最终策略调整 (API不变)
        """
        print("Finalizing adaptive group selector training...")
        print(f"Strategy before monotonicity enforcement: {self.group_choices}")
        
        # 1. 验证数据类型
        self._validate_group_choices()
        
        # 2. [核心] 强制单调性 (Forward Pass)
        self._enforce_monotonicity()
        
        # 3. 再次验证
        self._validate_group_choices()
        
        print(f"Final strategy after monotonicity enforcement: {self.group_choices}")
        
        # 锁定状态：将探索概率降至最低，并不再衰减
        self.exploration_prob = 0.0 
        self._finalized = True
        
        # 打印摘要
        strategy_summary = self.get_strategy_summary()
        print("\nFinal strategy summary:")
        for snr, group in strategy_summary:
            print(f"SNR {snr:.1f}dB -> {group} groups")
        
        return strategy_summary
    
    def set_inference_mode(self, inference_mode: bool):
        """API不变"""
        self.inference_mode = inference_mode

    def is_inference_mode(self) -> bool:
        """API不变"""
        return self.inference_mode