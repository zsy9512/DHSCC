import torch
import torch.nn as nn
import torch.nn.functional as F

class JointImportanceEstimator_group(nn.Module):
    """
    基于胶囊网络分组的联合重要性估计器 (优化版)
    
    主要改进：
    1. 移除了未使用的 CapsuleNetwork 类。
    2. 修正了融合层数值稳定性问题 (避免双重 exp)。
    3. 增加了参数初始化 (Xavier)。
    4. 增强了代码的健壮性。
    """
    def __init__(self, feature_dim=128, num_groups=8, group_dim=16, 
                 use_frequency_features=False, num_classes=10, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_groups = num_groups
        self.group_dim = group_dim
        
        # 维度检查
        assert feature_dim == num_groups * group_dim, \
            f"feature_dim ({feature_dim}) must equal num_groups * group_dim ({num_groups} * {group_dim})"

        # 1. 胶囊变换权重 [num_groups, group_dim, group_dim]
        # 相当于对每组特征做一个独立的线性变换
        self.W = nn.Parameter(torch.Tensor(num_groups, group_dim, group_dim))
        
        # 2. 自注意力缩放系数
        self.attn_scale = group_dim ** -0.5
        
        # 3. 门控融合 MLP (输入维度为2: p和q)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2, 8), 
            nn.ReLU(), 
            nn.Linear(8, 1), 
            nn.Sigmoid()
        )
        
        # 执行参数初始化
        self._reset_parameters()

    def _reset_parameters(self):
        """使用 Xavier 初始化权重，防止梯度异常"""
        nn.init.xavier_uniform_(self.W)
        for m in self.gate_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def squash(x, dim=-1):
        """胶囊挤压激活函数：将向量模长压缩至 0-1 之间"""
        norm_sq = (x ** 2).sum(dim=dim, keepdim=True)
        return norm_sq / (1 + norm_sq) * x / (norm_sq.sqrt() + 1e-8)

    def dynamic_routing(self, u, num_iter=2):
        """
        组内自路由：计算各组特征与全局上下文的聚合程度
        u: [B, num_groups, group_dim]
        """
        B = u.size(0)
        # Logits 初始化为 0
        logits = torch.zeros(B, self.num_groups, device=u.device)
        
        v = None # 全局上下文向量
        v_i = None # 加权后的组向量

        for i in range(num_iter):
            # c: [B, num_groups, 1] - 路由系数 (归一化)
            c = F.softmax(logits, dim=1).unsqueeze(-1)
            
            # v: [B, 1, group_dim] - 全局上下文 (所有组的加权和)
            v = (c * u).sum(dim=1, keepdim=True)
            v = self.squash(v)
            
            # 最后一次迭代不需要更新 logits，直接计算输出
            if i < num_iter - 1:
                # agreement: [B, num_groups] - 组向量与全局上下文的相似度
                agreement = (u * v).sum(dim=-1)
                logits = logits + agreement
        
        # 最终输出计算
        c = F.softmax(logits, dim=1).unsqueeze(-1)
        v_i = c * u  # [B, num_groups, group_dim] - 根据全局一致性重加权的特征
        return v_i, v

    def self_attention(self, v_i):
        """
        组间自注意力：捕捉组与组之间的相关性
        v_i: [B, num_groups, group_dim]
        """
        # 计算注意力分数: [B, num_groups, num_groups]
        scores = torch.einsum('bnd,bmd->bnm', v_i, v_i) * self.attn_scale
        attn_weights = F.softmax(scores, dim=-1)
        
        # 聚合特征
        v_hat = torch.einsum('bnm,bmd->bnd', attn_weights, v_i)
        return v_hat

    def expand_group_importance_to_feature_dim(self, group_importance):
        """将组重要性扩展回原始特征维度"""
        batch_size = group_importance.size(0)
        # [B, num_groups] -> [B, num_groups, 1] -> [B, num_groups, group_dim]
        expanded = group_importance.unsqueeze(-1).expand(-1, -1, self.group_dim)
        # [B, feature_dim]
        return expanded.reshape(batch_size, self.feature_dim)

    def forward(self, features, channel_importance=None, channel_noise=None, use_channel_guided=True):
        """
        Args:
            features: [B, feature_dim]
            channel_noise: 标量或张量，表示信道噪声水平 (值越大噪声越大)
        """
        B = features.size(0)
        
        # 1. 变换与初级胶囊
        # [B, feature_dim] -> [B, num_groups, group_dim]
        h = features.view(B, self.num_groups, self.group_dim)
        # 线性变换: [B, n, d] * [n, d, d] -> [B, n, d]
        u = torch.einsum('bnd,ndh->bnh', h, self.W)
        u = self.squash(u)
        
        # 2. 动态路由 (提取全局一致性特征)
        v_i, _ = self.dynamic_routing(u, num_iter=2)
        
        # 3. 自注意力 (提取组间关联特征)
        v_hat = self.self_attention(v_i)
        
        # 4. 计算模长 (作为“存在概率”或“信息量”)
        p = v_i.norm(dim=-1)   # [B, num_groups] (路由分支)
        q = v_hat.norm(dim=-1) # [B, num_groups] (注意力分支)
        
        # 5. 门控融合
        # gate: [B, num_groups] (0~1)
        gate_input = torch.stack([p, q], dim=-1) # [B, num_groups, 2]
        gate = self.gate_mlp(gate_input).squeeze(-1) 
        
        # 融合后的 Logits (未经过 Softmax)
        alpha_logits = gate * p + (1 - gate) * q 
        
        # 基础重要性 (归一化)
        alpha = F.softmax(alpha_logits, dim=1)
        final_importance = self.expand_group_importance_to_feature_dim(alpha)
        
        # 6. 信道感知融合 (New Scheme)
        fusion_importance = None
        
        if use_channel_guided and channel_noise is not None:
            #print(channel_noise)
            # 处理 channel_noise 维度，确保它是 [B, 1] 或 [1]
            if isinstance(channel_noise, float):
                channel_noise = torch.tensor(channel_noise, device=features.device)
            channel_noise = channel_noise.view(1, 1)
            
            # 温度系数计算：噪声越大 -> exp(-noise)越小 -> 温度越低 -> 分布越尖锐(稀疏)
            # 温度系数 T = exp(-noise) + epsilon
            temperature = torch.exp(channel_noise) + 1e-6
            
            # --- 关键修正 ---
            # 直接缩放 Logits，而不是对 exp(logits) 再做除法
            # 这样避免了数值溢出，且物理意义更明确：High Noise = Low Temperature = Sharp Softmax
            fusion_logits = alpha_logits * temperature
            
            fusion_alpha = F.softmax(fusion_logits, dim=1) # [B, num_groups]
            fusion_importance = self.expand_group_importance_to_feature_dim(fusion_alpha)

        return final_importance, fusion_importance, None