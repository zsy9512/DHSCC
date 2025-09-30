import torch
import torch.nn as nn
import torch.nn.functional as F


class CapsuleNetwork(nn.Module):
    """
    胶囊网络模块，用于将输入数据分为8组，每组16维
    """
    def __init__(self, input_dim, num_groups=8, group_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.num_groups = num_groups
        self.group_dim = group_dim
        self.output_dim = num_groups * group_dim
        
        # 输入映射层
        self.input_projection = nn.Linear(input_dim, self.output_dim)
        
        # 胶囊网络层
        self.primary_capsules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(group_dim, group_dim),
                nn.ReLU(),
                nn.Linear(group_dim, group_dim),
                nn.LayerNorm(group_dim)
            ) for _ in range(num_groups)
        ])
        
        # 动态路由参数
        self.routing_weights = nn.Parameter(torch.randn(num_groups, group_dim, group_dim))
        
    def squash(self, tensor):
        """胶囊激活函数"""
        squared_norm = (tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 投影到目标维度
        x_proj = self.input_projection(x)  # [batch, num_groups * group_dim]
        
        # 重塑为胶囊格式
        x_grouped = x_proj.view(batch_size, self.num_groups, self.group_dim)
        
        # 通过主胶囊层处理每组
        capsule_outputs = []
        for i, capsule in enumerate(self.primary_capsules):
            group_output = capsule(x_grouped[:, i, :])  # [batch, group_dim]
            capsule_outputs.append(group_output)
        
        # 堆叠胶囊输出
        capsule_features = torch.stack(capsule_outputs, dim=1)  # [batch, num_groups, group_dim]
        
        # 应用squash激活
        capsule_features = self.squash(capsule_features)
        
        return capsule_features


class JointImportanceEstimator_group(nn.Module):
    """
    简化的基于胶囊网络分组的联合重要性估计器
    
    简化流程：
    1. 将联合特征 [batch, 257] 直接输入到胶囊网络分8组进行压缩
    2. 对压缩特征估计出重要性
    3. 乘上信道条件系数进行加权
    
    Args:
        feature_dim: 特征维度，默认128
        num_groups: 胶囊网络组数，默认8
        group_dim: 每组维度，默认16
    """
    def __init__(self, feature_dim=128, num_groups=8, group_dim=16, use_frequency_features=False, num_classes=10, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_groups = num_groups
        self.group_dim = group_dim
        self.use_frequency_features = use_frequency_features
        self.num_classes = num_classes
        # 冗余参数保留，不做使用

        assert feature_dim == num_groups * group_dim, \
            f"feature_dim ({feature_dim}) must equal num_groups * group_dim ({num_groups} * {group_dim})"

        # 分组胶囊权重
        self.W = nn.Parameter(torch.randn(num_groups, group_dim, group_dim))
        # 自注意力缩放
        self.attn_scale = group_dim ** -0.5
        # 门控融合MLP
        self.gate_mlp = nn.Sequential(
            nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid()
        )

    @staticmethod
    def squash(x, dim=-1):
        norm_sq = (x ** 2).sum(dim=dim, keepdim=True)
        return norm_sq / (1 + norm_sq) * x / (norm_sq.sqrt() + 1e-8)

    def dynamic_routing(self, u, num_iter=2):
        # u: [B, num_groups, group_dim]
        B = u.size(0)
        logits = u.new_zeros(B, self.num_groups)
        for _ in range(num_iter):
            c = F.softmax(logits, dim=1).unsqueeze(-1)  # [B, num_groups, 1]
            v = (c * u).sum(dim=1, keepdim=True)        # [B, 1, group_dim]
            v = self.squash(v)
            agreement = (u * v).sum(dim=-1)             # [B, num_groups]
            logits = logits + agreement
        c = F.softmax(logits, dim=1).unsqueeze(-1)
        v_i = c * u                                    # [B, num_groups, group_dim]
        return v_i, v

    def self_attention(self, v_i):
        # v_i: [B, num_groups, group_dim]
        attn_score = torch.einsum('bnd,bmd->bnm', v_i, v_i) * self.attn_scale
        attn = F.softmax(attn_score, dim=-1)           # [B, num_groups, num_groups]
        v_hat = torch.einsum('bnm,bmd->bnd', attn, v_i)
        return v_hat

    def expand_group_importance_to_feature_dim(self, group_importance):
        batch_size = group_importance.size(0)
        expanded_importance = group_importance.unsqueeze(-1).repeat(1, 1, self.group_dim)
        feature_importance = expanded_importance.view(batch_size, self.feature_dim)
        return feature_importance

    def forward(self, features, channel_importance=None, channel_noise=None, use_channel_guided=True):
        """
        新流程：
        1. 特征分组，映射为初级胶囊
        2. 两次动态路由得到组胶囊
        3. 自注意力融合
        4. 范数融合加权，softmax归一化
        5. 新方案：融合信道可靠性（如巴氏参数）和信道噪声，指导特征重要性生成
        """
        B = features.size(0)
        h = features.view(B, self.num_groups, self.group_dim)  # [B, num_groups, group_dim]
        # 初级胶囊
        u = torch.einsum('bnd,ndh->bnh', h, self.W)            # [B, num_groups, group_dim]
        u = self.squash(u)
        # 两次动态路由
        v_i, v_global = self.dynamic_routing(u, num_iter=2)
        # 自注意力
        v_hat = self.self_attention(v_i)
        # 范数（存在概率）
        p = v_i.norm(dim=-1)                                  # [B, num_groups]
        q = v_hat.norm(dim=-1)                                # [B, num_groups]
        # 门控融合
        gate = self.gate_mlp(torch.stack([p, q], dim=-1)).squeeze(-1)  # [B, num_groups]
        alpha_logits = gate * p + (1 - gate) * q
        alpha = F.softmax(alpha_logits, dim=1)                # [B, num_groups]
        final_importance = self.expand_group_importance_to_feature_dim(alpha)
        # --- 新方法：信道可靠性引导融合 ---
        fusion_importance = None
        if use_channel_guided and channel_importance is not None and channel_noise is not None:
            # channel_importance: [B, feature_dim] 或 [1, feature_dim] 或 [feature_dim]W
            # 自动扩展到batch维度
            if channel_importance.dim() == 1 and channel_importance.size(0) == self.feature_dim:
                channel_importance = channel_importance.unsqueeze(0).expand(B, -1)
            elif channel_importance.dim() == 2 and channel_importance.size(0) == 1 and channel_importance.size(1) == self.feature_dim:
                channel_importance = channel_importance.expand(B, -1)
            elif channel_importance.dim() == 2 and channel_importance.size(0) == B and channel_importance.size(1) == self.feature_dim:
                pass  # already correct
            else:
                raise RuntimeError(f"channel_importance shape {channel_importance.shape} does not match [B, feature_dim]")
            #极化码的复杂度太高，弃用
            channel_group = channel_importance.view(B, self.num_groups, self.group_dim).mean(dim=-1)
            fusion_logits = torch.exp(alpha_logits)/(torch.exp(-channel_noise)+1e-6)
            
            fusion_alpha = F.softmax(fusion_logits,dim=1)  # [B, num_groups]
            # 稀疏性门控：噪声越大，稀疏性越强（用exp(-noise)缩放softmax温度）
            fusion_importance = self.expand_group_importance_to_feature_dim(fusion_alpha)# [B, num_groups]
            #print("fusion_importance shape:", fusion_importance)
        # 返回三种重要性估计，便于解耦评估
        return final_importance, fusion_importance, channel_importance

