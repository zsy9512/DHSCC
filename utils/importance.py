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
        
        # 验证维度匹配
        assert feature_dim == num_groups * group_dim, \
            f"feature_dim ({feature_dim}) must equal num_groups * group_dim ({num_groups} * {group_dim})"
        
        # 联合特征的维度 = feature_dim + feature_dim + 1 = 128 + 128 + 1 = 257
        joint_feature_dim = feature_dim * 2 
        
        # 单一胶囊网络处理联合特征
        self.joint_capsule = CapsuleNetwork(joint_feature_dim, num_groups, group_dim)
        
        # 简化的重要性估计器 - 直接从胶囊特征预测组级重要性
        self.importance_estimator = nn.Sequential(
            nn.Linear(num_groups * group_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_groups),
            nn.Softmax(dim=-1)
        )
        
    def compute_gradient_proxy(self, pre_task_logits):
        """
        从pre_task_logits计算梯度代理
        
        Args:
            pre_task_logits: 预任务logits [batch_size, num_classes]
            
        Returns:
            gradient_proxy: 梯度代理特征 [batch_size, feature_dim]
        """
        batch_size = pre_task_logits.size(0)
        
        # 计算softmax概率
        task_probs = F.softmax(pre_task_logits, dim=-1)
        
        # 计算不确定性（熵）
        entropy = -torch.sum(task_probs * torch.log(task_probs + 1e-8), dim=-1, keepdim=True)
        
        # 计算最大概率（置信度）
        max_prob = torch.max(task_probs, dim=-1, keepdim=True)[0]
        
        # 计算预测方差
        prob_var = torch.var(task_probs, dim=-1, keepdim=True)
        
        # 结合多个梯度代理指标
        gradient_indicators = torch.cat([entropy, max_prob, prob_var], dim=-1)
        
        # 扩展到特征维度
        gradient_proxy = gradient_indicators.repeat(1, self.feature_dim // 3 + 1)[:, :self.feature_dim]
        
        return gradient_proxy
    
    def expand_group_importance_to_feature_dim(self, group_importance):
        """
        将组级别的重要性扩展到特征维度
        
        Args:
            group_importance: 组级别重要性 [batch_size, num_groups]
            
        Returns:
            feature_importance: 特征级别重要性 [batch_size, feature_dim]
        """
        batch_size = group_importance.size(0)
        
        # 将每个组的重要性复制到组内的所有维度
        # [batch_size, num_groups] -> [batch_size, num_groups, group_dim]
        expanded_importance = group_importance.unsqueeze(-1).repeat(1, 1, self.group_dim)
        
        # 重塑为特征维度 [batch_size, feature_dim]
        feature_importance = expanded_importance.view(batch_size, self.feature_dim)
        
        return feature_importance
        
    def forward(self, features, pre_task_logits):
        """
        简化的前向传播流程
        
        Args:
            features: 特征 [batch_size, feature_dim]
            pre_task_logits: 预任务logits [batch_size, num_classes]
            
        Returns:
            tuple: (final_importance, channel_importance, task_importance)
        """
        batch_size = features.size(0)

        # 步骤1: 计算梯度代理
        task_proxy = self.compute_gradient_proxy(pre_task_logits)
        
        # 步骤2: 构建联合特征 [batch, 256] = [batch, 128+128]
        joint_features = torch.cat([features, task_proxy], dim=-1)
        
        # 步骤3: 通过胶囊网络压缩为8组特征
        joint_capsules = self.joint_capsule(joint_features)  # [batch, num_groups, group_dim]
        
        # 步骤4: 展平胶囊特征并估计组级重要性
        capsule_flat = joint_capsules.view(batch_size, -1)  # [batch, num_groups * group_dim]
        group_importance = self.importance_estimator(capsule_flat)  # [batch, num_groups]
        
        # 步骤5: 重新归一化确保概率分布
        final_group_importance = F.softmax(group_importance, dim=-1)
        
        # 步骤6: 扩展到特征维度
        final_importance = self.expand_group_importance_to_feature_dim(final_group_importance)
        
        # 为了保持接口兼容性，返回相同的结果作为channel_importance和task_importance
        return final_importance, final_importance, final_importance
