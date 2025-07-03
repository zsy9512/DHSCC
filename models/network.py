"""
Neural network modules for vibration signal classification with noise-condition attention.

This module contains the implementation of:
- Feature importance estimation (gamma functions)
- Mamba-based sequence classifier for advanced temporal modeling
- VibrationVLVFENet: Main model for vibration signal classification with variable-length features
- Noise-condition attention mechanism for noise-adaptive decoder
- Support for various wireless communication modes for feature transmission

The architecture implements a Variable-Length Variable Feature Encoder (VL-VFE)
that can dynamically adjust the number of features transmitted based on channel conditions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from models.vq import GroupVectorQuantizer
from wireless_communication import WirelessCodec
from utils.importance import JointImportanceEstimator_group,JointImportanceEstimator
from utils.frequency_importance import *  # Import frequency-based importance classes
from models.adaptive_group_selector import AdaptiveGroupSelector
from models.rl_adaptive_group_selector import RLAdaptiveGroupSelector
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class gamma_layer(nn.Module):
    """
    Implementation of a gamma layer for feature importance estimation.
    
    This layer computes importance scores for features based on channel noise,
    ensuring weights are positive for numerical stability.
    
    Args:
        input_channel: Input dimension
        output_channel: Output dimension
    """
    def __init__(self, input_channel, output_channel):
        super(gamma_layer, self).__init__()
        self.H = nn.Parameter(torch.ones(output_channel, input_channel))
        self.b = nn.Parameter(torch.ones(output_channel))
        # Initialize with small values for stable training
        self.H.data.normal_(0, 0.1)
        self.b.data.normal_(0, 0.001)
        self.output_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        H = torch.abs(self.H)  # Ensure weights are positive for stability
        x = F.linear(x, H)
        return torch.tanh(x)  # tanh activation for [-1,1] range

class gamma_function(nn.Module):
    """
    Multi-layer MLP implementation for mapping channel state to feature importance.
    
    This implements the feature importance function γ(σ²) from the paper,
    which maps channel noise level to importance scores for each feature.
    
    Args:
        intermediate_dim: Dimension of the final feature importance scores
    """
    def __init__(self, intermediate_dim=64):
        super(gamma_function, self).__init__()
        # 4-layer MLP as described in the paper section IV-B
        self.f1 = gamma_layer(1, 16)
        self.f2 = gamma_layer(16, 16)
        self.f3 = gamma_layer(16, 16)
        self.f4 = gamma_layer(16, intermediate_dim)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        return x

class Flatten(nn.Module):
    """Flatten layer for reshaping tensors."""
    def forward(self, x): 
        return x.view(x.size(0), -1)

class Mul(nn.Module):
    """Multiplication layer for scaling tensors."""
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x): 
        return x * self.weight
class SimpleClassifier(nn.Module):
    """
    Simple sequence classifier for vibration signals.
    
    Implements a CNN-based model for processing sequential data,
    which is efficient for shorter sequences like vibration signals.
    
    Args:
        input_dim: Input feature dimension
        seq_len: Sequence length
        hidden_dim: Hidden dimension for processing
        num_classes: Number of output classes
    """
    def __init__(self, input_dim, seq_len, hidden_dim, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Sequential model for classification
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Output projection
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # Input shape: [batch_size, channels, seq_len]
        x = self.feature_extractor(x)
        x = x.squeeze(-1)  # Remove the sequence dimension
        output = self.classifier(x)
        return output

class FeatureExtractor(nn.Module):
    """
    Feature extraction network for vibration signals with SSM-enhanced temporal modeling.
    
    This module combines convolutional layers for spatial feature extraction with Mamba-based
    SSM blocks for capturing long-range dependencies in the temporal domain. The architecture
    uses parallel paths to leverage the strengths of both approaches.
    
    Args:
        input_dim: Input channel dimension (usually 1 for raw signals)
        output_dim: Output feature dimension
        use_ssm: Whether to use SSM blocks for temporal modeling
    """
    def __init__(self, input_dim=1, output_dim=512, use_ssm=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_ssm = use_ssm
        
        # Initial feature extraction (keep as CNN for efficient local feature extraction)
        self.prep = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Layer 1 - Main CNN path with downsampling
        self.layer1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # Residual block for layer 1 (keep as CNN for local refinement)
        self.layer1_res = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Layer 2 with downsampling - CNN path
        self.layer2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Layer 3 with downsampling (keep as CNN)
        self.layer3 = nn.Sequential(
            nn.Conv1d(256, output_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Residual block for layer 3 (keep as CNN)
        self.layer3_res = nn.Sequential(
            nn.Conv1d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        """
        Forward pass through the feature extraction network.
        
        Args:
            x: Input tensor of shape [batch, channels, seq_len]
            
        Returns:
            Extracted features of shape [batch, output_dim, seq_len//16]
        """
        # Initial feature extraction
        x = self.prep(x)
        
        # Layer 1 processing - parallel CNN and SSM paths

        x_layer1 = self.layer1(x)
        
        # Apply residual connection for layer 1
        res = self.layer1_res(x_layer1)
        if res.size() != x_layer1.size():
            res = F.adaptive_avg_pool1d(res, x_layer1.size(2))
        x = res + x_layer1
        x = self.layer2(x)
        
        # Layer 3 processing (CNN)
        x = self.layer3(x)
        
        return x
    
    def apply_residual(self, x):
        """
        Apply residual connection for the output features.
        
        Args:
            x: Input tensor from decoder
            
        Returns:
            Tensor with residual connection applied
        """
        # Apply residual connection
        res = self.layer3_res(x)
        
        # Ensure compatible shapes
        if res.size() != x.size():
            res = F.adaptive_avg_pool1d(res, x.size(2))
        
        return res + x

class FeatureEncoder(nn.Module):
    """
    特征编码器模块，负责将高维特征压缩为低维特征表示。
    
    该模块实现了论文中的编码函数φ(s)，包括以下步骤：
    1. 卷积降维（encoder1）
    2. 固定维度池化（encoder_pool）
    3. 线性投影和归一化（encoder2）
    4. 特征稀疏化和权重归一化（encoder3）
    
    Args:
        input_channels (int): 输入通道数
        hidden_dim (int): 隐藏特征维度
        output_dim (int): 输出特征维度
    """
    def __init__(self, input_channels=512, hidden_dim=64, output_dim=64):
        super().__init__()
        
        # 卷积降维编码器
        self.conv_encoder = nn.Sequential(
            nn.Conv1d(input_channels, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )
        
        # 固定维度池化
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16)
        
        # 计算展平后的特征维度
        self.flat_dim = 8 * 16  # 8通道 x 16时间步
        
        # 线性投影和归一化
        self.feature_projector = nn.Sequential(
            nn.Linear(self.flat_dim, hidden_dim),
            nn.Sigmoid()
        )

        # 特征稀疏化参数 - 关键组件
        self.sparsifier_weight = nn.Parameter(torch.Tensor(output_dim, hidden_dim))
        self.sparsifier_bias = nn.Parameter(torch.Tensor(output_dim))
        
        # 参数初始化
        self.sparsifier_weight.data.normal_(0, 0.5)
        self.sparsifier_bias.data.normal_(0, 0.1)
        
    def forward(self, x):
        """
        编码器前向传播。
        
        Args:
            x: 输入特征 [batch_size, channels, time]
            
        Returns:
            tuple: (特征, 权重, 偏置) 包含：
                特征: 经过线性投影的特征
                权重: 归一化的权重参数
                偏置: 归一化的偏置参数
        """
        batch_size = x.size(0)
        
        # 第1阶段：卷积降维
        x = self.conv_encoder(x)
        x = self.adaptive_pool(x)
        x = x.view(batch_size, -1)
        x = self.feature_projector(x)
        
        # 特征归一化
        x_norm = torch.norm(x, dim=1, keepdim=True) + 1e-6
        x = x.size(-1) * x / x_norm  # 归一化特征向量长度
        
        # 第4阶段：准备稀疏化权重和偏置
        weight = F.tanh(self.sparsifier_weight)
        bias = F.tanh(self.sparsifier_bias)
        
        # 确保权重为非零以增加稳定性
        weight = torch.clamp(torch.abs(weight), min=1e-3) * torch.sign(weight.detach())
        bias = torch.clamp(torch.abs(bias), min=1e-3) * torch.sign(bias.detach())
        
        # 权重L2归一化
        l2_norm = (torch.sum(weight.pow(2), dim=1, keepdim=True) + 
                   bias.pow(2).unsqueeze(1)).sqrt() + 1e-6
        weight = weight / l2_norm.transpose(0, 1)
        bias = bias / l2_norm.squeeze()
        
        # 应用线性变换
        x = F.linear(x, weight, bias)
        
        return x, weight, bias

class NoiseConditionAttention(nn.Module):
    """
    噪声条件注意力模块，根据噪声水平调整特征注意力。
    
    该模块实现了论文"Learning Task-Oriented Communication for Edge Inference: An Information Bottleneck Approach"
    中的注意力特征(AF)模块设计思想的一部分，通过将噪声水平映射为特征权重，让解码器能够根据当前噪声水平自适应地
    调整不同特征的重要性。
    
    参数:
        hidden_dim (int): 隐藏层维度
        noise_feat_dim (int): 噪声特征维度
    """
    def __init__(self, hidden_dim, noise_feat_dim=16):
        super(NoiseConditionAttention, self).__init__()
        
        # 噪声编码器 - 将标量噪声映射为特征表示
        self.noise_encoder = nn.Sequential(
            nn.Linear(1, noise_feat_dim),
            nn.ReLU(),
            nn.Linear(noise_feat_dim, noise_feat_dim),
            nn.ReLU(),
            nn.Linear(noise_feat_dim, noise_feat_dim),
            nn.ReLU()
        )
        
        # 注意力生成网络 - 从噪声特征生成注意力权重
        self.attention_generator = nn.Sequential(
            nn.Linear(noise_feat_dim, noise_feat_dim),
            nn.Tanh(),
            nn.Linear(noise_feat_dim, hidden_dim),
            nn.Sigmoid()  # 使用Sigmoid确保权重在0-1之间
        )
        
    def forward(self, x, noise_level):
        """
        前向传播
        
        参数:
            x: 输入特征 [batch_size, hidden_dim]
            noise_level: 噪声水平 [batch_size, 1]
            
        返回:
            根据噪声条件调整的特征 [batch_size, hidden_dim]
        """
        # 编码噪声
        noise_feat = self.noise_encoder(noise_level)
        
        # 生成注意力权重
        noise_attention = self.attention_generator(noise_feat)
        
        # 应用注意力权重
        return x * noise_attention

class NoiseAdaptiveDecoder(nn.Module):
    """
    A noise-adaptive decoder with noise condition attention for vibration signal features.
    
    This module takes encoded features and channel noise information as input,
    and uses noise-conditioned attention to adaptively process features.
    
    Args:
        hidden_dim (int): Dimension of hidden features
        output_flat_dim (int): Flattened output dimension for reshaping
        output_channels (int): Number of output channels after reshaping
    """
    def __init__(self, hidden_dim, output_flat_dim, output_channels=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_flat_dim = output_flat_dim
        self.output_channels = output_channels
        
        # Initial feature processing
        self.feature_decoder = nn.Linear(hidden_dim, hidden_dim)
        
        # Noise condition attention - core improvement
        self.noise_condition_attention = NoiseConditionAttention(hidden_dim)
        
        # Feature refinement 
        self.feature_refinement = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        
        # Reshape layer for convolutional processing
        self.reshape_layer = nn.Linear(hidden_dim, output_flat_dim)
        
    def forward(self, x, noise_level):
        """
        Forward pass through the noise-adaptive decoder.
        
        Args:
            x: Input features [batch_size, hidden_dim]
            noise_level: Scalar noise level [batch_size, 1] or [1]
            
        Returns:
            Reshaped features ready for convolution
        """
        batch_size = x.size(0)
        
        # Initial feature processing
        x = F.relu(self.feature_decoder(x))
        
        # Apply noise condition attention - adaptive feature weighting
        x = self.noise_condition_attention(x, noise_level)
        # Further feature refinement
        x = self.feature_refinement(x)
        # Reshape for convolutional processing
        x_reshaped = self.reshape_layer(x)
        
        return x_reshaped

class FeatureReconstructionModule(nn.Module):
    """
    Module for reconstructing high-dimensional features from decoder output.
    
    Args:
        in_channels (int): Input channel dimension
        out_channels (int): Output channel dimension
    """
    def __init__(self, in_channels=8, out_channels=512):
        super().__init__()
        
        # Convolutional reconstruction network
        self.conv_reconstruction = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x, target_shape=None):
        """
        Reconstruct high-dimensional features.
        
        Args:
            x: Input tensor [batch, in_channels, seq_len]
            target_shape: Optional target shape for adaptive pooling
            
        Returns:
            Reconstructed features with shape [batch, out_channels, seq_len]
        """
        # Convolutional reconstruction
        features = self.conv_reconstruction(x)
        
        # Ensure output has the correct shape if specified
        if target_shape is not None and features.size() != target_shape:
            features = F.adaptive_avg_pool1d(features, target_shape[2])
            
        return features

class ClassifierModule(nn.Module):
    """
    Classification module for vibration signals.
    
    This module handles the final processing and classification of features.
    
    Args:
        input_dim (int): Input feature dimension
        seq_len (int): Target sequence length after pooling
        hidden_dim (int): Hidden dimension for classifier
        num_classes (int): Number of output classes
    """
    def __init__(self, input_dim=512, seq_len=8, hidden_dim=256, num_classes=10):
        super().__init__()
        
        # Fixed pooling to standardize sequence length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(seq_len)
        
        # Simple classifier 
        self.simple_classifier = SimpleClassifier(
            input_dim=input_dim,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )
        
    def forward(self, x):
        """
        Forward pass through the classifier module.
        
        Args:
            x: Input features [batch, channels, variable_seq_len]
            
        Returns:
            Classification scores [batch, num_classes]
        """
        # Standardize sequence length
        x = self.adaptive_pool(x)
        
        # Classify using Simple Classifier (or could use Mamba)
        output = self.simple_classifier(x)
        
        return output

class VibrationVLVFENet(nn.Module):
    """
    Variable-Length Variable Feature Encoder Network for vibration signal classification.
    
    This model implements an end-to-end system for vibration signal classification
    with dynamic feature selection based on channel conditions. It includes:
    
    1. Feature extraction using 1D convolutional networks
    2. Feature encoding with variable importance weighting
    3. Feature transmission through various communication modes
    4. Feature decoding and reconstruction with noise condition attention
    5. Final classification using a classifier
    
    Args:
        args: Configuration parameters
        num_classes: Number of output classes
    """
    def __init__(self, args, num_classes=10):
        super().__init__()

        self.args = args
        self.hidden_channel = args.intermediate_dim  # Compressed feature dimension
        self.num_classes = num_classes
        self.snr_max = args.snr_max
        self.snr_min = args.snr_min
        self.test = args.test
        self.threshold = args.threshold
        
        # 频率特征模式控制参数
        self.use_frequency_features = getattr(args, 'use_frequency_features', False)
        
        # Communication mode (default to mode 1 if not provided)
        self.comm_mode = getattr(args, 'comm_mode', 1)
        
        # 添加自适应组选择器 - 支持算法选择
        self.selector_algorithm = getattr(args, 'selector_algorithm', 'heuristic')  # 'heuristic' or 'rl'
        
        if self.selector_algorithm == 'rl':
            self.adaptive_group_selector = RLAdaptiveGroupSelector(
                psnr_min=args.snr_min,
                psnr_max=args.snr_max,
                psnr_bins=21,  # RL使用更多bins
                learning_rate=getattr(args, 'selector_lr', 1e-4),
                buffer_size=10000,
                batch_size=32
            )
            print(f"Using RL-based adaptive group selector with {self.adaptive_group_selector.psnr_bins} bins")
            print(f"RL parameters: lr={getattr(args, 'selector_lr', 1e-4)}, device={device}")
        else:
            self.adaptive_group_selector = AdaptiveGroupSelector(
                psnr_min=args.snr_min, 
                psnr_max=args.snr_max,
                psnr_bins=11  # 直接使用细粒度11个bins
            )
            print(f"Using heuristic adaptive group selector with {self.adaptive_group_selector.psnr_bins} bins")
        
        # 分阶段训练控制参数
        self.encoder_pretrain_epochs = 100  # 前100个epoch预训练编解码器
        self.encoder_weights_frozen = False  # 发送端权重是否已冻结
        self.encoder_select_epochs =200
        # 用于收集性能数据
        self.batch_accuracies = []
        self.batch_psnrs = []
        self.batch_groups = []
        
        # Communication modes:
        # 1: Direct Gaussian noise on encoded feature 
        # 2: Gaussian noise on quantized feature
        # 3: Digital communication without channel coding
        # 4: Digital communication with channel coding
        print(f"Using communication mode {self.comm_mode}")

        # Feature extraction network
        self.feature_extractor = FeatureExtractor(input_dim=1, output_dim=512,use_ssm=False)
        
        # Configuration for wireless communication
        self.n = 8
        self.N = 2**8
        self.k = 128
        self.wcodec = WirelessCodec(n=self.N, k=self.k, num_bits_per_symbol=2, device=device)
        
        # Classifier module
        self.classifier_module = ClassifierModule(
            input_dim=512,      # Channel dimension from preceding layers
            seq_len=8,          # Sequence length after adaptive pooling
            hidden_dim=256,     # Hidden dimension for processing
            num_classes=num_classes
        )
        
        # Feature encoder
        self.feature_encoder = FeatureEncoder(
            input_channels=512,
            hidden_dim=self.hidden_channel,
            output_dim=self.hidden_channel
        )

        # Noise-adaptive decoder with noise condition attention
        self.noise_adaptive_decoder = NoiseAdaptiveDecoder(
            hidden_dim=self.hidden_channel,
            output_flat_dim=self.feature_encoder.flat_dim,
            output_channels=8
        )
        
        # Feature reconstruction module
        self.feature_reconstruction = FeatureReconstructionModule(
            in_channels=8,
            out_channels=512
        )
        
        # Initialize Grouped Vector Quantizer (GVQ)
        self.GVQ = GroupVectorQuantizer(
            dimension=128,             # Feature dimension
            n_codes_group=8,           # Number of groups (GVQ)
            bins=16,                   # Codebook size per group
            n_q=4*8                    # Number of quantizers (total across all groups)
        )
        
        self.Tanh = nn.Tanh()

        frequency_config = {'use_frequency_features': False}
        
        self.gamma_mu = JointImportanceEstimator_group(
            feature_dim=self.hidden_channel,
            num_groups=8,
            group_dim=16,
            **frequency_config
        )
        # Upper triangular matrix for feature importance correlation
        self.pre_classifier = nn.Linear(self.hidden_channel, num_classes)
        self.upper_tri_matrix = torch.triu(torch.ones((self.hidden_channel, self.hidden_channel))).to(device)
    def forward(self, x, epoch, snr=0.1):
        """
        Forward pass through the VL-VFE network with integrated adaptive training.
        
        Training Strategy for Mode 4:
        - Encoder Pretrain Phase (epoch <= encoder_pretrain_epochs): 
          Use noise-free VQ-VAE training (similar to mode 2) to establish stable 
          encoding-decoding without channel noise interference
        - Formal Training Phase (epoch > encoder_pretrain_epochs):
          Apply physical layer transmission with channel coding
        - Test Phase: Always use physical layer transmission
        
        Args:
            x: Input vibration signal [batch_size, channels, time]
            epoch: Current training epoch for stage control
            snr: Signal-to-noise ratio (used in test mode)
            
        Returns:
            tuple: (classification_output, KL_divergence, commitment_loss, bit_error_rate)
        """
        batch_size = x.size(0)
        
        # 自动处理分阶段训练：在进入选择器训练阶段时冻结发送端权重
        if self.is_selector_update_phase(epoch) and not self.encoder_weights_frozen:
            self._freeze_sender_weights()
        
        # Feature extraction
        x = self.feature_extractor(x)
        
        # Save shape for later reconstruction
        decoded_feature_shape = x.shape

        # Feature encoding
        x, weight3, bias3 = self.feature_encoder(x)

        # Determine SNR/noise level
        if self.test:
            # Fixed SNR in test mode
            psnr_value = torch.tensor([snr], device=device).float()
        else:
            # Random SNR in training mode within specified range
            psnr_value = torch.rand(
                size=(1,),
                device=device
            ) * (self.snr_max - self.snr_min + torch.finfo(torch.float32).eps) + self.snr_min
        
        # Convert PSNR to noise value
        from utils.signal_processing import psnr_to_noise
        channel_noise = psnr_to_noise(psnr_value)
        
        # Calculate feature importance based on channel noise
        
        # Generate pre-task logits for the hybrid attention mechanism
        pre_task_logits = self.pre_classifier(x.detach()) 
        #print(f"Pre-task logits shape: {pre_task_logits}, PSNR: {psnr_value.item():.2f}dB")
        # Use joint importance estimator to calculate feature importance
        # Expand psnr_value to match batch size instead of reshaping
        channel_noise_bar= channel_noise.expand(batch_size, 1) 
        # 使用联合重要性估计器，返回三个重要性分数
        final_importance, channel_importance, task_importance = self.gamma_mu(
            channel_noise_bar, x, pre_task_logits
        )
        # Apply correlation matrix transformation 
        #mu = F.linear(final_importance, self.upper_tri_matrix)
        mu = torch.clamp(final_importance, min=1e-4)  # Ensure non-zero values for stability  
        #mu=torch.sort
        # 使用联合重要性直接与特征相乘得到encoded_feature
        sorted_mu, sort_indices = torch.sort(mu, dim=-1, descending=True)
        
        # 根据排序索引对特征x进行重新排序
        batch_size = x.size(0)
        # 扩展索引以匹配批次维度
        expanded_indices = sort_indices.expand(batch_size, -1)
        sorted_x = torch.gather(x, dim=1, index=expanded_indices)
        # 使用排序后的mu作为注意力权重
        attn_weights = sorted_mu
        original_feature=sorted_x
        weighted_features = sorted_x*attn_weights 
        
        # 保存排序索引以便后续可能的反向排序
        self.last_sort_indices = sort_indices
        

        # Apply tanh to constrain values between [-1, 1]
        encoded_feature = torch.tanh(weighted_features)
        
        # Ensure small values have some minimal magnitude, but preserve signs
        sign = torch.sign(encoded_feature)
        abs_encoded = torch.abs(encoded_feature)
        encoded_feature = torch.clamp(abs_encoded, min=1e-3) * sign
        
        # 确保特征值在[-1,1]范围内
        encoded_feature = torch.clamp(encoded_feature, min=-1.0, max=1.0)
        #print(f"Encoded feature shape: {encoded_feature.mean()}, PSNR: {encoded_feature.max():.2f}dB")
        # Calculate KL divergence for information bottleneck
        KL = self.KL_log_uniform(channel_noise, torch.abs(encoded_feature))
        
        # Default BER is 0 for non-digital modes
        ber = torch.tensor(0.0).to(device=device)
        commit_loss = torch.tensor(0.0).to(device=device)
        transmit_mse=torch.tensor(0.0).to(device=device)
        group_to_use =8
        # Initialize zeros_count for tracking pruned dimensions
        zeros_count = 0
        
        # Always create mask based on feature importance

        
        # #print(f"Zeros count: {zeros_count}, PSNR: {psnr_value.item():.2f}dB, Mask shape: {mask.shape}")
        # group_to_use = max(8 - zeros_count // 16, 2)  # Map zeros to groups
        
        # Implement the four communication modes
        if self.comm_mode == 1:
            mask = self.get_mask(attn_weights).mean(dim=0, keepdim=True)
            zeros_count = torch.sum(mask == 0).item()
            sent_feature= encoded_feature * mask
            # Mode 1: Direct Gaussian noise on encoded_feature (analog transmission)
            x = sent_feature+ torch.randn_like(encoded_feature) * channel_noise
            # Always apply pruning mask in mode 1
            
        elif self.comm_mode == 2:
            # Mode 2: Gaussian noise on quantized_feature (analog with quantization)
            quantized_feature, all_indices, commit_loss = self.GVQ(encoded_feature)
            x = quantized_feature
            sent_feature = quantized_feature  
        elif self.comm_mode == 3 or self.comm_mode == 4:
            # Modes 3/4: Digital transmission with/without channel coding
            quantized_feature, all_indices, commit_loss = self.GVQ(encoded_feature)
            
            use_coding = False

            # 修正：在预训练阶段，训练和测试都使用默认的group_to_use=8
            if epoch < self.encoder_pretrain_epochs:
                # 预训练阶段：始终使用默认的group_to_use=8，无论训练还是测试
                group_to_use = self.get_default_group_to_use(psnr_value.item())
                use_coding = False
                #print(f"Pretraining phase: using default group_to_use={group_to_use}")
            elif epoch < self.encoder_select_epochs:
                # 选择器训练阶段：使用自适应选择器进行训练和探索
                use_coding = True
                group_to_use = self.adaptive_group_selector.get_group_to_use(
                    psnr_value.item(), 
                    is_training=True
                )
                #print(f"Selector training phase: using adaptive group selection: {group_to_use} groups")
            else:
                # Final Training阶段：使用已训练好的选择器（推理模式）
                use_coding = True
                group_to_use = self.adaptive_group_selector.get_group_to_use(
                    psnr_value.item(), 
                    is_training=False  # 强制使用推理模式，不再探索
                )
                #print(f"Final training phase: using fixed strategy: {group_to_use} groups")
            # 记录当前批次的信息用于后续性能更新
            if self.training:
                self.batch_psnrs.append(psnr_value.item())
                self.batch_groups.append(group_to_use)
                
                # 只传输选定的组
            transmit_indices = all_indices[:, :group_to_use, :] 
            sent_feature=self.GVQ.decode(transmit_indices)
            #print(f"Transmit indices shape: {transmit_indices.shape}, Group to use: {group_to_use}")
            # 模拟无线传输
            received_indices, ber, _ = self.wcodec.transmit_through_channel(
                transmit_indices, 
                psnr_value, 
                bits_per_index=4,
                group_to_use=group_to_use,
                use_coding=use_coding
            )
            # 解码接收的索引
            rx = self.GVQ.decode(received_indices)
            
            # Update encoded feature with received values, keeping gradients intact
            x = encoded_feature + (rx - encoded_feature).detach()
        
        # Use noise-adaptive decoder with noise condition attention
        transmit_mse = F.mse_loss(x, sent_feature, reduction='mean')
        
        x_reshaped = self.noise_adaptive_decoder(x, channel_noise)
       
        # Reshape for convolutional processing
        x = x_reshaped.view(batch_size, 8, 16)
        
        # Reconstruct high-dimensional features
        decoded_feature = self.feature_reconstruction(x, target_shape=decoded_feature_shape)
        
        # Final residual connection
        x = self.feature_extractor.apply_residual(decoded_feature)
        
        # Classification with the classifier module
        output = self.classifier_module(x)
        

        # Return raw data for post-processing analysis
        return output, KL*0.1/channel_noise, torch.mean(commit_loss), ber, transmit_mse*channel_noise, self.threshold, zeros_count, group_to_use, {
            'sent_feature': sent_feature.detach().cpu(),
            'encoded_feature': encoded_feature.detach().cpu(), 
            'original_feature': original_feature.detach().cpu(),
            'psnr': psnr_value.item(),
            'attn_weights': attn_weights.detach().cpu()
        }

    def update_performance(self, batch_accuracy, epoch):
        """
        更新自适应选择器的性能数据（自动集成到训练流程中）
        
        Args:
            batch_accuracy: 当前批次的准确率
            epoch: 当前epoch
        """
        # 只在选择器训练阶段更新性能数据
        if self.is_selector_update_phase(epoch) and len(self.batch_psnrs) > 0 and len(self.batch_groups) > 0:
            # 使用最近一个批次的数据更新性能
            psnr = self.batch_psnrs[-1]
            group = self.batch_groups[-1]
            
            self.adaptive_group_selector.update_performance(
                psnr, group, batch_accuracy
            )

    def KL_log_uniform(self, channel_noise, encoded_feature):
        """
        Calculate KL divergence between encoded feature distribution and log-uniform prior.
        
        This implements the information bottleneck loss from the paper.
        
        Args:
            channel_noise: Current channel noise level
            encoded_feature: Encoded feature values
            
        Returns:
            torch.Tensor: KL divergence loss
        """
        log_alpha = torch.log(channel_noise) - torch.log(encoded_feature)
        k1 = 0.63576
        k2 = 1.8732
        k3 = 1.48695
        two_log_alpha = 2 * log_alpha
        KL_term = k1 * F.sigmoid(k2 + k3 * 2 * log_alpha) - 0.5 * F.softplus(-2 * two_log_alpha) - k1
        return -torch.sum(KL_term) / encoded_feature.size(0)

    def get_mask(self, mu, threshold=None):
        """
        Generate binary mask based on feature importance.
        
        Args:
            mu: Feature importance scores
            threshold: Threshold for binarization, uses self.threshold if None
            
        Returns:
            torch.Tensor: Binary mask (1 for important features, 0 for unimportant)
        """
        if threshold is None:
            threshold = self.threshold  # Use adaptively adjusted threshold
        alpha = mu.detach()
        hard_mask = (alpha > threshold).float()
        return hard_mask
    
    def get_group_strategy_summary(self):
        """获取当前的组选择策略摘要"""
        return self.adaptive_group_selector.get_strategy_summary()
    
    def get_adaptive_selector_statistics(self):
        """获取自适应选择器的统计信息"""
        stats = self.adaptive_group_selector.get_statistics()
        # 添加算法类型信息
        stats['algorithm_type'] = self.selector_algorithm
        return stats
    
    def reset_adaptive_selector_statistics(self):
        """重置自适应选择器的统计信息（保留学习到的策略）"""
        self.adaptive_group_selector.reset_statistics()
    
    def save_adaptive_selector_state(self):
        """保存自适应选择器的状态"""
        state = self.adaptive_group_selector.save_state()
        # 添加算法类型信息以便加载时确认兼容性
        state['algorithm_type'] = self.selector_algorithm
        return state
    
    def load_adaptive_selector_state(self, state):
        """加载自适应选择器的状态"""
        # 检查算法类型兼容性
        if 'algorithm_type' in state:
            if state['algorithm_type'] != self.selector_algorithm:
                print(f"Warning: Loading {state['algorithm_type']} selector state into {self.selector_algorithm} selector")
                print("This may cause compatibility issues. Consider using the same algorithm type.")
        
        self.adaptive_group_selector.load_state(state)
    
    def _freeze_sender_weights(self):
        """
        冻结发送端权重，保持接收端权重可训练
        发送端：feature_extractor, feature_encoder, gamma_mu, GVQ
        接收端：feature_reconstruction, noise_adaptive_decoder, classifier
        """
        if self.encoder_weights_frozen:
            return
            
        print(f"\n=== Freezing Sender Components for Adaptive Selector Training ===")
        
        # 冻结特征提取器（发送端）
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # 冻结特征编码器（发送端）
        for param in self.feature_encoder.parameters():
            param.requires_grad = False
        for param in self.gamma_mu.parameters():
            param.requires_grad = False    
        # 冻结GVQ码本（发送端）
        for param in self.GVQ.parameters():
            param.requires_grad = False
        
        # 保持接收端组件可训练
        # feature_reconstruction, noise_adaptive_decoder, classifier
        # 这些组件的requires_grad保持为True
        
        self.encoder_weights_frozen = True
        
        # 打印冻结的参数统计
        frozen_params = sum(1 for param in self.parameters() if not param.requires_grad)
        total_params = sum(1 for param in self.parameters())
        trainable_params = total_params - frozen_params
        print(f"Frozen {frozen_params}/{total_params} parameters")
        print(f"Remaining trainable parameters: {trainable_params}")
        print("Receiver components (feature_reconstruction, noise_adaptive_decoder, classifier) remain trainable")
        print("=" * 80)
    
    def get_default_group_to_use(self, psnr_value):
        """
        获取预训练阶段的默认group_to_use选择策略
        
        Args:
            psnr_value: 当前PSNR值
            
        Returns:
            int: 默认的group_to_use值
        """

        return 8
    
    def is_selector_training_phase(self, epoch):
        """
        判断当前是否处于选择器训练阶段
        
        Args:
            epoch: 当前训练epoch
            
        Returns:
            bool: 是否在选择器训练阶段
        """
        return epoch >= self.encoder_pretrain_epochs
    
    def is_selector_update_phase(self, epoch):
        """
        判断当前是否处于选择器训练阶段
        
        Args:
            epoch: 当前训练epoch
            
        Returns:
            bool: 是否在选择器更新阶段
        """
        return epoch >= self.encoder_pretrain_epochs  and epoch < self.encoder_select_epochs  
    
    def is_final_training_phase(self, epoch):
        """
        判断当前是否处于最终训练阶段
        
        Args:
            epoch: 当前训练epoch
            
        Returns:
            bool: 是否在最终训练阶段
        """
        return epoch >= self.encoder_select_epochs
    
    def start_final_training(self, epoch):
        """
        开始最终训练阶段：
        1. 关闭选择器的训练和探索
        2. 解冻编码器权重以便微调
        3. 设置选择器为纯推理模式
        
        Args:
            epoch: 当前训练epoch
        """
        if epoch >= self.encoder_select_epochs and not hasattr(self, 'final_training_started'):
            print("\n" + "=" * 80)
            print("STARTING FINAL TRAINING PHASE")
            print("=" * 80)
            
            # 1. 根据算法类型设置推理模式
            if self.selector_algorithm == 'rl':
                # RL算法：设置推理模式并完成训练
                if hasattr(self.adaptive_group_selector, 'set_inference_mode'):
                    self.adaptive_group_selector.set_inference_mode(True)
                    print("✓ RL Adaptive selector set to inference mode (no more training/exploration)")
                if hasattr(self.adaptive_group_selector, 'finalize_training'):
                    strategy_summary = self.adaptive_group_selector.finalize_training()
                    print("✓ RL selector training finalized with learned strategy")
            else:
                # 启发式算法：完成训练
                if hasattr(self.adaptive_group_selector, 'finalize_training'):
                    strategy_summary = self.adaptive_group_selector.finalize_training()
                    print("✓ Heuristic selector training finalized with monotonicity constraints")
            
            # 2. 解冻编码器权重以便微调
            self._unfreeze_encoder_weights()
            
            # 3. 标记最终训练已开始
            self.final_training_started = True
            
            # 4. 显示最终策略
            final_strategy = self.get_group_strategy_summary()
            print("\n✓ Final group selection strategy:")
            for psnr, group in final_strategy[:5]:  # 显示前5个作为示例
                print(f"   PSNR {psnr:.1f}dB -> {group} groups")
            if len(final_strategy) > 5:
                print(f"   ... and {len(final_strategy)-5} more PSNR levels")
            
            print("✓ Final training phase initialized")
            print("✓ Encoder weights unfrozen for fine-tuning")
            print("✓ Adaptive selector locked to trained strategy")
            print("=" * 80)
    
    def _unfreeze_encoder_weights(self):
        """
        解冻编码器权重以便在最终训练阶段进行微调
        """
        print("\nUnfreezing encoder weights for final training...")
        
        # 冻结特征提取器（发送端）
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
            
        # 冻结特征编码器（发送端）
        for param in self.feature_encoder.parameters():
            param.requires_grad = True
            
        # 冻结GVQ码本（发送端）
        for param in self.GVQ.parameters():
            param.requires_grad = True
        
        # 重置冻结标志
        self.encoder_weights_frozen = False
        
        # 打印解冻的参数统计
        trainable_params = sum(1 for param in self.parameters() if param.requires_grad)
        total_params = sum(1 for param in self.parameters())
        print(f"Unfrozen encoder weights: {trainable_params}/{total_params} parameters now trainable")
        print("All network components are now available for fine-tuning")
    
    def get_selector_algorithm_info(self):
        """获取当前使用的选择器算法信息"""
        info = {
            'algorithm': self.selector_algorithm,
            'psnr_range': (self.adaptive_group_selector.psnr_min, self.adaptive_group_selector.psnr_max),
            'psnr_bins': self.adaptive_group_selector.psnr_bins,
        }
        
        # 添加算法特定信息
        if self.selector_algorithm == 'rl':
            info.update({
                'epsilon': getattr(self.adaptive_group_selector, 'epsilon', 'N/A'),
                'memory_size': len(getattr(self.adaptive_group_selector, 'memory', [])),
                'total_updates': getattr(self.adaptive_group_selector, 'total_updates', 0),
                'inference_mode': getattr(self.adaptive_group_selector, 'inference_mode', False),
                'learning_rate': self.adaptive_group_selector.optimizer.param_groups[0]['lr'] if hasattr(self.adaptive_group_selector, 'optimizer') else 'N/A'
            })
        else:
            info.update({
                'exploration_rate': getattr(self.adaptive_group_selector, 'exploration_rate', 'N/A'),
                'total_updates': getattr(self.adaptive_group_selector, 'total_updates', 0),
                'strategy_changes': getattr(self.adaptive_group_selector, 'strategy_changes', 0),
                'accuracy_threshold': getattr(self.adaptive_group_selector, 'accuracy_threshold', 'N/A')
            })
        
        return info
    
    def set_selector_training_mode(self, training_mode: bool):
        """设置选择器的训练模式"""
        if self.selector_algorithm == 'rl':
            if hasattr(self.adaptive_group_selector, 'set_inference_mode'):
                self.adaptive_group_selector.set_inference_mode(not training_mode)
                mode_str = "training" if training_mode else "inference"
                print(f"RL selector set to {mode_str} mode")
        else:
            # 启发式算法的训练模式通过is_training参数控制
            print(f"Heuristic selector training mode: {training_mode}")
    
    def get_current_group_strategy(self):
        """获取当前的组选择策略（用于调试和监控）"""
        return {
            'algorithm': self.selector_algorithm,
            'strategy': self.adaptive_group_selector.group_choices.copy(),
            'psnr_values': [
                self.adaptive_group_selector.psnr_min + i * 
                (self.adaptive_group_selector.psnr_max - self.adaptive_group_selector.psnr_min) / 
                (self.adaptive_group_selector.psnr_bins - 1)
                for i in range(self.adaptive_group_selector.psnr_bins)
            ]
        }
