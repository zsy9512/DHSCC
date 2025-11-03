"""
Sender Network Module for Vibration Signal Classification

This module contains the sender-side components including:
- Feature extraction
- Feature encoding
- Importance estimation
- Pre-classification for task-aware importance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizer import VibrationVectorQuantizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImportanceLayer(nn.Module):
    """
    Implementation of a layer for feature importance estimation.

    This layer computes importance scores for features based on channel noise,
    ensuring weights are positive for numerical stability.

    Args:
        input_channel: Input dimension
        output_channel: Output dimension
    """
    def __init__(self, input_channel, output_channel):
        super(ImportanceLayer, self).__init__()
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

class ImportanceEstimator(nn.Module):
    """
    Multi-layer MLP implementation for mapping channel state to feature importance.

    This implements the feature importance function γ(σ²) from the paper,
    which maps channel noise level to importance scores for each feature.

    Args:
        intermediate_dim: Dimension of the final feature importance scores
    """
    def __init__(self, intermediate_dim=64):
        super(ImportanceEstimator, self).__init__()
        # 4-layer MLP as described in the paper section IV-B
        self.f1 = ImportanceLayer(1, 16)
        self.f2 = ImportanceLayer(16, 16)
        self.f3 = ImportanceLayer(16, 16)
        self.f4 = ImportanceLayer(16, intermediate_dim)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        return x

class CapsuleNetwork(nn.Module):
    """
    Capsule network module for grouping input data into 8 groups of 16 dimensions each.
    """
    def __init__(self, input_dim, num_groups=8, group_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.num_groups = num_groups
        self.group_dim = group_dim
        self.output_dim = num_groups * group_dim

        # Input mapping layer
        self.input_projection = nn.Linear(input_dim, self.output_dim)

        # Capsule network layers
        self.primary_capsules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(group_dim, group_dim),
                nn.ReLU(),
                nn.Linear(group_dim, group_dim),
                nn.LayerNorm(group_dim)
            ) for _ in range(num_groups)
        ])

        # Dynamic routing parameters
        self.routing_weights = nn.Parameter(torch.randn(num_groups, group_dim, group_dim))

    def squash(self, tensor):
        """Capsule activation function"""
        squared_norm = (tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)

    def forward(self, x):
        batch_size = x.size(0)

        # Project to target dimension
        x_proj = self.input_projection(x)  # [batch, num_groups * group_dim]

        # Reshape to capsule format
        x_grouped = x_proj.view(batch_size, self.num_groups, self.group_dim)

        # Process each group through primary capsules
        capsule_outputs = []
        for i, capsule in enumerate(self.primary_capsules):
            group_output = capsule(x_grouped[:, i, :])  # [batch, group_dim]
            capsule_outputs.append(group_output)

        # Stack capsule outputs
        capsule_features = torch.stack(capsule_outputs, dim=1)  # [batch, num_groups, group_dim]

        # Apply squash activation
        capsule_features = self.squash(capsule_features)

        return capsule_features

class JointImportanceEstimator_group(nn.Module):
    """
    Simplified joint importance estimator based on capsule network grouping.

    Args:
        feature_dim: Feature dimension, default 128
        num_groups: Number of capsule network groups, default 8
        group_dim: Dimension per group, default 16
    """
    def __init__(self, feature_dim=128, num_groups=8, group_dim=16, use_frequency_features=False, num_classes=10, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_groups = num_groups
        self.group_dim = group_dim
        self.use_frequency_features = use_frequency_features
        self.num_classes = num_classes

        assert feature_dim == num_groups * group_dim, \
            f"feature_dim ({feature_dim}) must equal num_groups * group_dim ({num_groups} * {group_dim})"

        # Group capsule weights
        self.W = nn.Parameter(torch.randn(num_groups, group_dim, group_dim))
        # Self-attention scaling
        self.attn_scale = group_dim ** -0.5
        # Gating fusion MLP
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
        New process:
        1. Feature grouping, map to primary capsules
        2. Dynamic routing twice to get group capsules
        3. Self-attention fusion
        4. Norm fusion weighting, softmax normalization
        5. New method: fuse channel reliability and channel noise to guide feature importance generation
        """
        B = features.size(0)
        h = features.view(B, self.num_groups, self.group_dim)  # [B, num_groups, group_dim]
        # Primary capsules
        u = torch.einsum('bnd,ndh->bnh', h, self.W)            # [B, num_groups, group_dim]
        u = self.squash(u)
        # Dynamic routing twice
        v_i, v_global = self.dynamic_routing(u, num_iter=2)
        # Self-attention
        v_hat = self.self_attention(v_i)
        # Norms (existence probabilities)
        p = v_i.norm(dim=-1)                                  # [B, num_groups]
        q = v_hat.norm(dim=-1)                                # [B, num_groups]
        # Gating fusion
        gate = self.gate_mlp(torch.stack([p, q], dim=-1)).squeeze(-1)  # [B, num_groups]
        alpha_logits = gate * p + (1 - gate) * q
        alpha = F.softmax(alpha_logits, dim=1)                # [B, num_groups]
        final_importance = self.expand_group_importance_to_feature_dim(alpha)

        # Channel reliability guided fusion
        fusion_importance = None
        if use_channel_guided and channel_importance is not None and channel_noise is not None:
            # channel_importance: [B, feature_dim] or [1, feature_dim] or [feature_dim]
            # Auto-expand to batch dimension
            if channel_importance.dim() == 1 and channel_importance.size(0) == self.feature_dim:
                channel_importance = channel_importance.unsqueeze(0).expand(B, -1)
            elif channel_importance.dim() == 2 and channel_importance.size(0) == 1 and channel_importance.size(1) == self.feature_dim:
                channel_importance = channel_importance.expand(B, -1)
            elif channel_importance.dim() == 2 and channel_importance.size(0) == B and channel_importance.size(1) == self.feature_dim:
                pass  # already correct
            else:
                raise RuntimeError(f"channel_importance shape {channel_importance.shape} does not match [B, feature_dim]")

            channel_group = channel_importance.view(B, self.num_groups, self.group_dim).mean(dim=-1)
            fusion_logits = torch.exp(alpha_logits)/(torch.exp(-channel_noise)+1e-6)

            fusion_alpha = F.softmax(fusion_logits, dim=1)  # [B, num_groups]
            fusion_importance = self.expand_group_importance_to_feature_dim(fusion_alpha)

        # Return three importance estimates for decoupled evaluation
        return final_importance, fusion_importance, channel_importance

class SenderFeatureExtractor(nn.Module):
    """
    Feature extraction network for vibration signals at sender side.

    This module combines convolutional layers for spatial feature extraction
    optimized for vibration signal processing.

    Args:
        input_dim: Input channel dimension (usually 1 for raw signals)
        output_dim: Output feature dimension
        use_ssm: Whether to use SSM blocks for temporal modeling
    """
    def __init__(self, input_dim=1, output_dim=512, use_ssm=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_ssm = use_ssm

        # Initial feature extraction
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
        # Residual block for layer 1
        self.layer1_res = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Layer 2 with downsampling
        self.layer2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Layer 3 with downsampling
        self.layer3 = nn.Sequential(
            nn.Conv1d(256, output_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Residual block for layer 3
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

        # Layer 1 processing
        x_layer1 = self.layer1(x)

        # Apply residual connection for layer 1
        res = self.layer1_res(x_layer1)
        if res.size() != x_layer1.size():
            res = F.adaptive_avg_pool1d(res, x_layer1.size(2))
        x = res + x_layer1
        x = self.layer2(x)

        # Layer 3 processing
        x = self.layer3(x)

        return x

    def apply_residual(self, x):
        """
        Apply residual connection for the output features.
        Used by receiver for residual connections.

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

class SenderFeatureEncoder(nn.Module):
    """
    Feature encoder module for compressing high-dimensional features to low-dimensional representation.

    This module implements the encoding function φ(s) from the paper, including:
    1. Convolutional dimension reduction
    2. Fixed dimension pooling
    3. Linear projection and normalization
    4. Feature sparsification and weight normalization

    Args:
        input_channels (int): Input channel dimension
        hidden_dim (int): Hidden feature dimension
        output_dim (int): Output feature dimension
    """
    def __init__(self, input_channels=512, hidden_dim=64, output_dim=64):
        super().__init__()

        # Convolutional dimension reduction encoder
        self.conv_encoder = nn.Sequential(
            nn.Conv1d(input_channels, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )

        # Fixed dimension pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16)

        # Calculate flattened feature dimension
        self.flat_dim = 8 * 16  # 8 channels x 16 time steps

        # Linear projection and normalization
        self.feature_projector = nn.Sequential(
            nn.Linear(self.flat_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Feature sparsification parameters - key component
        self.sparsifier_weight = nn.Parameter(torch.Tensor(output_dim, hidden_dim))
        self.sparsifier_bias = nn.Parameter(torch.Tensor(output_dim))

        # Parameter initialization
        self.sparsifier_weight.data.normal_(0, 0.5)
        self.sparsifier_bias.data.normal_(0, 0.1)

    def forward(self, x):
        """
        Encoder forward propagation.

        Args:
            x: Input features [batch_size, channels, time]

        Returns:
            tuple: (features, weights, bias) containing:
                features: Features after linear projection
                weights: Normalized weight parameters
                bias: Normalized bias parameters
        """
        batch_size = x.size(0)

        # Stage 1: Convolutional dimension reduction
        x = self.conv_encoder(x)
        x = self.adaptive_pool(x)
        x = x.view(batch_size, -1)
        x = self.feature_projector(x)

        # Feature normalization
        x_norm = torch.norm(x, dim=1, keepdim=True) + 1e-6
        x = x.size(-1) * x / x_norm  # Normalize feature vector length

        # Stage 4: Prepare sparsification weights and bias
        weight = F.tanh(self.sparsifier_weight)
        bias = F.tanh(self.sparsifier_bias)

        # Ensure weights are non-zero for stability
        weight = torch.clamp(torch.abs(weight), min=1e-3) * torch.sign(weight.detach())
        bias = torch.clamp(torch.abs(bias), min=1e-3) * torch.sign(bias.detach())

        # L2 normalization of weights
        l2_norm = (torch.sum(weight.pow(2), dim=1, keepdim=True) +
                   bias.pow(2).unsqueeze(1)).sqrt() + 1e-6
        weight = weight / l2_norm.transpose(0, 1)
        bias = bias / l2_norm.squeeze()

        # Apply linear transformation
        x = F.linear(x, weight, bias)

        return x, weight, bias

class SenderNet(nn.Module):
    """
    Complete sender network integrating all sender-side components.

    This module combines feature extraction, encoding, and importance estimation
    to prepare features for transmission over wireless channels.

    Args:
        args: Configuration parameters containing model settings
    """
    def __init__(self, args):
        super().__init__()

        self.hidden_channel = args.intermediate_dim
        self.num_classes = getattr(args, 'num_classes', 10)

        # Feature extraction network
        self.feature_extractor = SenderFeatureExtractor(
            input_dim=1,
            output_dim=512,
            use_ssm=False
        )

        # Feature encoder
        self.feature_encoder = SenderFeatureEncoder(
            input_channels=512,
            hidden_dim=self.hidden_channel,
            output_dim=self.hidden_channel
        )

        # Joint importance estimator
        self.importance_estimator = JointImportanceEstimator_group(
            feature_dim=self.hidden_channel,
            num_groups=8,
            group_dim=16,
        )

        # Pre-classifier for task-aware importance
        self.pre_classifier = nn.Linear(self.hidden_channel, self.num_classes)

    def forward(self, x, channel_noise):
        """
        Forward pass through the sender network.

        Args:
            x: Input vibration signal [batch_size, channels, time]
            channel_noise: Channel noise level [batch_size, 1]

        Returns:
            tuple: (encoded_features, importance_weights, pre_task_logits, feature_shape, original_encoded_features)
                encoded_features: Compressed and weighted features ready for transmission
                importance_weights: Feature importance weights
                pre_task_logits: Pre-classification logits for task-aware importance
                feature_shape: Original feature shape for reconstruction
                original_encoded_features: Raw encoded features before weighting and reordering
        """
        batch_size = x.size(0)
        # Feature extraction
        extracted_features = self.feature_extractor(x)
        
        feature_shape = extracted_features.shape

        # Feature encoding
        encoded_features, weight, bias = self.feature_encoder(extracted_features)
        # Generate pre-task logits for importance estimation
        pre_task_logits = self.pre_classifier(encoded_features.detach())

        # Calculate feature importance using joint estimator
        final_importance, _, _ = self.importance_estimator(encoded_features, pre_task_logits)

        # Ensure non-zero values for stability
        importance_weights = torch.clamp(final_importance*channel_noise, min=1e-4)

        # Sort features by importance
        sorted_importance, sort_indices = torch.sort(importance_weights, dim=-1, descending=True)
        #print(f"encoded_features {encoded_features}")
        #print(f"sort_indices {sort_indices}")
        # Reorder features according to importance
        expanded_indices = sort_indices.expand(batch_size, -1)
        sorted_features = torch.gather(encoded_features, dim=1, index=expanded_indices)

        # Apply importance weighting
        weighted_features = sorted_features * sorted_importance

        # Apply tanh constraint and ensure stable values
        final_features = torch.tanh(weighted_features)
        sign = torch.sign(final_features)
        abs_features = torch.abs(final_features)
        final_features = torch.clamp(abs_features, min=1e-3) * sign
        final_features = torch.clamp(final_features, min=-1.0, max=1.0)

        return final_features, sorted_importance, pre_task_logits, feature_shape, encoded_features

    def save_state(self, filepath):
        """
        Save sender network state to file.

        Args:
            filepath: Path to save the model state
        """
        state = {
            'feature_extractor': self.feature_extractor.state_dict(),
            'feature_encoder': self.feature_encoder.state_dict(),
            'importance_estimator': self.importance_estimator.state_dict(),
            'pre_classifier': self.pre_classifier.state_dict(),
        }
        torch.save(state, filepath)
        print(f"Sender network state saved to {filepath}")

    def load_state(self, filepath):
        """
        Load sender network state from file.

        Args:
            filepath: Path to load the model state from
        """
        state = torch.load(filepath, map_location=device)
        self.feature_extractor.load_state_dict(state['feature_extractor'])
        self.feature_encoder.load_state_dict(state['feature_encoder'])
        self.importance_estimator.load_state_dict(state['importance_estimator'])
        self.pre_classifier.load_state_dict(state['pre_classifier'])
        print(f"Sender network state loaded from {filepath}")