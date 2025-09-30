# Copyright (c) Meta Platforms, Inc. and affiliates."""

# All rights reserved.Quantizer Module for Vibration Signal Classification

#

# This source code is licensed under the license found in theThis module contains the vector quantization components:

# LICENSE file in the root directory of this source tree.- Wrapped GroupVectorQuantizer optimized for vibration signals

"""残差向量量化器实现。"""- Model state management for independent deployment

import math"""

import typing as tp

from dataclasses import dataclassimport torch

from dataclasses import fieldimport torch.nn as nn

import sys

import torchimport os

from torch import nnsys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.core_vq import ResidualVectorQuantizationfrom models.vq import ResidualVectorQuantizer



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass

class QuantizedResult:class VibrationVectorQuantizer(nn.Module):

    """量化过程的结果。"""    """

    quantized: torch.Tensor    Specialized vector quantizer for vibration signal features.

    codes: torch.Tensor

    bandwidth: torch.Tensor  # 每批次项使用的带宽（kb/s）    This module wraps the GroupVectorQuantizer with vibration-specific

    penalty: tp.Optional[torch.Tensor] = None    configurations and provides unified interface for quantization operations.

    metrics: dict = field(default_factory=dict)

    Args:

        dimension (int): Feature dimension for quantization

class ResidualVectorQuantizer(nn.Module):        n_codes_group (int): Number of quantization groups

    """残差向量量化器包装类。        bins (int): Codebook size per group

        n_q (int): Total number of quantizers across all groups

    此类为core_vq.py中的ResidualVectorQuantization实现提供了简化的接口。    """

    def __init__(self, args):

    参数:        super().__init__()

        dimension (int): 输入向量的维度。

        codebook_dim (int, optional): 码本条目的维度。        self.dimension = getattr(args, 'quantizer_dimension', 128)

        n_q (int): 使用的残差向量量化器数量。        self.n_codes_group = getattr(args, 'quantizer_n_codes_group', 8)

        bins (int): 码本大小。        self.bins = getattr(args, 'quantizer_bins', 16)

    """        self.n_q = getattr(args, 'quantizer_n_q', 32)



    def __init__(        # Initialize the core vector quantizer

            self,        self.gvq = ResidualVectorQuantizer(

            dimension: int=256,            dimension=self.dimension,

            codebook_dim: tp.Optional[int]=None,            n_q=self.n_q,

            n_q: int=8,            bins=self.bins

            bins: int=1024):        )

        super().__init__()

        self.n_q = n_q    def forward(self, features):

        self.dimension = dimension        """

        self.bins = bins        Forward pass through vector quantization.

        self.codebook_dim = codebook_dim

        Args:

        # 初始化核心RVQ实现            features: Input features to be quantized [batch_size, feature_dim]

        self.vq = ResidualVectorQuantization(

            dim=self.dimension,        Returns:

            codebook_dim=self.codebook_dim,            tuple: (quantized_features, indices, commitment_loss)

            codebook_size=self.bins,                quantized_features: Quantized feature representations

            num_quantizers=self.n_q)                indices: Quantization indices for each group

                commitment_loss: VQ commitment loss for training

    def forward(self, x: torch.Tensor):        """

        """对给定的输入张量执行残差向量量化。        # Get quantized features and loss

        quantized_st, quantized, loss = self.gvq(features)

        参数:

            x (torch.Tensor): 形状为[batch_size, dim]或[batch_size, dim, n_samples]的输入张量。        # Get quantization indices

        indices = self.gvq.encode(features)

        返回:

            tuple: (quantized_st, quantized, loss)        return quantized_st, indices, loss

                - quantized_st: 具有直通梯度的量化张量

                - quantized: 量化张量    def encode(self, features):

                - loss: 量化损失        """

        """        Encode features to quantization indices.

        # 检查输入是1D还是3D

        is_1d = len(x.shape) == 2        Args:

        if is_1d:            features: Input features [batch_size, feature_dim]

            # 将1D向量转换为3D形状以与现有实现兼容

            x = x.unsqueeze(-1)        Returns:

            tuple: (quantized_features, indices, commitment_loss)

        quantized_st, quantized, loss = self.vq(x, n_q=self.n_q)        """

        return self.forward(features)

        # 如果输入是1D的，将输出转换回1D

        if is_1d:    def decode(self, indices):

            quantized_st = quantized_st.squeeze(-1)        """

            quantized = quantized.squeeze(-1)        Decode quantization indices back to features.



        return quantized_st, quantized, loss        Args:

            indices: Quantization indices [batch_size, n_groups, group_size]

    def encode(self, x: torch.Tensor, st: tp.Optional[int]=None) -> torch.Tensor:

        """将给定的输入张量编码为码本索引。        Returns:

            torch.Tensor: Reconstructed features from indices

        参数:        """

            x (torch.Tensor): 形状为[batch_size, dim]或[batch_size, dim, n_samples]的输入张量。        return self.gvq.decode(indices)

            st (int, optional): 起始量化器索引。

    def get_codebook_usage(self):

        返回:        """

            torch.Tensor: 如果输入是1D[batch_size, dim]，则返回形状为[n_q, batch_size]的码本索引        Get statistics about codebook usage for analysis.

                        如果输入是3D[batch_size, dim, n_samples]，则返回形状为[n_q, batch_size, n_samples]的码本索引

        """        Returns:

        st = st or 0            dict: Codebook usage statistics

        is_1d = len(x.shape) == 2        """

        if hasattr(self.gvq, 'get_codebook_usage'):

        # 对于1D向量，转换为3D格式            return self.gvq.get_codebook_usage()

        if is_1d:        else:

            x = x.unsqueeze(-1)  # [batch_size, dim] -> [batch_size, dim, 1]            return {"message": "Codebook usage tracking not available"}



        # 编码向量    def reset_codebook_usage(self):

        codes = self.vq.encode(x, n_q=self.n_q, st=st)        """Reset codebook usage statistics."""

        if hasattr(self.gvq, 'reset_codebook_usage'):

        # 对于1D向量输入，调整输出形状            self.gvq.reset_codebook_usage()

        if is_1d:

            # 从[n_q, batch_size, 1]转换为[n_q, batch_size]    def save_state(self, filepath):

            codes = codes.squeeze(-1)        """

            # 转换为[batch_size, n_q]形式，符合请求的输出        Save quantizer state to file.

            codes = codes.transpose(0, 1)

        Args:

        return codes            filepath: Path to save the quantizer state

        """

    def decode(self, codes: torch.Tensor) -> torch.Tensor:        state = {

        """将给定的代码解码为量化表示。            'gvq_state': self.gvq.state_dict(),

            'config': {

        参数:                'dimension': self.dimension,

            codes (torch.Tensor): 如果是1D输入场景，形状为[batch_size, n_q]或[n_q, batch_size]的码本索引                'n_codes_group': self.n_codes_group,

                                如果是3D输入场景，形状为[n_q, batch_size, n_samples]的码本索引                'bins': self.bins,

                'n_q': self.n_q

        返回:            }

            torch.Tensor: 重构后的向量，形状与输入时一致        }

        """        torch.save(state, filepath)

        # 确定输入类型        print(f"Quantizer state saved to {filepath}")

        is_batch_first = len(codes.shape) == 2 and codes.shape[0] != self.n_q

    def load_state(self, filepath):

        # 如果输入是[batch_size, n_q]格式，转换为[n_q, batch_size]        """

        if is_batch_first:        Load quantizer state from file.

            codes = codes.transpose(0, 1)  # [batch_size, n_q] -> [n_q, batch_size]

        Args:

        # 如果是1D向量场景（无n_samples维度）            filepath: Path to load the quantizer state from

        is_1d = len(codes.shape) == 2        """

        if is_1d:        state = torch.load(filepath, map_location=device)

            # 添加一个虚拟的样本维度 [n_q, batch_size] -> [n_q, batch_size, 1]

            codes = codes.unsqueeze(-1)        # Verify configuration compatibility

        config = state.get('config', {})

        # 解码        if config:

        quantized = self.vq.decode(codes)            assert config['dimension'] == self.dimension, f"Dimension mismatch: {config['dimension']} vs {self.dimension}"

            assert config['n_codes_group'] == self.n_codes_group, f"Groups mismatch: {config['n_codes_group']} vs {self.n_codes_group}"

        # 如果是1D向量场景，移除样本维度            assert config['bins'] == self.bins, f"Bins mismatch: {config['bins']} vs {self.bins}"

        if is_1d:            assert config['n_q'] == self.n_q, f"Quantizers mismatch: {config['n_q']} vs {self.n_q}"

            quantized = quantized.squeeze(-1)

        self.gvq.load_state_dict(state['gvq_state'])

        return quantized        print(f"Quantizer state loaded from {filepath}")



    def get_config(self):

class GroupVectorQuantizer(nn.Module):        """

    """组向量量化器。        Get quantizer configuration.



    将输入特征分成组并对每组应用独立的向量量化。        Returns:

    这种方法使高维特征能够更有效地表示，并且支持任意分组数量。            dict: Configuration parameters

        """

    参数:        return {

        dimension (int): 输入特征维度。            'dimension': self.dimension,

        codebook_dim (int, optional): 码本条目维度。            'n_codes_group': self.n_codes_group,

        n_q (int): 在组间分配的量化器总数。            'bins': self.bins,

        bins (int): 每个量化器的码本大小。            'n_q': self.n_q

        n_codes_group (int): 使用的特征组数量。        }
    """
    def __init__(self,
                dimension: int=256,
                codebook_dim: tp.Optional[int]=None,
                n_q: int=8,
                bins: int=256,
                n_codes_group: int=2):
        super(GroupVectorQuantizer, self).__init__()

        # 验证维度
        if dimension % n_codes_group != 0:
            raise ValueError(f"输入维度 {dimension} 必须能被分组数 {n_codes_group} 整除")

        if codebook_dim is not None and codebook_dim % n_codes_group != 0:
            raise ValueError(f"码本维度 {codebook_dim} 必须能被分组数 {n_codes_group} 整除")

        self.n_codes_group = n_codes_group
        self.dimension = dimension
        self.feature_dim_per_group = dimension // n_codes_group

        # 在组间分配码本维度
        if codebook_dim is not None:
            self.codebook_dim_per_group = codebook_dim // n_codes_group
        else:
            self.codebook_dim_per_group = None

        # 损失权重参数
        self.codebook_loss_lambda = 1.0
        self.commitment_loss_lambda = 0.2

        # 在组间分配量化器
        if n_q % n_codes_group == 0:
            # 均匀分配
            quantizers_per_group = [n_q // n_codes_group] * n_codes_group
        else:
            # 处理不均匀分配
            quantizers_per_group = [n_q // n_codes_group] * n_codes_group
            for i in range(n_q % n_codes_group):
                quantizers_per_group[i] += 1

        self.quantizers_per_group = quantizers_per_group

        # 为每个组创建量化器
        self.quantizers = nn.ModuleList([
            ResidualVectorQuantizer(
                dimension=self.feature_dim_per_group,
                codebook_dim=self.codebook_dim_per_group,
                n_q=quantizers_per_group[i],
                bins=bins
            ) for i in range(self.n_codes_group)
        ])

    def forward(self, x):
        """对输入张量应用分组向量量化。

        参数:
            x (torch.Tensor): 形状为[batch_size, dimension]或[batch_size, dimension, n_samples]的输入张量

        返回:
            tuple: (quantized_st, indices, loss)
                - quantized_st: 具有直通梯度的量化张量
                - indices: 量化索引
                - loss: 所有组的总量化损失
        """
        # 检查输入是1D还是3D
        is_1d = len(x.shape) == 2
        if is_1d:
            # 将1D向量转换为3D形状以与现有实现兼容
            x = x.unsqueeze(-1)

        # 沿特征维度拆分输入
        feature_chunks = torch.split(x, self.feature_dim_per_group, dim=1)

        # 使用nn.ModuleList并行处理每个组
        z_q_st_chunks = []
        z_q_chunks = []
        indices_list = []
        total_loss = 0.0

        for i, (chunk, quantizer) in enumerate(zip(feature_chunks, self.quantizers)):
            # 对每个块应用量化
            z_q_st, z_q, loss = quantizer(chunk)

            # 获取该组的量化索引
            chunk_indices = quantizer.encode(chunk)

            # 收集结果
            z_q_st_chunks.append(z_q_st)
            z_q_chunks.append(z_q)
            indices_list.append(chunk_indices)
            total_loss += loss

        # 沿特征维度连接量化结果
        z_q_st = torch.cat(z_q_st_chunks, dim=1)
        # 堆叠所有编码后的索引
        if is_1d:
            indices = torch.stack(indices_list, dim=1).squeeze(-1).transpose(2,0)# [batch_size, n_groups, n_q_per_group]
        else:
            # 对于3D场景，使用原始堆叠方法 -> [n_groups, n_quantizers, batch_size, n_samples]
            indices = torch.stack(indices_list)

        # 如果输入是1D的，将输出转换回1D
        if is_1d:
            z_q_st = z_q_st.squeeze(-1)
        return z_q_st, indices, total_loss

    def encode(self, x):
        """将输入张量编码为码本索引。

        参数:
            x (torch.Tensor): 形状为[batch_size, dimension]或[batch_size, dimension, n_samples]的输入张量

        返回:
            torch.Tensor:
                - 对于1D输入: 形状为[batch_size, n_groups, n_q_per_group]的索引
                - 对于3D输入: 形状为[n_groups, n_quantizers, batch_size, n_samples]的索引
        """
        # 检查输入是1D还是3D
        is_1d = len(x.shape) == 2
        if is_1d:
            # 将1D向量转换为3D形状以与现有实现兼容
            x = x.unsqueeze(-1)

        # 沿特征维度拆分输入
        feature_chunks = torch.split(x, self.feature_dim_per_group, dim=1)
        indices_list = []

        for i, (chunk, quantizer) in enumerate(zip(feature_chunks, self.quantizers)):
            chunk_indices = quantizer.encode(chunk)
            indices_list.append(chunk_indices)

        if is_1d:
            # 对于1D场景，调整为[batch_size, n_groups, n_quantizers_per_group]
            batch_indices = []
            for group_idx, group_indices in enumerate(indices_list):
                batch_indices.append(group_indices)  # 已经是[batch_size, n_q]形状

            # 堆叠所有组的索引 -> [batch_size, n_groups, max_n_q_per_group]
            indices = torch.stack(batch_indices, dim=1)
        else:
            # 对于3D场景，使用原始堆叠方法 -> [n_groups, n_quantizers, batch_size, n_samples]
            indices = torch.stack(indices_list)

        return indices

    def decode(self, indices):
        """将索引解码回特征向量。

        参数:
            indices (torch.Tensor):
                - 对于1D输入: 形状为[batch_size, n_groups, n_q_per_group]或包含少于n_groups组的索引
                - 对于3D输入: 形状为[n_groups, n_quantizers, batch_size, n_samples]或包含少于n_groups组的索引

        返回:
            torch.Tensor: 重构的特征张量，形状与输入时一致。缺失的组将被设为0值。
        """
        # 确定输入类型 - 更可靠的检测方法
        # 1D: [batch_size, n_groups, n_q_per_group] - 第三维通常很小(4)
        # 3D: [n_groups, n_quantizers, batch_size, n_samples] - 四维
        is_1d = len(indices.shape) == 3

        if is_1d:
            # 1D情况: [batch_size, n_groups, n_q_per_group]，n_groups可能小于self.n_codes_group
            decoded_chunks = []

            # 获取批次大小和输入的组数
            batch_size = indices.shape[0]
            input_n_groups = indices.shape[1]

            # 对每个组进行解码，最多处理实际配置的组数
            for i in range(min(input_n_groups, self.n_codes_group)):
                group_indices = indices[:, i, :]
                decoded = self.quantizers[i].decode(group_indices)
                decoded_chunks.append(decoded)

            # 为缺失的组添加零填充
            for i in range(input_n_groups, self.n_codes_group):
                # 创建与其他解码块形状匹配的零张量
                zero_chunk = torch.zeros(batch_size, self.feature_dim_per_group, device=indices.device)
                decoded_chunks.append(zero_chunk)

            # 沿特征维度连接解码后的块
            decoded = torch.cat(decoded_chunks, dim=1)
        else:
            # 3D情况: [n_groups, n_quantizers, batch_size, n_samples]
            decoded_chunks = []

            # 处理输入的所有组
            input_n_groups = indices.shape[0]

            # 确定batch_size和sample_size以便创建零填充
            _, _, batch_size, n_samples = indices[0].shape

            # 对每个提供的组进行解码
            for i in range(min(input_n_groups, self.n_codes_group)):
                decoded = self.quantizers[i].decode(indices[i])
                decoded_chunks.append(decoded)

            # 为缺失的组添加零填充
            for i in range(input_n_groups, self.n_codes_group):
                # 创建与其他解码块形状匹配的零张量
                zero_chunk = torch.zeros(batch_size, self.feature_dim_per_group, n_samples,
                                        device=indices.device)
                decoded_chunks.append(zero_chunk)

            # 沿特征维度连接解码后的块
            decoded = torch.cat(decoded_chunks, dim=1)

        return decoded

    def get_output_from_indices(self, indices):
        """与decode相同，但保持与网络.py中的调用一致。

        参数:
            indices (torch.Tensor):
                - 对于1D输入: 形状为[batch_size, n_groups, n_q_per_group]的索引
                - 对于3D输入: 形状为[n_groups, n_quantizers, batch_size, n_samples]的索引

        返回:
            torch.Tensor: 重构的特征张量，形状为[batch_size, dimension]或[batch_size, dimension, n_samples]
        """
        return self.decode(indices)