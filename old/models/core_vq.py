# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Core vector quantization implementation."""
import typing as tp

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def default(val: tp.Any, d: tp.Any) -> tp.Any:
    """返回val（如果不为None），否则返回d。"""
    return val if val is not None else d


class VQEmbedding(nn.Module):
    """向量量化嵌入模块。
    
    实现带有直通梯度估计（straight-through estimator）的向量量化层，支持梯度传播。
    """
    def __init__(self, dim, codebook_size, codebook_dim=None):
        """初始化VQEmbedding模块。
        
        参数:
            dim: 输入特征维度
            codebook_size: 码本中的条目数量
            codebook_dim: 码本条目的维度（如未指定则默认为dim）
        """
        super().__init__()
        self.nu = 0.0  # EMA更新的超参数（0.0表示不使用EMA）
        self.codebook_size = codebook_size
        self._codebook_dim = default(codebook_dim, dim)
        
        # 如有需要，将输入特征投影到码本维度
        requires_projection = self._codebook_dim != dim
        self.project_in = (nn.Linear(dim, self._codebook_dim)
                          if requires_projection else nn.Identity())
        self.project_out = (nn.Linear(self._codebook_dim, dim)
                           if requires_projection else nn.Identity())
        
        # 初始化码本
        self.embedding = nn.Embedding(self.codebook_size, self._codebook_dim)
        self.embedding.weight.data.uniform_(-1./self.codebook_size, 1./self.codebook_size)    
    
    def forward(self, input):
        """为输入向量找到最接近的码本条目。
        
        参数:
            input: 形状为[batch_size, dim]或[batch_size, dim, n]的输入张量
            
        返回:
            最接近的码本条目的索引
        """
        # 检查输入是否为1D向量 (batch_size, dim)
        if len(input.shape) == 2:
            # 1D向量情况: [batch_size, dim] -> [batch_size, dim, 1]
            x = input.unsqueeze(-1)
            x = rearrange(x, "b d n -> b n d")  # 转换为 [batch_size, 1, dim]
        else:
            # 原始3D情况: [batch_size, dim, n]
            x = rearrange(input, "b d n -> b n d")
            
        x = self.project_in(x)
        latents = vq(x, self.embedding.weight)
        return latents
        
    def straight_through(self, z_e_x):
        """使用直通梯度估计进行向量量化。
        
        参数:
            z_e_x: 形状为[batch_size, dim]或[batch_size, dim, n]的输入张量
            
        返回:
            (quantized_with_st, quantized)的元组
                quantized_with_st: 具有直通梯度的量化张量
                quantized: 不向编码器传递梯度的量化张量
        """
        # 检查输入是否为1D向量 (batch_size, dim)
        is_1d = len(z_e_x.shape) == 2
        
        if is_1d:
            # 1D向量情况: [batch_size, dim] -> [batch_size, dim, 1]
            z_e_x = z_e_x.unsqueeze(-1)
            
        # 将形状从(b d n)转换为(b n d)以进行量化
        z_e_x_ = rearrange(z_e_x, "b d n -> b n d")
        z_e_x_ = self.project_in(z_e_x_)     
        
        # 应用向量量化
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        
        # 获取码本向量
        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar_ = self.project_out(z_q_x_bar_)
        z_q_x_bar = rearrange(z_q_x_bar_, "b n d -> b d n")
        
        # 应用直通梯度技巧
        z_q_x = z_e_x + (z_q_x_bar - z_e_x).detach() + \
                (self.nu * z_q_x_bar) + (-self.nu * z_q_x_bar).detach()
        
        # 如果输入是1D的，则将输出转回1D形式
        if is_1d:
            z_q_x = z_q_x.squeeze(-1)
            z_q_x_bar = z_q_x_bar.squeeze(-1)
                
        return z_q_x, z_q_x_bar
        
    def compute_loss(self, z_q_x, z_e_x, beta=0.25):
        """计算VQ损失（码本损失和承诺损失）。
        
        参数:
            z_q_x: 量化后的向量
            z_e_x: 量化前的输入向量
            beta: 承诺损失的权重
            
        返回:
            组合的VQ损失
        """
        # 码本损失（将码本向量移向输入）
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # 承诺损失（将输入移向码本向量）
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
        return loss_vq + beta * loss_commit
        
    def decode(self, indices):
        """将索引解码为特征向量。
        
        参数:
            indices: 包含码本索引的张量
            
        返回:
            重构的特征向量
        """
        # 处理索引的形状可能是[batch_size]或[batch_size, n_samples]
        input_shape = indices.shape
        flatten_indices = indices.reshape(-1)
        
        z_q_x_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=flatten_indices)
            
        if len(input_shape) == 1:
            # 对于1D情况 [batch_size]
            z_q_x_ = z_q_x_flatten  # 保持[batch_size, codebook_dim]形状
        else:
            # 对于原始情况 [batch_size, n_samples]
            z_q_x_ = z_q_x_flatten.view([indices.shape[0], indices.shape[1], self._codebook_dim])
            
        z_q_x_ = self.project_out(z_q_x_)
        
        if len(input_shape) == 1:
            # 对于1D情况，保持形状为[batch_size, dim]
            return z_q_x_
        else:
            # 对于原始情况，转换为[batch_size, dim, n_samples]
            z_q_x = rearrange(z_q_x_, "b n d -> b d n")
            return z_q_x


from torch.autograd import Function

class VectorQuantization(Function):
    """寻找最近码本向量的向量量化函数。
    
    此函数不可微分，仅用于推理。
    对于训练，请使用VectorQuantizationStraightThrough。
    """
    
    @staticmethod
    def forward(ctx, inputs, codebook, k=1):
        """为每个输入向量找到最接近的码本条目。
        
        参数:
            inputs: 形状为[batch_size, n_samples, embedding_dim]的输入特征
            codebook: 形状为[codebook_size, embedding_dim]的码本向量
            k: 要考虑的最近向量数量（默认：1，目前仅支持k=1）
            
        返回:
            indices: 最近码本条目的索引
        """
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            
            # 将输入重塑为[batch_size*n_samples, embedding_dim]
            inputs_flatten = inputs.reshape(-1, embedding_size)       
            
            # 计算平方范数以进行高效的距离计算
            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
            
            # 计算平方欧几里得距离：||x-y||^2 = ||x||^2 + ||y||^2 - 2*x·y
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                                   inputs_flatten, codebook.t(), 
                                   alpha=-2.0, beta=1.0)
            
            # 寻找最近的码本条目
            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            
            ctx.mark_non_differentiable(indices)
            return indices

    @staticmethod
    def backward(ctx, grad_output):
        """此函数不可微分。"""
        raise RuntimeError('尝试在包含`VectorQuantization`的图上调用`.grad()`。'
            '函数`VectorQuantization`不可微分。如果您需要梯度的直通估计器，'
            '请使用`VectorQuantizationStraightThrough`。')


class VectorQuantizationStraightThrough(Function):
    """带有直通梯度估计的向量量化。
    
    此函数使用直通估计器技术允许梯度通过量化操作。
    """
    
    @staticmethod
    def forward(ctx, inputs, codebook):
        """应用支持梯度的向量量化，通过直通估计器。
        
        参数:
            inputs: 形状为[batch_size, n_samples, embedding_dim]的输入特征
            codebook: 形状为[codebook_size, embedding_dim]的码本向量
            
        返回:
            codes: 量化后的向量
            indices_flatten: 用于下游处理的扁平化索引
        """
        # 寻找最近邻索引
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        
        # 保存用于反向传播
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)
        
        # 查找对应的码本向量
        codes_flatten = torch.index_select(codebook, dim=0, index=indices_flatten)
        codes = codes_flatten.view_as(inputs)
        
        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices=None):
        """直通估计器用于反向传播。
        
        参数:
            grad_output: 从后续层流入的梯度
            grad_indices: 未使用，Function API需要
            
        返回:
            grad_inputs: 直接传递给输入的梯度
            grad_codebook: 用于更新码本向量的梯度
        """
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # 直通估计器：梯度直接传递给输入
            grad_inputs = grad_output.clone()
            
        if ctx.needs_input_grad[1]:
            # 基于分配的向量更新码本向量
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)
            grad_output_flatten = grad_output.contiguous().view(-1, embedding_size)
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)


# 创建函数接口
vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply

__all__ = ["vq", "vq_st"]


class ResidualVectorQuantization(nn.Module):
    """残差向量量化实现。
    
    遵循https://arxiv.org/pdf/2107.03312.pdf中的算法1。
    使用多个VQ层逐步编码残差，实现更精确的量化。
    """

    def __init__(self, *, dim, codebook_size, codebook_dim=None, num_quantizers=8, shared=False):
        """初始化RVQ模块。
        
        参数:
            dim: 输入特征维度
            codebook_size: 码本中的条目数量
            codebook_dim: 码本条目的维度（如未指定则默认为dim）
            num_quantizers: 量化阶段的数量
            shared: 是否在所有阶段共享相同的码本
        """
        super().__init__() 
        
        # 创建量化层（共享或独立）
        if shared:
            self.emb = VQEmbedding(dim=dim, codebook_size=codebook_size, codebook_dim=codebook_dim)
            self.layers = nn.ModuleList([self.emb for _ in range(num_quantizers)])
        else:     
            self.layers = nn.ModuleList([
                VQEmbedding(dim=dim, codebook_size=codebook_size, codebook_dim=codebook_dim) 
                for _ in range(num_quantizers)
            ])
    
    def forward(self, x, n_q: tp.Optional[int]=None):
        """应用残差向量量化。
        
        参数:
            x: 形状为[batch_size, dim, n_samples]的输入张量
            n_q: 要使用的量化器数量（默认：使用所有可用量化器）
            
        返回:
            quantized_out_st: 具有直通梯度的量化张量
            quantized_out: 量化张量
            loss: 量化损失
        """
        quantized_out = 0.0
        quantized_out_st = 0.0
        loss = 0.0
        residual = x
        
        # 默认：使用所有量化器
        n_q = n_q or len(self.layers)
        
        # 顺序应用每个量化器
        for layer in self.layers[:n_q]:
            z_q_x_st, z_q_x = layer.straight_through(residual)
            
            # 计算此阶段的损失
            res_loss = self.compute_loss(z_q_x, residual)
            loss += res_loss
            
            # 更新残差并累积量化输出
            residual = residual - z_q_x_st
            quantized_out += z_q_x
            quantized_out_st += z_q_x_st
            
        return quantized_out_st, quantized_out, loss
        
    def encode(self, x: torch.Tensor, n_q: tp.Optional[int]=None, st: tp.Optional[int]=None) -> torch.Tensor:
        """将输入张量编码为码本索引。
        
        参数:
            x: 形状为[batch_size, dim, n_samples]的输入张量
            n_q: 要使用的量化器数量
            st: 起始量化器索引
            
        返回:
            out_indices: 所有量化器的堆叠索引
        """
        residual = x
        all_indices = []
        
        # 默认值
        n_q = n_q or len(self.layers)
        st = st or 0
        
        # 使用指定范围的量化器进行编码
        for layer in self.layers[st:n_q]:
            indices = layer(residual)
            z_q_x = layer.decode(indices)
            residual = residual - z_q_x
            all_indices.append(indices)
            
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        """将索引解码回特征向量。
        
        参数:
            q_indices: 来自多个量化器的堆叠索引
            
        返回:
            quantized_out: 重构的特征向量
        """
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        
        # 解码并从每个量化器累积
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
            
        return quantized_out
        
    def compute_loss(self, z_q_x, z_e_x, beta=0.25):
        """使用平滑L1损失计算量化损失。
        
        参数:
            z_q_x: 量化后的向量
            z_e_x: 量化前的输入向量
            beta: 码本损失的权重
            
        返回:
            组合的损失
        """
        # 向量量化损失（将输入向量移向码本）
        loss_vq = F.smooth_l1_loss(z_e_x, z_q_x.detach(), beta=0.1)
        
        # 承诺损失（将码本向量移向输入）
        loss_commit = F.smooth_l1_loss(z_e_x.detach(), z_q_x, beta=0.1)
        
        return loss_vq+ beta * loss_commit