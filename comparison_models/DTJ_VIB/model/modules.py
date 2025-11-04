import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
# 导入DHSCC无线传输模块
# 添加DHSCC路径
dhscc_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if dhscc_root not in sys.path:
    sys.path.insert(0, dhscc_root)
from wireless_comm.interface import WirelessInterface as DTJ_WirelessInterface
from models.quantizer import VibrationVectorQuantizer
print("✓ DHSCC无线通信模块导入成功")

class Resblock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False)
        )
        
    def forward(self, x):
        return x + self.model(x)
    

class Resblock_down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 2, bias=False)
        )
        
    def forward(self, x):
        return self.downsample(x)+self.model(x)
  
class RPQSampler(nn.Module):
    def __init__(self, dim_dic, num_embeddings=16, snr=2.0, **kwargs):
        super().__init__()
        self.quantizer = VibrationVectorQuantizer(
            dimension=256,
            n_codes_group=16,
            bins=num_embeddings,
            n_q=64
        )
        # 初始化无线传输接口
        self.wireless_interface = DTJ_WirelessInterface(comm_mode=3,channel_model='awgn')
        # 训练状态标志
        self.enable_wireless = True
        self.snr_db = snr
        

    def forward(self, X, mod= None,former_shape=None,snr=None   ): 
        X = X.view(-1, 256)  # Flatten the input
        device = X.device
        if self.training:
            # Random SNR sampling during training
            #snr_db = (torch.rand(1, device=device) * (10 + 1e-6) + 5)
            snr_db = (torch.rand(1, device=device) * (5 + 1e-6))
        else:
            # Convert SNR to channel noise for testing
            snr_db = torch.tensor([snr], device=device)
        quantized_features, all_indices, commit_loss = self.quantizer(X)
        transmission_result = self.wireless_interface.transmit_mode3(
            indices=all_indices,
            snr_db=snr_db,
            bits_per_index=4,
            group_to_use=8
        )
        received_indices = transmission_result['received_indices']
        # 反量化
        received_features_decoded = self.quantizer.decode(received_indices)
        # For gradient flow: use quantized_features as base and add detached difference
        # This ensures gradients flow through the quantizer but receiver only sees transmitted+zero data
        received_features = quantized_features + (received_features_decoded - quantized_features).detach()
        received_features = received_features.view(former_shape)  # Reshape back to original dimensions
        return received_features, torch.tensor(0.0, device=device), commit_loss
        
class MaskAttentionSampler(nn.Module):
    def __init__(self, dim_dic, num_embeddings=50, snr=8.0, **kwargs):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim_dic = dim_dic
        self.embedding = nn.Parameter(torch.Tensor(num_embeddings, dim_dic))
        nn.init.uniform_(self.embedding, -1/num_embeddings, 1/num_embeddings)
        # 初始化无线传输接口
        self.wireless_interface = DTJ_WirelessInterface(comm_mode=3, channel_model='awgn')
        # 训练状态标志
        self.enable_wireless = True
        self.snr_db = snr
        
   
    def compute_score(self, X):
        #量化：每个 patch 选出最相似的 codebook 向量（即最大分数的 index），这就是量化的结果。
        #采样：训练时用 Gumbel-Softmax 近似 one-hot，测试时直接 argmax 得到 one-hot。
        return torch.matmul(X, self.embedding.transpose(1,0))/np.sqrt(self.dim_dic)
    
    def compute_distance(self,X):
        m = torch.sum(self.embedding ** 2, dim=1).unsqueeze(0 )\
            + torch.sum(X ** 2, dim=1, keepdim=True) 
        return -torch.addmm(m, X, self.embedding.transpose(1,0),
                           alpha=-2.0, beta=1.0)
    
    def sample(self, score, mod=None,former_shape=None,snr=None):
        dist = F.softmax(score, dim=-1)

        #训练时用 Gumbel-Softmax 近似 one-hot
        if self.training:
            samples = F.gumbel_softmax(
                score, tau=0.5, hard=True
            ) # one_not 
            noise = self.construct_noise(mod, samples)#只传最大值把噪声加到所有值上
            samples = samples + noise
        else:
            samples= torch.argmax(score, dim=-1)
            indices_for_channel = samples.reshape(former_shape[0],1,64)
            transmission_result = self.wireless_interface.transmit_mode3(
                indices=indices_for_channel,
                snr_db=snr,
                bits_per_index=4,
                group_to_use=64
            )
            received_indices = transmission_result['received_indices']
            samples =received_indices.view(samples.shape)
            #samples = self.mod_channel_demod(mod, samples)
            samples = F.one_hot(samples, num_classes=self.num_embeddings).float()
        return samples, dist,None
    
    def mod_channel_demod(self,mod, x):
        X = mod.modulate(x)
        X = mod.awgn(X)
        return mod.demodulate(X).to(self.embedding.device)

    def construct_noise(self, mod, samples):
        x = torch.argmax(samples, dim=-1)
        indices_for_channel = x.reshape(64,1,64)   
        if self.training:
            # Random SNR sampling during training
            #snr_db = (torch.rand(1) * (15 - 5+1e-6) + 5).to("cpu")
            snr_db = (torch.rand(1) * (5+1e-6)).to("cpu")
        else:
            # Convert SNR to channel noise for testing
            snr_db = torch.tensor([self.snr_db])
        transmission_result = self.wireless_interface.transmit_mode3(
            indices=indices_for_channel,
            snr_db=snr_db,
            bits_per_index=4,
            group_to_use=64
        )  
        received_indices = transmission_result['received_indices'] 
        x_tilde =received_indices.view(x.shape)
        #x_tilde = self.mod_channel_demod(mod, x)
        noise = F.one_hot(x_tilde, num_classes=self.num_embeddings).float() -\
                F.one_hot(x, num_classes=self.num_embeddings).float()
        return noise

    def recover(self, samples):
        out = torch.matmul(samples, self.embedding)
        return out
        
    def forward(self, X, mod= None,former_shape=None,snr=None):    
        score = self.compute_score(X)
        samples, dist,_ = self.sample(score, mod=mod,former_shape=former_shape,snr=snr)
        out = self.recover(samples)
        return out, dist,None
    
    
