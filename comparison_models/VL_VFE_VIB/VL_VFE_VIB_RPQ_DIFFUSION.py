#!/usr/bin/env python3
"""
VL-VFE (Variable-Length Vibration Feature Extraction) with VIB (Variational Information Bottleneck)
and Diffusion Denoising for vibration signal classification.

This model integrates channel-aware diffusion denoising to replace the simple decoder,
improving performance under AWGN channel conditions in 0-5dB SNR range.

Three-stage training:
1. VIB Encoder Pre-training: Learn feature extraction, compression and quantization
2. Diffusion Denoiser Training: Learn AWGN noise distribution and denoising
3. End-to-end Fine-tuning: Optimize classification performance

Usage:
    # Stage 1: VIB Encoder Pre-training
    python VL_VFE_VIB_RPQ_DIFFUSION.py --stage 1 --epochs 200 --lr 1e-3
    
    # Stage 2: Diffusion Denoiser Training
    python VL_VFE_VIB_RPQ_DIFFUSION.py --stage 2 --epochs 150 --lr 1e-3 --weights checkpoints/diffusion/stage1_best_acc.pth
    
    # Stage 3: End-to-end Fine-tuning
    python VL_VFE_VIB_RPQ_DIFFUSION.py --stage 3 --epochs 60 --lr 2e-5 --weights checkpoints/stage2_best_loss.pth
    
    # Testing
    python VL_VFE_VIB_RPQ_DIFFUSION.py --test 1 --weights checkpoints/stage3_final_acc_best.pth --snr -1

Requirements:
    pip install torch torchaudio numpy matplotlib
"""

# Check for required dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import StepLR
    import torchaudio
except ImportError as e:
    print(f"Error: Missing PyTorch dependencies: {e}")
    print("Please install PyTorch with: pip install torch torchaudio")
    exit(1)

try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error: Missing dependencies: {e}")
    print("Please install with: pip install numpy matplotlib")
    exit(1)

import argparse
import copy
import math
import os
import sys
import os.path

# Add the project root to path to import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

from utils.data import load_file_list, load_class_names, WavDataset
from wireless_comm.interface import WirelessInterface
from models.quantizer import VibrationVectorQuantizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser(
    description='VL-VFE VIB with Diffusion: Variable-Length Vibration Feature Extraction with Diffusion Denoising',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Model parameters
parser.add_argument('--intermediate_dim', type=int, default=256, 
                    help='Intermediate dimension size for feature compression')

# Training parameters
parser.add_argument('--stage', type=int, default=1, choices=[1, 2, 3],
                    help='Training stage: 1=VIB pre-training, 2=Diffusion training, 3=Fine-tuning')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--gamma', type=float, default=0.5, help='Learning rate decay factor')
parser.add_argument('--beta', type=float, default=1e-2, help='KL divergence weight coefficient')
parser.add_argument('--threshold', type=float, default=0, help='Threshold for feature pruning')
parser.add_argument('--decay_step', type=int, default=20, help='Learning rate decay step size')

# Diffusion parameters
parser.add_argument('--diffusion_steps', type=int, default=500, help='Number of diffusion steps')
parser.add_argument('--denoise_steps', type=int, default=50, help='Number of denoising steps during inference')

# Execution mode
parser.add_argument('--test', type=int, default=0, choices=[0, 1],
                    help='Execution mode: 0=training, 1=testing')
parser.add_argument('--weights', type=str, default=None,
                    help='Path to model weights file for testing/resuming training')

# SNR parameters (fixed to 0-5dB range)
parser.add_argument('--snr', type=float, default=2.0, 
                    help='SNR for single test in dB (use negative value for range testing)')
parser.add_argument('--snr_min', type=float, default=0, 
                    help='Minimum SNR for training/testing range in dB')
parser.add_argument('--snr_max', type=float, default=6, 
                    help='Maximum SNR for training/testing range in dB')
parser.add_argument('--snr_step', type=int, default=11, 
                    help='Number of SNR steps for comprehensive testing')

# Data parameters
parser.add_argument('--sample_rate', type=int, default=2800, 
                    help='Sample rate for audio files')

args = parser.parse_args()

# Validate arguments
if args.test == 1 and args.weights is None:
    print("Warning: Testing mode enabled but no weights file provided. Using random weights.")

if args.snr_min >= args.snr_max:
    print("Error: snr_min must be less than snr_max")
    exit(1)

# Dataset preparation (same as original)
def prepare_datasets():
    """Prepare train and test datasets for vibration classification"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..')
    dataset_root = os.path.join(project_root, 'dataset', 'vibcls')
    
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset directory not found at {dataset_root}")
        exit(1)
    
    required_files = ['clsmap.txt', 'train.txt', 'test.txt']
    for file in required_files:
        file_path = os.path.join(dataset_root, file)
        if not os.path.exists(file_path):
            print(f"Error: Required dataset file not found: {file_path}")
            exit(1)
    
    data_dir = os.path.join(dataset_root, 'vibrodata')
    if not os.path.exists(data_dir):
        print(f"Error: Vibration data directory not found at {data_dir}")
        exit(1)
    
    print(f"Dataset root: {dataset_root}")
    
    try:
        class_names = load_class_names(os.path.join(dataset_root, 'clsmap.txt'))
        train_list = load_file_list(os.path.join(dataset_root, 'train.txt'))
        test_list = load_file_list(os.path.join(dataset_root, 'test.txt'))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)
    
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_list)}")
    print(f"Test samples: {len(test_list)}")
    
    train_dataset = WavDataset(train_list, data_dir, augment=True, sample_rate=args.sample_rate)
    test_dataset = WavDataset(test_list, data_dir, augment=False, sample_rate=args.sample_rate)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader, num_classes

def seed_torch(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Utility layers (same as original)
class gamma_layer(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(gamma_layer, self).__init__()
        self.H = nn.Parameter(torch.ones(output_channel, input_channel))
        self.b = nn.Parameter(torch.ones(output_channel))
        self.H.data.normal_(0, 0.1)
        self.b.data.normal_(0, 0.001)

    def forward(self, x):
        H = torch.abs(self.H)
        x = F.linear(x, H)
        return torch.tanh(x)

class gamma_function(nn.Module):
    def __init__(self):
        super(gamma_function, self).__init__()
        self.f1 = gamma_layer(1, 16)
        self.f2 = gamma_layer(16, 16)
        self.f3 = gamma_layer(16, 16)
        self.f4 = gamma_layer(16, args.intermediate_dim)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        return x

class Flatten(nn.Module):
    def forward(self, x): 
        return x.view(x.size(0), x.size(1))

class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x): 
        return x * self.weight

# Attention mechanism for better feature learning
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, D = x.shape
        residual = x
        x = self.norm(x)
        
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).reshape(B, D)
        out = self.proj(out)
        
        return out + residual

# Improved ResNet block with attention
class ResNetBlock(nn.Module):
    def __init__(self, dim, use_attention=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GroupNorm(8, dim) if dim >= 8 else nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.GroupNorm(8, dim) if dim >= 8 else nn.LayerNorm(dim),
        )
        self.attention = SelfAttention(dim) if use_attention else None
        
    def forward(self, x, time_emb=None, snr_emb=None):
        residual = x
        x = self.block(x)
        
        # Add conditioning
        if time_emb is not None:
            x = x + time_emb
        if snr_emb is not None:
            x = x + snr_emb
            
        x = F.silu(x + residual)
        
        if self.attention is not None:
            x = self.attention(x)
            
        return x

# Enhanced 1D UNet for diffusion denoising
class Enhanced1DUNet(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, out_channels=None, num_layers=6):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Time and SNR embedding projections with more capacity
        self.time_proj = nn.Sequential(
            nn.Linear(128, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.snr_proj = nn.Sequential(
            nn.Linear(64, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # Down blocks with increasing complexity
        self.down_blocks = nn.ModuleList()
        for i in range(num_layers):
            use_attention = i >= num_layers // 2  # Add attention in deeper layers
            self.down_blocks.append(ResNetBlock(hidden_channels, use_attention))
        
        # Middle block with attention
        self.middle_block = nn.Sequential(
            ResNetBlock(hidden_channels, use_attention=True),
            ResNetBlock(hidden_channels, use_attention=True)
        )
        
        # Up blocks with skip connections
        self.up_blocks = nn.ModuleList()
        for i in range(num_layers):
            use_attention = i < num_layers // 2  # Add attention in earlier up blocks
            self.up_blocks.append(nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                ResNetBlock(hidden_channels, use_attention)
            ))
        
        # Output projection with residual
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, hidden_channels) if hidden_channels >= 8 else nn.LayerNorm(hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        
    def get_timestep_embedding(self, timesteps, embedding_dim=128):
        """Create sinusoidal timestep embeddings"""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb
    
    def forward(self, x, time_emb, snr_emb):
        # x: [batch, feature_dim]
        # time_emb: [batch, 128] (from timestep embedding)
        # snr_emb: [batch, 64] (from SNR encoder)
        
        # Project input
        h = self.input_proj(x)
        
        # Project embeddings
        time_emb_proj = self.time_proj(time_emb)
        snr_emb_proj = self.snr_proj(snr_emb)
        
        # Store skip connections
        skip_connections = []
        
        # Down path
        for down_block in self.down_blocks:
            skip_connections.append(h)
            h = down_block(h, time_emb_proj, snr_emb_proj)
        
        # Middle processing
        h = self.middle_block[0](h, time_emb_proj, snr_emb_proj)
        h = self.middle_block[1](h, time_emb_proj, snr_emb_proj)
        
        # Up path
        for up_block in self.up_blocks:
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=1)
            h = up_block[0](h)  # Linear layer
            h = up_block[1](h, time_emb_proj, snr_emb_proj)  # ResNet block
        
        # Output
        return self.output_proj(h)

# Enhanced AWGN Diffusion Denoiser with improved architecture
class AWGNDiffusionDenoiser(nn.Module):
    def __init__(self, feature_dim, diffusion_steps=500):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.diffusion_steps = diffusion_steps
        
        # Enhanced 1D UNet with attention and better architecture
        self.unet = Enhanced1DUNet(
            in_channels=feature_dim,
            hidden_channels=256,  # Increased capacity
            out_channels=feature_dim,
            num_layers=6  # Deeper network
        )
        
        # Improved noise schedule - cosine schedule for better performance
        self.register_buffer('betas', self._cosine_beta_schedule(diffusion_steps))
        
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
        # For sampling
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Enhanced SNR condition encoder with more capacity
        self.snr_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU()
        )
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine noise schedule for better diffusion performance"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward(self, noisy_features, timestep, snr):
        """Forward pass during training"""
        batch_size = noisy_features.shape[0]
        #snr=torch.tensor(snr, device=noisy_features.device)  # Ensure correct shape
        snr = torch.tensor(snr, device=noisy_features.device)
        # SNR condition encoding
        snr_emb = self.snr_encoder(snr.view(-1, 1))  # [batch, 64]
        
        # Time embedding
        t_emb = self.unet.get_timestep_embedding(timestep)  # [batch, 128]
        
        # UNet prediction (UNet will handle projection internally)
        return self.unet(noisy_features, t_emb, snr_emb)
    
    def add_noise(self, x_start, noise, timesteps):
        """Add noise to clean features according to diffusion schedule"""
        sqrt_alphas_cumprod_t = self.alphas_cumprod[timesteps].sqrt()
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[timesteps]).sqrt()
        
        return (
            sqrt_alphas_cumprod_t.view(-1, 1) * x_start +
            sqrt_one_minus_alphas_cumprod_t.view(-1, 1) * noise
        )
    
    def denoise(self, noisy_features, snr, steps=None):
        """Denoising process during inference"""
        if steps is None:
            steps = args.denoise_steps
        
        batch_size = noisy_features.shape[0]
        x = noisy_features
        
        # Adaptive step selection based on SNR
        if snr > 3:
            actual_steps = max(10, steps // 3)  # High SNR, less denoising
        else:
            actual_steps = steps  # Low SNR, more denoising
        
        step_size = max(1, self.diffusion_steps // actual_steps)
        
        for i in reversed(range(0, self.diffusion_steps, step_size)):
            t = torch.full((batch_size,), i, device=x.device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.forward(x, t, torch.full((batch_size,), snr, device=x.device))
            
            # Denoising step
            alpha_t = self.alphas_cumprod[i]
            alpha_prev = self.alphas_cumprod_prev[i]
            
            # Compute x_{t-1}
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            if i > 0:
                x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * predicted_noise
            else:
                x = pred_x0
        
        return x

# Main model with diffusion integration
class VibNetWithDiffusion(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()

        self.args = args
        self.hidden_channel = args.intermediate_dim
        self.num_classes = num_classes

        # Feature extraction layers (same as original)
        self.prep = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.layer1_res = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        
        self.layer3_res = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # VIB Encoder (same as original)
        self.encoder1 = nn.Sequential(
            nn.Conv1d(512, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(4),
            nn.ReLU()
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16)
        
        self.encoder2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.Sigmoid()
        )

        # Encoder 3 parameters
        self.encoder3_weight = nn.Parameter(torch.Tensor(self.hidden_channel, 64))
        self.encoder3_bias = nn.Parameter(torch.Tensor(self.hidden_channel))
        self.encoder3_weight.data.normal_(0, 0.5)
        self.encoder3_bias.data.normal_(0, 0.1)

        # Gamma function for VIB
        self.gamma_mu = gamma_function().to(device)
        self.upper_tri_matrix = torch.triu(torch.ones((args.intermediate_dim, args.intermediate_dim))).to(device)
        
        # Quantizer
        self.quantizer = VibrationVectorQuantizer(
            dimension=256,
            n_codes_group=16,
            bins=16,
            n_q=64
        )
        
        # Wireless interface
        self.wireless_interface = WirelessInterface(comm_mode=3,channel_model='awgn' )
        self.diffusion_steps = args.diffusion_steps
        # NEW: Diffusion denoiser
        self.diffusion_denoiser = AWGNDiffusionDenoiser(
            feature_dim=args.intermediate_dim,
            diffusion_steps=args.diffusion_steps
        )
        
        # Simple decoder (for stage 1 training)
        self.simple_decoder = nn.Sequential(
            nn.Linear(self.hidden_channel, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Classifier
        self.classifier1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Flatten()
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(512, self.num_classes, bias=False),
            Mul(0.125)
        )
        
        # Feature reconstruction layers (for diffusion output)
        self.feature_reconstructor = nn.Sequential(
            nn.Linear(self.hidden_channel, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.decoder2_upsample = nn.Upsample(size=32, mode='linear', align_corners=False)
        self.decoder2 = nn.Sequential(
            nn.Conv1d(4, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

    def extract_features(self, x):
        """Extract CNN features"""
        x = self.prep(x)
        x = self.layer1(x)
        res = self.layer1_res(x)
        x = res + x
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    
    def vib_encode(self, features, snr):
        """VIB encoding with channel noise conditioning"""
        # Encoder
        encoded_features = self.encoder1(features)
        encoded_features = self.adaptive_pool(encoded_features)
        x = torch.reshape(encoded_features, (encoded_features.size()[0], 4*16))
        
        # Channel noise calculation
        if isinstance(snr, torch.Tensor):
            if snr.dim() == 0:
                snr = snr.unsqueeze(0)
            channel_noise = 10 ** (-snr / 20.0)
        else:
            channel_noise = torch.FloatTensor([10 ** (-snr / 20.0)])
        channel_noise = channel_noise.to(device)
        
        x = self.encoder2(x)
        x_norm2 = torch.norm(x, dim=1)
        x = 64 * (x.permute(1,0)/(x_norm2+1e-6)).permute(1,0)
        
        # Encoder 3
        weight3 = F.tanh(self.encoder3_weight)
        bias3 = F.tanh(self.encoder3_bias)
        weight3 = torch.clamp(torch.abs(weight3), min=1e-3) * torch.sign(weight3.detach())
        bias3 = torch.clamp(torch.abs(bias3), min=1e-3) * torch.sign(bias3.detach())
        l2_norm_squared = torch.sum(weight3.pow(2), dim=1) + bias3.pow(2)
        l2_norm = l2_norm_squared.pow(0.5)
        weight3 = (weight3.permute(1,0) / (l2_norm+1e-6)).permute(1,0)
        bias3 = bias3 / (l2_norm+1e-6)
        x = F.linear(x, weight3, bias3)

        # Gamma function
        mu = self.gamma_mu(channel_noise)
        mu = F.linear(mu, self.upper_tri_matrix)
        mu = torch.clamp(mu, min=1e-4)
        encoded_feature = torch.tanh(x * mu)
        encoded_feature = torch.clamp(torch.abs(encoded_feature), min=1e-2) * torch.sign(encoded_feature.detach())
        
        return encoded_feature, channel_noise, mu
    
    def get_quantized_features(self, x,snr):
        """Get quantized features (for stage 2 training)"""
        features = self.extract_features(x)
        encoded, _, _ = self.vib_encode(features, snr=snr)  # Use middle SNR
        quantized, all_indices, _ = self.quantizer(encoded)
    
        return encoded, encoded,quantized,all_indices
    
    def awgn_channel(self, features, snr):
        """Simulate AWGN channel"""
        sigma = 10 ** (-snr / 20.0)
        noise = torch.randn_like(features) * sigma
        return features + noise
    
    def forward_stage1(self, x, epoch, snr):
        """Stage 1: VIB encoder pre-training"""
        # Feature extraction and VIB encoding
        features = self.extract_features(x)
        encoded_feature, channel_noise, mu = self.vib_encode(features, snr)
        
        # Quantization
        quantized_features, all_indices, commit_loss = self.quantizer(encoded_feature)
        
        # KL divergence
        KL = self.KL_log_uniform(channel_noise, torch.abs(quantized_features.detach()))
        
        # Wireless transmission
        transmission_result = self.wireless_interface.transmit_mode3(
            indices=all_indices,
            snr_db=snr.cpu().item() if isinstance(snr, torch.Tensor) else snr,
            bits_per_index=4,
            group_to_use=16
        )
        received_indices = transmission_result['received_indices']
        received_features_decoded = self.quantizer.decode(received_indices)
        
        # For gradient flow
        received_features = quantized_features + (received_features_decoded - quantized_features).detach()
        
        # Simple decoding (stage 1)
        decoded = self.simple_decoder(received_features)
        #decoded = self.feature_reconstructor(received_features)
        # Reconstruct to feature space
        x = torch.reshape(decoded, (-1, 4, 16))
        x = self.decoder2_upsample(x)
        decoded_feature = self.decoder2(x)
        
        # Residual connection
        x = self.layer3_res(decoded_feature)
        x = x + decoded_feature
        
        # Classification
        x = self.classifier1(x)
        output = self.classifier2(x)
        
        return output, (KL * 0.1 / channel_noise), torch.mean(commit_loss)
    
    def forward_stage2(self, x, snr):
        """Stage 2: 学习真实信道噪声的扩散去噪"""
        batch_size = x.shape[0]
        
        # 获取干净量化特征
        with torch.no_grad():
            clean_features,encode_features,quantized_features,all_indices = self.get_quantized_features(x,snr)
        
        # 模拟真实的无线传输过程
        transmission_result = self.wireless_interface.transmit_mode3(indices=all_indices,
            snr_db=snr.cpu().item() if isinstance(snr, torch.Tensor) else snr,
            bits_per_index=4,
            group_to_use=16)
        received_indices = transmission_result['received_indices']
        received_features_decoded = self.quantizer.decode(received_indices)
        
        # For gradient flow
        received_features = quantized_features + (received_features_decoded - quantized_features).detach()
        
        
        # 真实的信道噪声
        real_channel_noise = received_features - clean_features
        
        # 扩散训练：学习预测真实信道噪声
        t = torch.randint(0, self.diffusion_steps, (batch_size,), device=x.device)
        
        # 在真实噪声基础上添加扩散噪声
        diffusion_noise = torch.randn_like(real_channel_noise)
        #noisy_input = self.add_noise(real_channel_noise, diffusion_noise, t)
        noisy_input = self.diffusion_denoiser.add_noise(real_channel_noise, diffusion_noise, t)
        # 预测扩散噪声（而不是信道噪声）
        predicted_noise = self.diffusion_denoiser(noisy_input, t, snr)
         
        # 损失：预测的扩散噪声 vs 真实的扩散噪声
        diffusion_loss = F.mse_loss(predicted_noise, diffusion_noise)
        
        return diffusion_loss
    
    def forward_stage3(self, x, snr):
        """Stage 3: End-to-end fine-tuning"""
        # Full forward pass with frozen encoder and diffusion
        with torch.no_grad():
            # Feature extraction and encoding
            features = self.extract_features(x)
            encoded_feature, _, _ = self.vib_encode(features, snr)
            
            # Quantization and transmission
            quantized_features, all_indices, _ = self.quantizer(encoded_feature)
            transmission_result = self.wireless_interface.transmit_mode3(
                indices=all_indices,
                snr_db=snr.cpu().item() if isinstance(snr, torch.Tensor) else snr,
                bits_per_index=4,
                group_to_use=16
            )
            received_indices = transmission_result['received_indices']
            received_features = self.quantizer.decode(received_indices)
            
            # Diffusion denoising - 关键修复！
            denoised_features = self.diffusion_denoiser.denoise(received_features, snr)
        
        # Feature reconstruction (trainable) - 使用去噪后的特征！
        decoded = self.simple_decoder(denoised_features)
        
        # Reconstruct to feature space
        x = torch.reshape(decoded, (-1, 4, 16))
        x = self.decoder2_upsample(x)
        decoded_feature = self.decoder2(x)
        
        # Residual connection
        x = self.layer3_res(decoded_feature)
        x = x + decoded_feature
        
        # Classification
        x = self.classifier1(x)
        output = self.classifier2(x)
        
        return output
    
    def forward_full(self, x, snr):
        """Full forward pass for inference"""
        # Feature extraction and encoding
        features = self.extract_features(x)
        encoded_feature, _, _ = self.vib_encode(features, snr)
        
        # Quantization and transmission
        quantized_features, all_indices, _ = self.quantizer(encoded_feature)
        transmission_result = self.wireless_interface.transmit_mode3(
            indices=all_indices,
            snr_db=snr.cpu().item() if isinstance(snr, torch.Tensor) else snr,
            bits_per_index=4,
            group_to_use=16
        )
        received_indices = transmission_result['received_indices']
        received_features = self.quantizer.decode(received_indices)
        
        # Diffusion denoising
        denoised_features = self.diffusion_denoiser.denoise(received_features, snr)
        
        # Feature reconstruction - 使用与Stage 3相同的解码器！
        decoded = self.simple_decoder(denoised_features)
        
        # Reconstruct to feature space
        x = torch.reshape(decoded, (-1, 4, 16))
        x = self.decoder2_upsample(x)
        decoded_feature = self.decoder2(x)
        
        # Residual connection
        x = self.layer3_res(decoded_feature)
        x = x + decoded_feature
        
        # Classification
        x = self.classifier1(x)
        output = self.classifier2(x)
        
        return output
    
    def forward(self, x, epoch, snr=2.0, stage='full'):
        """Main forward function"""
        # Ensure input is the right shape
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        
        # Random SNR sampling during training
        if self.training and stage != 'stage2':
            snr = torch.rand(1) * (args.snr_max - args.snr_min) + args.snr_min
            snr = snr.to(device)
        
        if isinstance(snr, torch.Tensor):
            snr = snr.cpu().item() if snr.dim() == 0 else snr[0].cpu().item()
        
        if stage == 'stage1':
            return self.forward_stage1(x, epoch, snr)
        elif stage == 'stage2':
            return self.forward_stage2(x, snr)
        elif stage == 'stage3':
            return self.forward_stage3(x, snr)
        else:
            return self.forward_full(x, snr)
    
    def KL_log_uniform(self, channel_noise, encoded_feature):
        """KL divergence calculation"""
        alpha = (channel_noise/encoded_feature)
        k1 = 0.635766
        k2 = 1.8732
        k3 = 1.48695
        batch_size = alpha.size(0)
        KL_term = k1 * F.sigmoid(k2 + k3 * 2 * torch.log(alpha)) - 0.5 * F.softplus(-2 * torch.log(alpha)) - k1
        return - torch.sum(KL_term) / batch_size
    
    def get_mask_inference(self, snr, threshold=0):
        """Get mask for inference"""
        channel_noise = torch.FloatTensor([10 ** (-snr / 20.0)]).to(device)
        mu = self.gamma_mu(channel_noise)
        alpha = F.linear(mu, self.upper_tri_matrix)
        hard_mask = (alpha > threshold).float()
        return hard_mask, alpha
    
    def freeze_encoder(self):
        """Freeze encoder components for stage 2"""
        for param in self.prep.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer1_res.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        for param in self.layer3.parameters():
            param.requires_grad = False
        for param in self.encoder1.parameters():
            param.requires_grad = False
        for param in self.encoder2.parameters():
            param.requires_grad = False
        self.encoder3_weight.requires_grad = False
        self.encoder3_bias.requires_grad = False
        for param in self.gamma_mu.parameters():
            param.requires_grad = False
        for param in self.quantizer.parameters():
            param.requires_grad = False
    
    def freeze_diffusion(self):
        """Freeze diffusion components for stage 3"""
        for param in self.diffusion_denoiser.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True

# Training functions
def train_stage1(model, train_loader, test_loader, optimizer, scheduler):
    """Stage 1: VIB encoder pre-training"""
    print("=== Stage 1: VIB Encoder Pre-training ===")
    
    best_acc = 0
    saved_model = {}
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(current_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        model.train()
        total_loss = 0
        num_batches = 0
        
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            
            # Random SNR sampling
            snr = torch.rand(1) * (args.snr_max - args.snr_min+1e-6) + args.snr_min
            
            output, KL, commit_loss = model(x, epoch, snr, stage='stage1')
            
            # Loss calculation
            criterion = nn.CrossEntropyLoss()
            cls_loss = criterion(output, y)
            
            if epoch <= 20:
                loss = cls_loss + 0.1 * commit_loss
            else:
                anneal_ratio = min(1, (epoch - 20) / 20)
                loss = cls_loss + args.beta * KL * anneal_ratio + 0.01 * commit_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if i % 50 == 0:
                print(f'Batch {i}, Loss: {loss.item():.4f}')
        
        scheduler.step()
        
        # Test
        acc = test_stage1(model, test_loader, epoch,snr)
        print(f'Epoch {epoch+1}, Avg Loss: {total_loss/num_batches:.4f}, Test Acc: {acc:.2f}%')
        
        # Save best model
        if acc > best_acc and epoch >= args.epochs-20:
            best_acc = acc
            saved_model = copy.deepcopy(model.state_dict())
            checkpoint_path = os.path.join(checkpoint_dir, f'stage1_best_acc.pth')
            torch.save({'model': saved_model, 'epoch': epoch, 'acc': acc}, checkpoint_path)
            print(f'Best model saved: {acc:.2f}%')
    
    print(f'Stage 1 completed. Best accuracy: {best_acc:.2f}%')
    return saved_model

def test_stage1(model, test_loader, epoch,snr):
    """Test function for stage 1"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            
            outputs, _, _ = model(signals, epoch, snr=snr, stage='stage1')
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def train_stage2(model, train_loader, optimizer, scheduler):
    """Stage 2: Diffusion denoiser training"""
    print("=== Stage 2: Diffusion Denoiser Training ===")
    
    model.freeze_encoder()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(current_dir, 'checkpoints')
    
    best_loss = float('inf')
    saved_model = {}
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        model.train()
        total_loss = 0
        num_batches = 0
        
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            
            # Random SNR sampling
            snr = torch.rand(1) * (args.snr_max - args.snr_min+1e-6) + args.snr_min
            snr = snr.item()
            
            diffusion_loss = model(x, epoch, snr, stage='stage2')
            
            optimizer.zero_grad()
            diffusion_loss.backward()
            optimizer.step()
            
            total_loss += diffusion_loss.item()
            num_batches += 1
            
            if i % 50 == 0:
                print(f'Batch {i}, Diffusion Loss: {diffusion_loss.item():.4f}')
        
        scheduler.step()
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1}, Avg Diffusion Loss: {avg_loss:.4f}')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            saved_model = copy.deepcopy(model.state_dict())
            checkpoint_path = os.path.join(checkpoint_dir, f'stage2_best_loss.pth')
            torch.save({'model': saved_model, 'epoch': epoch, 'loss': avg_loss}, checkpoint_path)
            print(f'Best diffusion model saved: {avg_loss:.4f}')
    
    print(f'Stage 2 completed. Best loss: {best_loss:.4f}')
    return saved_model

def train_stage3(model, train_loader, test_loader, optimizer, scheduler):
    """Stage 3: End-to-end fine-tuning"""
    print("=== Stage 3: End-to-end Fine-tuning ===")
    
    model.freeze_encoder()
    model.freeze_diffusion()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(current_dir, 'checkpoints')
    
    best_acc = 0
    saved_model = {}
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        model.train()
        total_loss = 0
        num_batches = 0
        
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            
            # Random SNR sampling
            snr = torch.rand(1) * (args.snr_max - args.snr_min+1e-6) + args.snr_min
            snr = snr.item()
            
            output = model(x, epoch, snr, stage='stage3')
            
            # Classification loss only
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if i % 50 == 0:
                print(f'Batch {i}, Loss: {loss.item():.4f}')
        
        scheduler.step()
        
        # Test
        acc = test_stage3(model, test_loader, epoch,snr)
        print(f'Epoch {epoch+1}, Avg Loss: {total_loss/num_batches:.4f}, Test Acc: {acc:.2f}%')
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            saved_model = copy.deepcopy(model.state_dict())
            checkpoint_path = os.path.join(checkpoint_dir, f'stage3_final_acc_best.pth')
            torch.save({'model': saved_model, 'epoch': epoch, 'acc': acc}, checkpoint_path)
            print(f'Best final model saved: {acc:.2f}%')
    
    print(f'Stage 3 completed. Best accuracy: {best_acc:.2f}%')
    return saved_model

def test_stage3(model, test_loader, epoch,snr):
    """Test function for stage 3"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            
            outputs = model(signals, epoch, snr=snr, stage='stage3')
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def test_full_model(model, test_loader, snr=2.0):
    """Test the full model"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            
            outputs = model(signals, 0, snr, stage='full')
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def test_snr_range(model, test_loader, snr_min=0, snr_max=5, steps=11):
    """Test model across SNR range"""
    snr_values = np.linspace(snr_min, snr_max, steps)
    accuracies = []
    
    for snr in snr_values:
        print(f"Testing SNR: {snr:.2f} dB")
        accuracy = 0
        
        # Run multiple tests and average
        for _ in range(5):
            acc = test_full_model(model, test_loader, snr)
            accuracy += acc
        
        accuracy /= 5
        accuracies.append(accuracy)
        print(f'SNR: {snr:.2f} dB, Accuracy: {accuracy:.2f}%')
    
    return snr_values, accuracies

def plot_results(snr_values, accuracies, output_dir='.'):
    """Plot SNR vs Accuracy results"""
    plt.figure(figsize=(10, 6))
    plt.plot(snr_values, accuracies, 'o-', color='blue', linewidth=2, markersize=8)
    plt.grid(True, alpha=0.3)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy (%)')
    plt.title('VL-VFE-VIB with Diffusion Denoising: Accuracy vs SNR')
    plt.xlim(snr_values[0], snr_values[-1])
    plt.ylim(0, 100)
    
    # Add text annotations for key points
    for i, (snr, acc) in enumerate(zip(snr_values, accuracies)):
        if i % 2 == 0:  # Annotate every other point to avoid crowding
            plt.annotate(f'{acc:.1f}%', (snr, acc), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/VL_VFE_VIB_Diffusion_SNR_results.png', dpi=300, bbox_inches='tight')
    print(f"Results saved to {output_dir}/VL_VFE_VIB_Diffusion_SNR_results.png")
    plt.close()

if __name__ == '__main__':
    seed_torch(0)
    
    # Prepare datasets
    train_loader, test_loader, num_classes = prepare_datasets()
    
    # Initialize model
    model = VibNetWithDiffusion(args, num_classes=num_classes).to(device)
    
    if args.test == 0:
        # Training mode
        if args.stage == 1:
            # Stage 1: VIB encoder pre-training
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
            
            if args.weights:
                print(f"Loading weights from {args.weights}")
                checkpoint = torch.load(args.weights)
                model.load_state_dict(checkpoint['model'])
            
            saved_model = train_stage1(model, train_loader, test_loader, optimizer, scheduler)
            
        elif args.stage == 2:
            # Stage 2: Diffusion denoiser training
            if args.weights is None:
                print("Error: Stage 2 requires pre-trained weights from Stage 1")
                exit(1)
            
            print(f"Loading Stage 1 weights from {args.weights}")
            checkpoint = torch.load(args.weights)
            model.load_state_dict(checkpoint['model'])
            
            # Only optimize diffusion parameters
            optimizer = torch.optim.Adam(model.diffusion_denoiser.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
            
            saved_model = train_stage2(model, train_loader, optimizer, scheduler)
            
        elif args.stage == 3:
            # Stage 3: End-to-end fine-tuning
            if args.weights is None:
                print("Error: Stage 3 requires pre-trained weights from Stage 2")
                exit(1)
            
            print(f"Loading Stage 2 weights from {args.weights}")
            checkpoint = torch.load(args.weights)
            model.load_state_dict(checkpoint['model'])
            
            # Only optimize classifier and feature reconstructor
            trainable_params = list(model.feature_reconstructor.parameters()) + \
                             list(model.classifier1.parameters()) + \
                             list(model.classifier2.parameters()) + \
                             list(model.layer3_res.parameters()) + \
                             list(model.decoder2.parameters())
            
            optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-4)
            scheduler = StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
            
            saved_model = train_stage3(model, train_loader, test_loader, optimizer, scheduler)
    
    else:
        # Testing mode
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model weights
        if args.weights:
            print(f"Loading weights from {args.weights}")
            checkpoint = torch.load(args.weights)
            model.load_state_dict(checkpoint['model'])
        else:
            print("Warning: No weights provided, using random initialization")
        
        if args.snr > 0:
            # Test single SNR
            accuracy = test_full_model(model, test_loader, args.snr)
            print(f'SNR: {args.snr:.2f} dB, Test Accuracy: {accuracy:.2f}%')
        else:
            # Test SNR range
            print("Running comprehensive SNR analysis...")
            snr_values, accuracies = test_snr_range(
                model, test_loader, args.snr_min, args.snr_max, args.snr_step
            )
            plot_results(snr_values, accuracies, output_dir)
            
            # Save numerical results
            results = {
                'snr_values': snr_values.tolist(),
                'accuracies': accuracies,
            }
            np.save(f'{output_dir}/VL_VFE_VIB_Diffusion_snr_results.npy', results)
            print(f"Numerical results saved to {output_dir}/VL_VFE_VIB_Diffusion_snr_results.npy")
