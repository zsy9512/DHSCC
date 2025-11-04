"""
Signal processing utilities for vibration classification.
"""

import numpy as np
import torch

def noise_to_psnr(noise):
    """Convert noise level to PSNR (Peak Signal-to-Noise Ratio) in dB"""
    return 20 * np.log10(1 / noise)

def psnr_to_noise(psnr):
    """Convert PSNR in dB to noise level"""
    return 1 / (10 ** (psnr / 20))

def set_seed(seed=0):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
def compute_reliability(sigma, N=128, device='cuda'):
    """
    巴氏参数递推法计算极化码信道可靠性，并按可靠性从高到低输出（可靠性=1-巴氏参数）
    """
    sigma = torch.tensor(sigma, dtype=torch.float32, device=device) if isinstance(sigma, (int, float)) else sigma
    if sigma.dim() == 0:
        Z = torch.zeros(N, device=device)
        Z[0] = torch.exp(-1 / (2 * sigma ** 2))
        n = int(np.log2(N))
        for i in range(n):
            j = 2 ** i
            Z_prev = Z.clone()
            Z[:j] = 2 * Z_prev[:j] - Z_prev[:j] ** 2
            Z[j:2*j] = Z_prev[:j] ** 2
        reliability = 1 - Z
        sorted_reliability, indices = torch.sort(reliability, descending=True)
        return sorted_reliability, indices
    else:
        # batch处理
        B = sigma.shape[0]
        Z = torch.zeros(B, N, device=device)
        Z[:, 0] = torch.exp(-1 / (2 * sigma ** 2))
        n = int(np.log2(N))
        for i in range(n):
            j = 2 ** i
            Z_prev = Z.clone()
            Z[:, :j] = 2 * Z_prev[:, :j] - Z_prev[:, :j] ** 2
            Z[:, j:2*j] = Z_prev[:, :j] ** 2
        reliability = 1 - Z
        sorted_reliability, indices = torch.sort(reliability, dim=1, descending=True)
        return sorted_reliability, indices