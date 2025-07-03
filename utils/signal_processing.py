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
