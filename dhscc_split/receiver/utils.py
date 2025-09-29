"""
Utility functions for sender.
"""

import torch
import random
import numpy as np
import os

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(dir_path):
    """Ensure directory exists."""
    os.makedirs(dir_path, exist_ok=True)

def save_model(model, quantizer, path):
    """Save sender model and quantizer."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'quantizer_state_dict': quantizer.state_dict()
    }, path)