"""
Data loader module for DHSCC project.

This module provides data loading functionality for vibration signal classification.
"""

import os
import torch
import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
import ast
import csv
import scipy.signal

# Import from local data module
from .data import WavDataset, load_file_list, load_class_names

def create_data_loaders_from_args(args):
    """
    Create data loaders from command line arguments.

    Args:
        args: Parsed command line arguments containing data paths

    Returns:
        tuple: (train_loader, test_loader, num_classes)
    """
    data_dir = args.data_dir
    train_list = args.train_list
    test_list = args.test_list
    class_map = args.class_map
    batch_size = getattr(args, 'batch_size', 64)
    augment_train = getattr(args, 'augment_train', True)

    # print(f"Loading data from:")
    # print(f"  Data directory: {data_dir}")
    # print(f"  Train list: {train_list}")
    # print(f"  Test list: {test_list}")
    # print(f"  Class map: {class_map}")

    # Check if data files exist
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not os.path.exists(train_list):
        raise FileNotFoundError(f"Train list file not found: {train_list}")
    if not os.path.exists(test_list):
        raise FileNotFoundError(f"Test list file not found: {test_list}")
    if not os.path.exists(class_map):
        raise FileNotFoundError(f"Class map file not found: {class_map}")

    # Load data files and class information
    train_files = load_file_list(train_list)
    test_files = load_file_list(test_list)
    class_names = load_class_names(class_map)
    num_classes = len(class_names)

    # print(f"  Number of classes: {num_classes}")
    # print(f"  Training samples: {len(train_files)}")
    # print(f"  Test samples: {len(test_files)}")

    # Create datasets
    train_dataset = WavDataset(train_files, data_dir, augment=augment_train)
    test_dataset = WavDataset(test_files, data_dir, augment=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    return train_loader, test_loader, num_classes