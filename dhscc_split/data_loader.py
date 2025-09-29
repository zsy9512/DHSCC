"""
Common data loading utilities for DHSCC end-to-end training.

This module provides shared functions for loading vibration datasets
across different training scripts and modules.
"""

import os
import torch
import sys
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.data import WavDataset, load_file_list, load_class_names


def create_data_loaders_from_config(data_dir, train_list, test_list, class_map,
                                   batch_size=64, augment_train=True):
    """
    Create data loaders from configuration paths.

    Args:
        data_dir (str): Directory containing the WAV files
        train_list (str): Path to training file list
        test_list (str): Path to test file list
        class_map (str): Path to class mapping file
        batch_size (int): Batch size for data loaders
        augment_train (bool): Whether to apply augmentation to training data

    Returns:
        tuple: (train_loader, test_loader, num_classes)
    """
    print(f"Loading data from:")
    print(f"  Data directory: {data_dir}")
    print(f"  Train list: {train_list}")
    print(f"  Test list: {test_list}")
    print(f"  Class map: {class_map}")

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
    train_file_list = load_file_list(train_list)
    test_file_list = load_file_list(test_list)
    class_names = load_class_names(class_map)

    print(f"Loaded {len(train_file_list)} training samples and {len(test_file_list)} test samples")
    print(f"Classes: {list(class_names.values())}")
    num_classes = len(class_names)

    # Create datasets
    train_dataset = WavDataset(train_file_list, data_dir, augment=augment_train)
    test_dataset = WavDataset(test_file_list, data_dir, augment=False)

    # Create data loaders - optimized for PyTorch 2.x on Windows
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # PyTorch 2.x on Windows may have multiprocessing issues
        pin_memory=False,  # Disable pin_memory for compatibility
        persistent_workers=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Single process for testing to improve speed
        pin_memory=False,  # Disable pin_memory
        persistent_workers=False
    )

    print(f"Created data loaders:")
    print(f"  Train: {len(train_dataset)} samples, {num_classes} classes")
    print(f"  Test: {len(test_dataset)} samples, {num_classes} classes")
    print(f"  Batch size: {batch_size}")

    return train_loader, test_loader, num_classes


def create_data_loaders_from_args(args):
    """
    Create data loaders from argument parser args.

    This is a convenience function that extracts data paths from args
    and calls create_data_loaders_from_config.

    Args:
        args: Argument parser namespace with data path attributes

    Returns:
        tuple: (train_loader, test_loader, num_classes)
    """
    # Use dataset from the current dhscc_split directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Adjust data paths to point to current directory's dataset
    data_dir = os.path.join(current_dir, args.data_dir.lstrip('./').lstrip('/'))
    train_list = os.path.join(current_dir, args.train_list.lstrip('./').lstrip('/'))
    test_list = os.path.join(current_dir, args.test_list.lstrip('./').lstrip('/'))
    class_map = os.path.join(current_dir, args.class_map.lstrip('./').lstrip('/'))

    return create_data_loaders_from_config(
        data_dir=data_dir,
        train_list=train_list,
        test_list=test_list,
        class_map=class_map,
        batch_size=args.batch_size,
        augment_train=True
    )


def get_default_data_paths(base_dir="dataset"):
    """
    Get default data paths for the current setup.

    Args:
        base_dir (str): Base dataset directory name

    Returns:
        dict: Dictionary with data path keys
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    return {
        'data_dir': os.path.join(current_dir, base_dir, "vibrodata"),
        'train_list': os.path.join(current_dir, base_dir, "train.txt"),
        'test_list': os.path.join(current_dir, base_dir, "test.txt"),
        'class_map': os.path.join(current_dir, base_dir, "clsmap.txt")
    }


def validate_dataset_exists(base_dir="dataset"):
    """
    Validate that all required dataset files exist.

    Args:
        base_dir (str): Base dataset directory name

    Returns:
        bool: True if all files exist, False otherwise
    """
    paths = get_default_data_paths(base_dir)

    missing_files = []
    for name, path in paths.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")

    if missing_files:
        print("Missing dataset files:")
        for missing in missing_files:
            print(f"  - {missing}")
        return False

    print("All dataset files found:")
    for name, path in paths.items():
        print(f"  - {name}: {path}")
    return True