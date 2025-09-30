"""
Common data loading utilities for DHSCC end-to-end training.

This module provides shared functions for loading vibration datasets
across different training scripts and modules.
"""

import os
import torch
import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
import ast
import csv
import scipy.signal


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


def load_file_list(file_list_path):
    """
    Load file list from a text file.

    Args:
        file_list_path (str): Path to the file containing sample filenames and their labels
                             Format expected: "filename label" on each line

    Returns:
        list: List of tuples (filename, label) for each sample

    Raises:
        FileNotFoundError: If the file list cannot be found
    """
    if not os.path.exists(file_list_path):
        raise FileNotFoundError(f"File list not found at {file_list_path}")

    file_list = []
    with open(file_list_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                filename, label = parts
                file_list.append((filename, int(label)))

    return file_list


def load_class_names(class_map_path):
    """
    Load class names from a mapping file.

    Supports different formats:
    1. Space-separated format: "0 Class1"
    2. JSON-like format: "{'Class1': 0, 'Class2': 1}"
    3. Colon-separated format: "0: Class1" or "Class1: 0"
    4. CSV format

    Args:
        class_map_path (str): Path to the class mapping file

    Returns:
        dict: Dictionary mapping class indices to class names

    Raises:
        FileNotFoundError: If the class map file cannot be found
    """
    if not os.path.exists(class_map_path):
        raise FileNotFoundError(f"Class map not found at {class_map_path}")

    class_names = {}

    # Try to read the file contents
    with open(class_map_path, 'r') as f:
        content = f.read().strip()

    # Check if it appears to be JSON-like content (starts with '{')
    if content.startswith('{'):
        try:
            # Try to handle it as a Python dictionary
            class_dict = ast.literal_eval(content)

            # Class dict might be {class_name: idx} or {idx: class_name}
            # We'll handle both formats
            for key, value in class_dict.items():
                if isinstance(key, int) and isinstance(value, str):
                    class_names[key] = value
                elif isinstance(key, str) and isinstance(value, int):
                    class_names[value] = key

        except (SyntaxError, ValueError) as e:
            print(f"Warning: Could not parse class map as dictionary: {e}")

    # If dictionary parsing didn't work or didn't find valid mappings, try line-by-line
    if not class_names:
        with open(class_map_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                try:
                    # Try different parsing approaches
                    if ':' in line:
                        # Format like "Class1: 0" or "0: Class1"
                        parts = [p.strip() for p in line.split(':', 1)]
                        if parts[0].isdigit():
                            class_names[int(parts[0])] = parts[1].strip('"\'')
                        elif parts[1].strip().isdigit():
                            class_names[int(parts[1])] = parts[0].strip('"\'')
                    else:
                        # Format like "0 Class1"
                        parts = line.strip().split()
                        if parts and parts[0].isdigit():
                            idx = int(parts[0])
                            name = ' '.join(parts[1:])
                            class_names[idx] = name
                except Exception as e:
                    print(f"Warning: Skipping line '{line}': {e}")

    # If still empty, try one last approach - maybe it's a CSV
    if not class_names:
        try:
            with open(class_map_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        if row[0].isdigit():
                            class_names[int(row[0])] = row[1]
                        elif row[1].isdigit():
                            class_names[int(row[1])] = row[0]
        except Exception as e:
            print(f"Warning: Could not parse class map as CSV: {e}")

    if not class_names:
        print(f"Warning: No class names could be loaded from {class_map_path}")
        print(f"File content sample: {content[:100]}...")

    return class_names


class WavDataset(Dataset):
    """
    Dataset class for loading and processing vibration waveform data.

    Features:
    - Loads WAV files for vibration signals
    - Supports data augmentation
    - Handles resampling if needed

    Args:
        file_list (list): List of (filename, label) tuples
        data_dir (str): Directory containing the audio files
        augment (bool): Whether to apply data augmentation
        sample_rate (int): Target sample rate for audio
        seed (int): Random seed for reproducibility
    """
    def __init__(self, file_list, data_dir, augment=False, sample_rate=2800, seed=42):
        self.file_list = file_list
        self.data_dir = data_dir
        self.augment = augment
        self.sample_rate = sample_rate
        self.seed = seed

        # Set seed for any random operations in the dataset
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            tuple: (waveform, label) where waveform is a torch.Tensor and label is an integer
        """
        filename, label = self.file_list[idx]
        file_path = os.path.join(self.data_dir, filename)
        file_path = file_path.replace('\\', '/')  # Ensure path compatibility across platforms
        file_path = os.path.abspath(file_path)  # Convert to absolute path for torchaudio

        # Load audio file using scipy instead of torchaudio
        sr, waveform_np = wavfile.read(file_path)
        # Convert to torch tensor and ensure float type
        waveform = torch.from_numpy(waveform_np).float()
        # Add channel dimension if mono
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Resample if needed
        if sr != self.sample_rate:
            # Simple resampling using scipy
            target_length = int(len(waveform[0]) * self.sample_rate / sr)
            waveform = torch.from_numpy(scipy.signal.resample(waveform.numpy(), target_length, axis=1)).float()

        # Apply augmentation if enabled
        if self.augment:
            # Set seed based on idx to ensure reproducibility but different augmentations
            # for different samples
            aug_seed = self.seed + idx
            torch.manual_seed(aug_seed)
            np.random.seed(aug_seed)

            # Add minor random noise
            noise_factor = 0.01
            noise = torch.randn_like(waveform) * noise_factor
            waveform = waveform + noise
            if torch.rand(1).item() > 0.5:
                shift_amount = int(torch.randint(0, 512// 10, (1,)).item())
                waveform = torch.roll(waveform, shifts=shift_amount, dims=1)
        return waveform, label