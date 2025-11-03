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
# Functions moved inline to avoid import issues

def create_data_loaders_from_args(args):
    """
    Create data loaders from command line arguments.

    Args:
        args: Parsed command line arguments containing data paths

    Returns:
        tuple: (train_loader, test_loader, num_classes)
    """
    # Use fixed paths for simplified setup
    data_dir = "data/vibcls/vibrodata"
    train_list = "data/vibcls/train.txt"
    
    # Check if category is specified
    category = getattr(args, 'category', None)
    if category is not None:
        test_list = f"data/vibcls/test_{category}.txt"
        print(f"Using category-specific test data: {test_list} (category {category})")
    else:
        test_list = "data/vibcls/test.txt"
        print("Using default test data: test.txt")
    
    class_map = "data/vibcls/clsmap.txt"
    batch_size = getattr(args, 'batch_size', 5)

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
    train_files = load_file_list_with_category(train_list, None)  # train.txt should have labels
    test_files = load_file_list_with_category(test_list, category)
    class_names = load_class_names(class_map)
    num_classes = len(class_names)

    # Create datasets
    train_dataset = WavDataset(train_files, data_dir, augment=False)  # No augmentation for test
    test_dataset = WavDataset(test_files, data_dir, augment=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    return train_loader, test_loader, num_classes


def load_file_list_with_category(file_list_path, category=None):
    """
    Load file list from a text file, with optional category assignment.
    
    Args:
        file_list_path (str): Path to the file containing sample filenames
        category (int, optional): If specified, assign this category to all files
        
    Returns:
        list: List of tuples (filename, label) for each sample
    """
    if not os.path.exists(file_list_path):
        raise FileNotFoundError(f"File list not found at {file_list_path}")
    
    file_list = []
    with open(file_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            filename = parts[0]
            
            if category is not None:
                # For category-specific files, assign the specified category
                label = category
            elif len(parts) >= 2:
                # For regular files with labels
                label = int(parts[1])
            else:
                raise ValueError(f"File {file_list_path} must contain labels when category is not specified")
            
            file_list.append((filename, label))
    
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
            import ast
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
                        key, value = line.split(':', 1)
                        key, value = key.strip(), value.strip()
                    else:
                        parts = line.split()
                        if len(parts) >= 2:
                            key, value = parts[0], ' '.join(parts[1:])
                    
                    # Try to convert key to int (class index)
                    try:
                        idx = int(key)
                        class_names[idx] = value
                    except ValueError:
                        # If key is not int, assume value is index
                        try:
                            idx = int(value)
                            class_names[idx] = key
                        except ValueError:
                            print(f"Warning: Could not parse line '{line}' in class map")
                
                except Exception as e:
                    print(f"Warning: Error parsing line '{line}': {e}")
    
    return class_names


class WavDataset(Dataset):
    """
    Dataset for loading vibration WAV files.
    
    Args:
        file_list (list): List of (filename, label) tuples
        data_dir (str): Directory containing the WAV files
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
            import scipy.signal
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