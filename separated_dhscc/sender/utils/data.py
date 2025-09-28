"""
Data loading and processing functions for the vibration classification project.

This module provides functionality for:
1. Loading vibration data from WAV files
2. Managing dataset operations for training and testing
3. Handling class mappings and file lists
"""

import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset

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
            import csv
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
        
        # Load audio file
        waveform, sr = torchaudio.load(file_path)
        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        
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