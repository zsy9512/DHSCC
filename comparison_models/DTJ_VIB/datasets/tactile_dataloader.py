"""
Tactile Data Loader for DTJ-VIB System

This module provides data loading functionality compatible with the DTJ framework,
converting 512-sample vibration signals to 2x16x16 pseudo-images for processing.
"""

import os
import sys
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional
import logging
from pathlib import Path

# Add the utils path for existing data utilities
utils_path = os.path.join(os.path.dirname(__file__), '../../../utils')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

try:
    from data import load_file_list, load_class_names
except ImportError:
    # Fallback: try direct import from utils directory
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../utils'))
        from data import load_file_list, load_class_names
    except ImportError:
        print("警告: 无法导入DHSCC数据工具，使用本地实现")
        
        def load_file_list(file_path):
            """加载文件列表"""
            file_list = []
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            filename = parts[0]
                            label = int(parts[1])
                            file_list.append((filename, label))
            return file_list
        
        def load_class_names(file_path):
            """加载类别名称映射"""
            class_names = {}
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            idx = int(parts[0])
                            name = parts[1]
                            class_names[idx] = name
            return class_names


class TactileSignalProcessor:
    """
    Processes raw 512-sample vibration signals into 2x16x16 pseudo-images.
    
    This class handles the conversion from 1D vibration signals to 2D pseudo-images
    that can be processed by the DTJ encoder.
    """
    
    def __init__(self, normalize: bool = True, augment: bool = False):
        """
        Initialize the signal processor.
        
        Args:
            normalize: Whether to normalize signals to [-1, 1] range
            augment: Whether to apply data augmentation
        """
        self.normalize = normalize
        self.augment = augment
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def signal_to_pseudo_image(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Convert 512-sample 1D signal to 2x16x16 pseudo-image.
        
        Args:
            signal: Input signal tensor of shape [512] or [1, 512]
            
        Returns:
            Pseudo-image tensor of shape [2, 16, 16]
        """
        # Ensure signal is 1D with 512 samples
        if signal.dim() == 2:
            signal = signal.squeeze(0)
        
        if signal.shape[0] != 512:
            raise ValueError(f"Expected 512 samples, got {signal.shape[0]}")
        
        # Reshape to 16x32 matrix
        matrix = signal.reshape(16, 32)
        
        # Split into two 16x16 channels
        channel1 = matrix[:, :16]  # First half
        channel2 = matrix[:, 16:]  # Second half
        
        # Stack to create 2-channel pseudo-image
        pseudo_image = torch.stack([channel1, channel2], dim=0)
        
        # Normalize to [-1, 1] range if requested
        if self.normalize:
            pseudo_image = self._normalize_tensor(pseudo_image)
            
        return pseudo_image
    
    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to [-1, 1] range."""
        min_val = tensor.min()
        max_val = tensor.max()
        
        if max_val > min_val:
            tensor = 2.0 * (tensor - min_val) / (max_val - min_val) - 1.0
        else:
            tensor = torch.zeros_like(tensor)
            
        return tensor
    
    def apply_augmentation(self, pseudo_image: torch.Tensor) -> torch.Tensor:
        """
        Apply simple data augmentation to pseudo-image.
        
        Args:
            pseudo_image: Input pseudo-image of shape [2, 16, 16]
            
        Returns:
            Augmented pseudo-image
        """
        if not self.augment:
            return pseudo_image
            
        # Random light noise addition (20% chance)
        if torch.rand(1) > 0.8:
            noise_level = 0.01 + 0.02 * torch.rand(1)  # 0.01-0.03 light noise level
            noise = torch.randn_like(pseudo_image) * noise_level
            pseudo_image = pseudo_image + noise
            
        # Re-normalize after augmentation to ensure [-1, 1] range
        if self.normalize:
            pseudo_image = self._normalize_tensor(pseudo_image)
            
        return pseudo_image


class VibrationDataset(Dataset):
    """
    PyTorch Dataset for vibration classification compatible with DTJ framework.
    
    This dataset loads vibration WAV files, converts them to pseudo-images, and provides
    them along with class labels for training the DTJ system.
    """
    
    def __init__(
        self,
        dataset_root: str,
        split: str = 'train',
        transform: bool = True,
        augment: bool = False,
        target_length: int = 512
    ):
        """
        Initialize the vibration dataset.
        
        Args:
            dataset_root: Path to dataset/vibcls directory
            split: 'train' or 'test'
            transform: Whether to apply signal processing transformations
            augment: Whether to apply data augmentation (only for train split)
            target_length: Expected signal length (default 512)
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform
        self.target_length = target_length
        
        # Only augment training data
        self.augment = augment and (split == 'train')
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize signal processor
        self.processor = TactileSignalProcessor(
            normalize=self.transform,
            augment=self.augment
        )
        
        # Load dataset files and class mappings
        self._load_dataset()
        
        self.logger.info(f"Loaded {len(self.file_list)} samples for {split} split")
        self.logger.info(f"Found {len(self.class_names)} classes")
    
    def _load_dataset(self):
        """Load file lists and class mappings."""
        # Load file list for the specified split
        split_file = self.dataset_root / f"{self.split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
            
        self.file_list = load_file_list(str(split_file))
        
        # Load class name mappings
        class_file = self.dataset_root / "clsmap.txt"
        if not class_file.exists():
            raise FileNotFoundError(f"Class mapping file not found: {class_file}")
            
        self.class_names = load_class_names(str(class_file))
        self.num_classes = len(self.class_names)
        
        # The load_class_names function returns {index: name} format
        # We need to handle both possible formats
        if isinstance(next(iter(self.class_names.keys())), int):
            # Format is {index: name}, keep as is for idx_to_class
            self.idx_to_class = self.class_names.copy()
            # Create name_to_idx mapping
            self.name_to_idx = {name: idx for idx, name in self.class_names.items()}
        else:
            # Format is {name: index}
            self.name_to_idx = self.class_names.copy()
            # Create idx_to_class mapping
            self.idx_to_class = {idx: name for name, idx in self.class_names.items()}
        
        # Validate file list format
        if not self.file_list:
            raise ValueError(f"No files found in {split_file}")
            
        # Check if files have labels
        if len(self.file_list[0]) < 2:
            raise ValueError("File list must contain filename and label columns")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (pseudo_image, label)
            - pseudo_image: [2, 16, 16] tensor
            - label: integer class label
        """
        filename, label = self.file_list[idx]
        
        # Load audio file
        audio_path = self.dataset_root / "vibrodata" / filename
        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
        except Exception as e:
            self.logger.error(f"Failed to load {audio_path}: {e}")
            raise
        
        # Ensure we have the right signal length
        if waveform.shape[1] != self.target_length:
            self.logger.warning(
                f"Signal length mismatch: expected {self.target_length}, "
                f"got {waveform.shape[1]} for {filename}"
            )
            # Pad or truncate as needed
            if waveform.shape[1] < self.target_length:
                padding = self.target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            else:
                waveform = waveform[:, :self.target_length]
        
        # Convert to pseudo-image
        if self.transform:
            pseudo_image = self.processor.signal_to_pseudo_image(waveform[0])
            
            # Apply augmentation if enabled
            pseudo_image = self.processor.apply_augmentation(pseudo_image)
        else:
            # Return raw signal reshaped to pseudo-image format (no normalization)
            signal = waveform[0]
            matrix = signal.reshape(16, 32)
            pseudo_image = torch.stack([matrix[:, :16], matrix[:, 16:]], dim=0)
        
        return pseudo_image, label


def get_data(dataset_name: str, batch_size: int, n_worker: int = 8, train: bool = True) -> DataLoader:
    """
    Get dataloader compatible with DTJ framework.
    
    Args:
        dataset_name: Name of the dataset ('VIB' for vibration data)
        batch_size: Batch size for the dataloader
        n_worker: Number of worker processes
        train: Whether to get training or test data
        
    Returns:
        DataLoader for the specified dataset split
    """
    if dataset_name == 'VIB':
        # Default dataset path - can be modified as needed
        dataset_root = '/home/zsy/workplace/DHSCC/dataset/vibcls'
        
        dataset = VibrationDataset(
            dataset_root=dataset_root,
            split='train' if train else 'test',
            transform=True,
            augment=train  # Only augment training data
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=n_worker,
            pin_memory=True,
            drop_last=train  # Drop last incomplete batch for consistent training
        )
        
        return dataloader
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def create_vib_dataloaders(
    dataset_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_augment: bool = True,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for vibration classification.
    
    Args:
        dataset_root: Path to dataset/vibcls directory
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        train_augment: Whether to apply augmentation to training data
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_dataloader, test_dataloader)
    """
    # Create datasets
    train_dataset = VibrationDataset(
        dataset_root=dataset_root,
        split='train',
        transform=True,
        augment=train_augment
    )
    
    test_dataset = VibrationDataset(
        dataset_root=dataset_root,
        split='test',
        transform=True,
        augment=False  # No augmentation for test data
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch for consistent training
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    """
    Test the tactile dataloader implementation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Tactile Dataloader")
    parser.add_argument(
        "--dataset_root", 
        type=str, 
        default="/home/zsy/workplace/DHSCC/dataset/vibcls",
        help="Path to vibcls dataset"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Test DTJ-style dataloader
        logger.info("Testing DTJ-style dataloader...")
        train_loader = get_data('VIB', args.batch_size, n_worker=4, train=True)
        test_loader = get_data('VIB', args.batch_size, n_worker=4, train=False)
        
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")
        
        # Test loading a batch
        logger.info("Testing batch loading...")
        batch_images, batch_labels = next(iter(train_loader))
        
        print(f"\nBatch shape: {batch_images.shape}")  # Should be [batch_size, 2, 16, 16]
        print(f"Labels shape: {batch_labels.shape}")   # Should be [batch_size]
        print(f"Label range: {batch_labels.min().item()} to {batch_labels.max().item()}")
        print(f"Signal value range: {batch_images.min().item():.3f} to {batch_images.max().item():.3f}")
        
        logger.info("DTJ-VIB dataloader test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error testing dataloader: {e}")
        raise
