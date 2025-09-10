"""
Tactile Dataset for DCDDM Communication System

This module implements data processing for tactile vibration signals to be used with the DCDDM 
communication pipeline. It converts 512-sample 1D vibration signals into 2x16x16 pseudo-images
suitable for the DCDDM encoder that expects 2-channel input (instead of 3-channel RGB images).

Key Features:
- Loads vibration data from the vibcls dataset format
- Converts 512x1 signals to 2x16x16 pseudo-images using reshaping and splitting
- Provides train/test split according to dataset files
- Supports data augmentation and normalization
- Compatible with DCDDM pipeline requirements (4QAM, 256bit output, 10-class classification)

Author: Generated for DHSCC Integration
Date: 2025
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
utils_path = os.path.join(os.path.dirname(__file__), '../../utils')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

try:
    from data import load_file_list, load_class_names
except ImportError:
    # Fallback: try direct import from utils directory
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../utils'))
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
    that can be processed by the DCDDM encoder. The conversion strategy:
    1. Reshape 512 samples into 16x32 matrix
    2. Split into two 16x16 channels (first half and second half)
    3. Apply normalization and optional augmentation
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
            
        # 1. Random horizontal flip (5% chance)
        if torch.rand(1) > 1:
            pseudo_image = torch.flip(pseudo_image, dims=[2])
            
        # 2. Random light noise addition (20% chance)
        if torch.rand(1) > 1:
            noise_level = 0.01 + 0.02 * torch.rand(1)  # 0.01-0.03 light noise level
            noise = torch.randn_like(pseudo_image) * noise_level
            pseudo_image = pseudo_image + noise
            
        # Re-normalize after augmentation to ensure [-1, 1] range
        if self.normalize:
            pseudo_image = self._normalize_tensor(pseudo_image)
            
        return pseudo_image


class TactileDataset(Dataset):
    """
    PyTorch Dataset for tactile vibration classification using DCDDM pipeline.
    
    This dataset loads vibration WAV files, converts them to pseudo-images, and provides
    them along with class labels for training the combined DCDDM + classifier system.
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
        Initialize the tactile dataset.
        
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
        self.logger.info(f"Found {len(self.class_names)} classes: {list(self.idx_to_class.values())}")
    
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


def create_tactile_dataloaders(
    dataset_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_augment: bool = True,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for tactile classification.
    
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
    train_dataset = TactileDataset(
        dataset_root=dataset_root,
        split='train',
        transform=True,
        augment=train_augment
    )
    
    test_dataset = TactileDataset(
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


def analyze_dataset_statistics(dataset_root: str) -> Dict:
    """
    Analyze and return statistics about the tactile dataset.
    
    Args:
        dataset_root: Path to dataset/vibcls directory
        
    Returns:
        Dictionary containing dataset statistics
    """
    # Create datasets for analysis
    train_dataset = TactileDataset(dataset_root, split='train', transform=False)
    test_dataset = TactileDataset(dataset_root, split='test', transform=False)
    
    stats = {
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'total_samples': len(train_dataset) + len(test_dataset),
        'num_classes': train_dataset.num_classes,
        'class_names': list(train_dataset.idx_to_class.values()),
        'class_distribution': {}
    }
    
    # Analyze class distribution in training set
    # Use a more efficient approach for large datasets
    print("Analyzing class distribution...")
    
    train_label_counts = {}
    test_label_counts = {}
    
    # Count training labels
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        train_label_counts[label] = train_label_counts.get(label, 0) + 1
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(train_dataset)} training samples...")
    
    # Count test labels  
    for i in range(len(test_dataset)):
        _, label = test_dataset[i]
        test_label_counts[label] = test_label_counts.get(label, 0) + 1
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(test_dataset)} test samples...")
    
    # Create distribution statistics using class index to name mapping
    for class_idx, class_name in train_dataset.idx_to_class.items():
        train_count = train_label_counts.get(class_idx, 0)
        test_count = test_label_counts.get(class_idx, 0)
        stats['class_distribution'][class_name] = {
            'index': class_idx,
            'train': train_count,
            'test': test_count,
            'total': train_count + test_count
        }
    
    # Calculate signal statistics from a sample
    sample_signal, _ = train_dataset[0]
    stats['signal_shape'] = list(sample_signal.shape)  # Should be [2, 16, 16]
    
    return stats


if __name__ == "__main__":
    """
    Test the tactile dataset implementation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Tactile Dataset")
    parser.add_argument(
        "--dataset_root", 
        type=str, 
        default="/home/zsy/workplace/DHSCC/dataset/vibcls",
        help="Path to vibcls dataset"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--analyze", action="store_true", help="Analyze dataset statistics")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        if args.analyze:
            logger.info("Analyzing dataset statistics...")
            stats = analyze_dataset_statistics(args.dataset_root)
            
            print("\n=== TACTILE DATASET STATISTICS ===")
            print(f"Training samples: {stats['train_samples']}")
            print(f"Test samples: {stats['test_samples']}")
            print(f"Total samples: {stats['total_samples']}")
            print(f"Number of classes: {stats['num_classes']}")
            print(f"Signal shape (pseudo-image): {stats['signal_shape']}")
            
            print("\nClass distribution:")
            for class_name, counts in stats['class_distribution'].items():
                print(f"  {class_name} (idx={counts['index']}): Train={counts['train']}, Test={counts['test']}, Total={counts['total']}")
        
        # Test dataloader creation
        logger.info("Creating dataloaders...")
        train_loader, test_loader = create_tactile_dataloaders(
            args.dataset_root,
            batch_size=args.batch_size
        )
        
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")
        
        # Test loading a batch
        logger.info("Testing batch loading...")
        batch_images, batch_labels = next(iter(train_loader))
        
        print(f"\nBatch shape: {batch_images.shape}")  # Should be [batch_size, 2, 16, 16]
        print(f"Labels shape: {batch_labels.shape}")   # Should be [batch_size]
        print(f"Label range: {batch_labels.min().item()} to {batch_labels.max().item()}")
        print(f"Signal value range: {batch_images.min().item():.3f} to {batch_images.max().item():.3f}")
        
        # Show some sample information
        print(f"Unique labels in batch: {sorted(batch_labels.unique().tolist())}")
        print(f"Label distribution: {[(label.item(), (batch_labels == label).sum().item()) for label in batch_labels.unique()]}")
        
        logger.info("Tactile dataset test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error testing tactile dataset: {e}")
        raise
