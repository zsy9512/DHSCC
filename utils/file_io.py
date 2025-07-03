"""
File I/O utilities for DHSCC system.

This module provides utility functions for file and directory operations,
including ensuring directories exist, saving/loading models, and managing
checkpoints.
"""

import os
import torch
import json
import pickle
from pathlib import Path


def ensure_dir(directory):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory (str): Path to the directory
    """
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def save_checkpoint(state, filename, is_best=False, best_filename=None):
    """
    Save a training checkpoint.
    
    Args:
        state (dict): State dictionary containing model, optimizer, etc.
        filename (str): Path to save the checkpoint
        is_best (bool): Whether this is the best model so far
        best_filename (str): Path to save the best model
    """
    ensure_dir(os.path.dirname(filename))
    torch.save(state, filename)
    
    if is_best and best_filename:
        ensure_dir(os.path.dirname(best_filename))
        torch.save(state, best_filename)


def load_checkpoint(filename, map_location=None):
    """
    Load a training checkpoint.
    
    Args:
        filename (str): Path to the checkpoint file
        map_location: Device to map the checkpoint to
        
    Returns:
        dict: Loaded checkpoint state
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")
    
    return torch.load(filename, map_location=map_location)


def save_json(data, filename):
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save (must be JSON serializable)
        filename (str): Path to save the JSON file
    """
    ensure_dir(os.path.dirname(filename))
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filename):
    """
    Load data from a JSON file.
    
    Args:
        filename (str): Path to the JSON file
        
    Returns:
        Loaded data
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"JSON file not found: {filename}")
    
    with open(filename, 'r') as f:
        return json.load(f)


def save_pickle(data, filename):
    """
    Save data to a pickle file.
    
    Args:
        data: Data to save
        filename (str): Path to save the pickle file
    """
    ensure_dir(os.path.dirname(filename))
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filename):
    """
    Load data from a pickle file.
    
    Args:
        filename (str): Path to the pickle file
        
    Returns:
        Loaded data
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Pickle file not found: {filename}")
    
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_latest_checkpoint(checkpoint_dir, pattern="*.pth"):
    """
    Get the path to the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir (str): Directory to search for checkpoints
        pattern (str): File pattern to match
        
    Returns:
        str or None: Path to the latest checkpoint, or None if none found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = list(Path(checkpoint_dir).glob(pattern))
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time and return the latest
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    return str(latest_checkpoint)


def create_experiment_dir(base_dir, experiment_name=None):
    """
    Create a unique experiment directory.
    
    Args:
        base_dir (str): Base directory for experiments
        experiment_name (str): Optional experiment name
        
    Returns:
        str: Path to the created experiment directory
    """
    import datetime
    
    if experiment_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"exp_{timestamp}"
    
    experiment_dir = os.path.join(base_dir, experiment_name)
    ensure_dir(experiment_dir)
    
    return experiment_dir


def list_files(directory, extension=None, recursive=False):
    """
    List files in a directory with optional filtering.
    
    Args:
        directory (str): Directory to search
        extension (str): File extension to filter by (e.g., '.txt')
        recursive (bool): Whether to search recursively
        
    Returns:
        list: List of file paths
    """
    if not os.path.exists(directory):
        return []
    
    files = []
    
    if recursive:
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if extension is None or filename.endswith(extension):
                    files.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                if extension is None or filename.endswith(extension):
                    files.append(filepath)
    
    return sorted(files)


def copy_file(src, dst):
    """
    Copy a file from source to destination.
    
    Args:
        src (str): Source file path
        dst (str): Destination file path
    """
    import shutil
    
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)


def move_file(src, dst):
    """
    Move a file from source to destination.
    
    Args:
        src (str): Source file path
        dst (str): Destination file path
    """
    import shutil
    
    ensure_dir(os.path.dirname(dst))
    shutil.move(src, dst)


def get_file_size(filepath):
    """
    Get the size of a file in bytes.
    
    Args:
        filepath (str): Path to the file
        
    Returns:
        int: File size in bytes
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    return os.path.getsize(filepath)


def get_directory_size(directory):
    """
    Get the total size of a directory in bytes.
    
    Args:
        directory (str): Path to the directory
        
    Returns:
        int: Total size in bytes
    """
    total_size = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    
    return total_size
