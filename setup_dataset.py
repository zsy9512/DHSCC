"""
Dataset setup script for DHSCC Split end-to-end training.

This script helps create the required dataset structure in the dhscc_split/dataset directory.
Run this script to set up the basic dataset structure for your vibration data.
"""

import os
import argparse

def create_dataset_structure(base_dir="dataset"):
    """Create the basic dataset directory structure."""

    # Get the current directory (dhscc_split)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Create directories
    dirs_to_create = [
        base_dir,
        os.path.join(base_dir, "vibrodata"),
        os.path.join(base_dir, "vibrodata", "train"),
        os.path.join(base_dir, "vibrodata", "test")
    ]

    for dir_path in dirs_to_create:
        full_path = os.path.join(current_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")

def create_sample_files(base_dir="dataset", num_classes=10):
    """Create sample data files for testing."""

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Create class mapping file
    clsmap_path = os.path.join(current_dir, base_dir, "clsmap.txt")
    with open(clsmap_path, 'w') as f:
        for i in range(num_classes):
            f.write(f"{i} Class{i}\n")
    print(f"Created class mapping: {clsmap_path}")

    # Create sample train.txt and test.txt
    train_list = []
    test_list = []

    # Generate sample entries (you'll need to replace these with actual WAV files)
    for i in range(num_classes):
        # Add some samples for each class
        for j in range(10):  # 10 samples per class for training
            filename = f"train/class_{i}_sample_{j}.wav"
            train_list.append(f"{filename} {i}")

        for j in range(3):  # 3 samples per class for testing
            filename = f"test/class_{i}_sample_{j}.wav"
            test_list.append(f"{filename} {i}")

    # Write train.txt
    train_path = os.path.join(current_dir, base_dir, "train.txt")
    with open(train_path, 'w') as f:
        for line in train_list:
            f.write(line + '\n')
    print(f"Created train list: {train_path} ({len(train_list)} samples)")

    # Write test.txt
    test_path = os.path.join(current_dir, base_dir, "test.txt")
    with open(test_path, 'w') as f:
        for line in test_list:
            f.write(line + '\n')
    print(f"Created test list: {test_path} ({len(test_list)} samples)")

    print("\n" + "="*60)
    print("DATASET STRUCTURE CREATED")
    print("="*60)
    print("Next steps:")
    print("1. Place your WAV files in the following directories:")
    print("   - dhscc_split/dataset/vibrodata/train/  (for training data)")
    print("   - dhscc_split/dataset/vibrodata/test/   (for testing data)")
    print("2. Update train.txt and test.txt with your actual filenames")
    print("3. Ensure WAV files are mono, sampled at 2800 Hz")
    print("4. Update clsmap.txt with your actual class names")
    print("="*60)

def validate_dataset(base_dir="dataset"):
    """Validate the dataset structure."""

    from data_loader import validate_dataset_exists
    return validate_dataset_exists(base_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Setup DHSCC Split dataset structure')
    parser.add_argument('--base_dir', type=str, default="dataset",
                        help='Base directory for dataset')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes')
    parser.add_argument('--validate', action='store_true',
                        help='Validate existing dataset instead of creating')

    args = parser.parse_args()

    if args.validate:
        print("Validating dataset structure...")
        success = validate_dataset(args.base_dir)
        if success:
            print("✅ Dataset validation passed!")
        else:
            print("❌ Dataset validation failed!")
    else:
        create_dataset_structure(args.base_dir)
        create_sample_files(args.base_dir, args.num_classes)