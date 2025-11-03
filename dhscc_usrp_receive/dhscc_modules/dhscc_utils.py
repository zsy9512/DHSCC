"""
Utility functions for DHSCC receiver.
"""

import torch
import os
import glob
from dhscc_modules.dhscc_quantizer import VibrationVectorQuantizer


def binary_to_indices(binary, original_shape, bits_per_index=4):
    """Convert binary back to quantizer indices."""
    device = binary.device
    batch_size, total_bits = binary.shape
    indices = []
    for b in range(batch_size):
        sample_bits = binary[b]
        sample_indices = []
        for i in range(0, total_bits, bits_per_index):
            bit_chunk = sample_bits[i:i+bits_per_index]
            bit_values = bit_chunk.int().tolist()
            idx = sum(bit_val * (2**(bits_per_index-1-j)) for j, bit_val in enumerate(bit_values))
            sample_indices.append(idx)
        indices.append(sample_indices)
    indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
    return indices_tensor.view(original_shape)


def load_quantizer_for_receiver(args, device):
    """Load trained quantizer for feature reconstruction at receiver."""
    quantizer = VibrationVectorQuantizer(
        dimension=args.intermediate_dim,
        n_codes_group=args.quantizer_n_codes_group,
        bins=args.quantizer_bins,
        n_q=args.quantizer_n_q
    ).to(device)

    quantizer_path = getattr(args, 'quantizer_path', None)
    if not quantizer_path:
        # Use the correct path for the quantizer weights
        quantizer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'final_quantizer.pth')
        quantizer_path = os.path.abspath(quantizer_path)

    if quantizer_path and os.path.exists(quantizer_path):
        quantizer_checkpoint = torch.load(quantizer_path, map_location=device)

        # Try to adapt the checkpoint format if needed
        try:
            quantizer.load_state_dict(quantizer_checkpoint)
            print(f"✓ Loaded quantizer weights from {quantizer_path}")
        except RuntimeError as e:
            if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                print(f"[Warning] Checkpoint format mismatch, attempting to adapt...")
                # Try to load with adapted state dict
                adapted_state = adapt_quantizer_checkpoint(quantizer_checkpoint, quantizer.state_dict())
                if adapted_state:
                    quantizer.load_state_dict(adapted_state)
                    print(f"✓ Loaded quantizer weights with adaptation from {quantizer_path}")
                else:
                    print(f"[Warning] Could not adapt checkpoint format: {e}")
            else:
                print(f"[Warning] Could not load quantizer weights: {e}")
    else:
        print(f"[Warning] Could not find quantizer weights at {quantizer_path}")

    return quantizer


def adapt_quantizer_checkpoint(checkpoint, model_state_dict):
    """Adapt checkpoint format to match model expectations."""
    adapted = {}

    # Check if checkpoint uses the expected format
    checkpoint_keys = set(checkpoint.keys())
    model_keys = set(model_state_dict.keys())

    # If formats match, return original
    if checkpoint_keys == model_keys:
        return checkpoint

    # Try to map old format to new format
    # Old format: gvq.quantizers.X.vq.layers.Y.embedding.weight
    # New format: gvq.quantizers.X.vq.quantizers.Y.weight and gvq.quantizers.X.vq.codebooks.Y.weight

    for key in checkpoint_keys:
        if key.startswith('gvq.quantizers.') and 'layers' in key and 'embedding.weight' in key:
            # Extract indices
            parts = key.split('.')
            if len(parts) >= 7:
                group_idx = int(parts[2])  # X
                layer_idx = int(parts[5])  # Y

                # Map to new format - assume embedding weights map to codebook weights
                new_key = f'gvq.quantizers.{group_idx}.vq.codebooks.{layer_idx}.weight'
                if new_key in model_keys:
                    adapted[new_key] = checkpoint[key]

    # If we successfully adapted some keys, fill in missing ones with model defaults
    if adapted:
        for key in model_keys:
            if key not in adapted:
                adapted[key] = model_state_dict[key]
        return adapted

    return None


def load_receiver_models(args, device, num_classes):
    """Load trained receiver model."""
    # Import here to avoid circular imports
    from dhscc_usrp_receive.dhscc_modules.dhscc_model import DHSCC_ReceiverNet

    receiver_model = DHSCC_ReceiverNet(args, num_classes).to(device)
    receiver_model_path = getattr(args, 'receiver_model_path', None)
    if not receiver_model_path:
        checkpoint_dir = getattr(args, 'checkpoint_dir', './checkpoints')
        if os.path.exists(checkpoint_dir):
            subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
            training_dirs = [d for d in subdirs if 'training' in d or 'direct_polar' in d]
            if training_dirs:
                training_dirs.sort(key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)), reverse=True)
                latest_dir = training_dirs[0]
                receiver_model_path = os.path.join(checkpoint_dir, latest_dir, 'final_receiver_model.pth')

    if receiver_model_path and os.path.exists(receiver_model_path):
        receiver_checkpoint = torch.load(receiver_model_path, map_location=device)
        if 'state_dict' in receiver_checkpoint:
            receiver_model.load_state_dict(receiver_checkpoint['state_dict'])
        else:
            receiver_model.load_state_dict(receiver_checkpoint)
    receiver_model.eval()
    return receiver_model