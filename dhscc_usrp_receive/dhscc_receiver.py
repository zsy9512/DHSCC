#!/usr/bin/env python3
"""
DHSCC Receiver Integration Module

This module provides a unified interface for DHSCC receiver functionality,
integrating polar decoding, dequantization, and neural network classification.
Independent copy for USRP integration.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
from typing import Tuple, Optional, Dict, Any

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'dhscc_modules'))

from dhscc_modules.dhscc_polar_decoder import Polar5GEncoder, Polar5GDecoder
from dhscc_modules.dhscc_model import ReceiverNet


class DHSCC_Receiver:
    """
    Integrated DHSCC receiver for polar decoding, dequantization, and classification.

    This class combines 5G polar decoding with neural network-based dequantization
    and vibration signal classification for the DHSCC system.
    """

    def __init__(self,
                 polar_config: Dict[str, Any],
                 model_checkpoint: str,
                 intermediate_dim: int = 128,
                 num_classes: int = 10,
                 device: str = 'auto',
                 use_5g_decoder: bool = False):
        """
        Initialize the DHSCC receiver.

        Args:
            polar_config: Configuration for polar decoder
            model_checkpoint: Path to trained receiver model checkpoint
            intermediate_dim: Dimension of intermediate representation
            num_classes: Number of output classes
            device: Device to run on ('auto', 'cpu', 'cuda')
            use_5g_decoder: Whether to use 5G compliant polar decoder
        """
        # Set device
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"DHSCC Receiver using device: {self.device}")

        # Configuration - set these first
        self.intermediate_dim = intermediate_dim
        self.num_classes = num_classes
        self.use_5g_decoder = use_5g_decoder

        # Initialize polar decoder
        self.polar_decoder = self._init_polar_decoder(polar_config)

        # Initialize quantizer for dequantization
        self.quantizer = self._init_quantizer()

        # Initialize neural network receiver
        self.nn_receiver = self._init_nn_receiver(model_checkpoint, intermediate_dim, num_classes)

        print("✓ DHSCC Receiver initialized successfully")

    def _init_polar_decoder(self, polar_config: Dict[str, Any]):
        """Initialize the polar decoder."""
        print("Initializing polar decoder...")
        
        k_polar = polar_config.get('k_polar', 128)
        n_polar = polar_config.get('n_polar', 256)
        channel_type = polar_config.get('channel_type', 'uplink')

        # Create encoder first (needed for decoder initialization)
        polar_encoder = Polar5GEncoder(
            k=k_polar,
            n=n_polar,
            channel_type=channel_type,
            verbose=False
        )

        # Create decoder using the encoder
        polar_decoder = Polar5GDecoder(
            enc_polar=polar_encoder,
            dec_type="SC",  # Use SC decoder
            return_crc_status=False  # Don't return CRC status for DHSCC
        )

        return polar_decoder.to(self.device)

    def _init_quantizer(self):
        """Initialize the quantizer for dequantization."""
        print("Initializing quantizer...")

        from dhscc_modules.dhscc_utils import load_quantizer_for_receiver

        # Create args object with necessary parameters
        class QuantizerArgs:
            def __init__(self, intermediate_dim):
                self.intermediate_dim = intermediate_dim
                self.quantizer_n_codes_group = 8  # 8 groups
                self.quantizer_bins = 16  # Match the 4-bit indices (2^4 = 16)
                self.quantizer_n_q = 32  # 8 groups * 4 quantizers per group = 32 total

        quantizer_args = QuantizerArgs(self.intermediate_dim)

        quantizer = load_quantizer_for_receiver(quantizer_args, self.device)
        return quantizer

    def _init_nn_receiver(self, checkpoint_path: str, intermediate_dim: int, num_classes: int) -> ReceiverNet:
        """Initialize the neural network receiver."""
        print("Initializing neural network receiver...")

        # Create model arguments
        class ModelArgs:
            def __init__(self, intermediate_dim, num_classes):
                self.intermediate_dim = intermediate_dim
                self.num_classes = num_classes

        model_args = ModelArgs(intermediate_dim, num_classes)

        # Initialize model
        receiver_model = ReceiverNet(model_args, num_classes).to(self.device)

        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load receiver model state
        try:
            receiver_model.load_state_dict(checkpoint)
            print(f"✓ Loaded checkpoint as complete state_dict from: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            raise

        # Set to evaluation mode
        receiver_model.eval()
        return receiver_model

    def decode_and_classify(self,
                           received_llrs: torch.Tensor,
                           channel_noise: float = 0.0,
                           selected_indices_shape: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode received LLRs and perform classification following the correct DHSCC flow.

        Args:
            received_llrs: Received LLR values [batch_size, n]
            channel_noise: Channel noise level (scalar)
            selected_indices_shape: Shape of the selected indices for binary_to_indices conversion

        Returns:
            Tuple of (decoded_bits, classification_logits)
        """
        with torch.no_grad():
            # Ensure input is on correct device
            received_llrs = received_llrs.to(self.device)


            # Step 1: Polar decode the received LLRs
            decoded_bits = self.polar_decoder(received_llrs)  # [batch_size, k] where k=128 for rate-1/2
            
            # Step 2: For rate-1/2 Polar code, no CRC bits to remove
            info_bits = decoded_bits  # Direct use of decoded bits
            
            # Step 3: Convert decoded bits to quantizer indices
            if selected_indices_shape is None:
                # Default shape - this should be provided by the caller
                # For now, assume a reasonable default based on typical DHSCC configuration
                # With 8 groups and 4 quantizers per group (32 total), each group has 4 quantizers
                selected_indices_shape = (received_llrs.shape[0], 8, 4)  # batch_size, n_groups, n_q_per_group
           
            from dhscc_modules.dhscc_utils import binary_to_indices
            received_indices = binary_to_indices(info_bits, selected_indices_shape, bits_per_index=4)
            #print(f"[调试] received_indices shape: {received_indices}")
            # Check if all indices are the same
            # Step 3: Dequantize using the quantizer
            reconstructed_features = self.quantizer.decode(received_indices)
            # Check if all features are the same

            # Step 4: Prepare inputs for neural network classification
            batch_size = reconstructed_features.shape[0]

            # Create target shape for reconstruction (this should match sender's output shape)
            # For vibration signals, typically [batch_size, 512, 8]
            target_shape = (batch_size, 512, 8)

            # Convert noise to tensor
            noise_tensor = torch.tensor([channel_noise], dtype=torch.float32, device=self.device)
            noise_tensor = noise_tensor.expand(batch_size, 1)

            # Step 5: Neural network processing and classification
            classification_logits = self.nn_receiver(
                reconstructed_features,
                noise_tensor,
                target_shape
            )  # [batch_size, num_classes]
            
            
            # Check prediction distribution
            predictions = torch.argmax(classification_logits, dim=1)
            # unique_preds = torch.unique(predictions)
            # #print(f"[调试] 预测类别分布: {torch.bincount(predictions, minlength=10).cpu().numpy()}")
            # if len(unique_preds) == 1:
            #     print(f"[警告] 所有预测都是类别 {unique_preds[0].item()}！")

        return predictions 

    def get_predictions(self,
                       received_llrs: torch.Tensor,
                       channel_noise: float = 0.0,
                       selected_indices_shape: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get predictions from received LLRs.

        Args:
            received_llrs: Received LLR values [batch_size, n]
            channel_noise: Channel noise level (scalar)
            selected_indices_shape: Shape of the selected indices for binary_to_indices conversion

        Returns:
            Tuple of (decoded_bits, predictions, confidence_scores)
        """
        decoded_bits, logits = self.decode_and_classify(received_llrs, channel_noise, selected_indices_shape)

        # Get predictions and confidence scores
        predictions = torch.argmax(logits, dim=1)
        confidence_scores = torch.softmax(logits, dim=1)

        return decoded_bits, predictions, confidence_scores

    def process_frame(self,
                     frame_data: torch.Tensor,
                     frame_index: int,
                     channel_noise: float = 0.0) -> Dict[str, Any]:
        """
        Process a single frame for the USRP integration.

        Args:
            frame_data: Frame data as numpy array [5, 256] LLR values
            frame_index: Frame index for tracking
            channel_noise: Channel noise level

        Returns:
            Dictionary with processing results
        """

        # Convert numpy array to torch tensor if needed
        if isinstance(frame_data, np.ndarray):
            llr_tensor = torch.from_numpy(frame_data).float()
        else:
            llr_tensor = frame_data
        #print(f"[DHSCC] 处理帧 {frame_index}，形状: {llr_tensor.shape}")
        # Define selected_indices_shape based on DHSCC configuration
        # batch_size=5 (5 polar code blocks), n_groups=2, n_q_per_group=4
        selected_indices_shape = (5, 8, 4)

        # Process through receiver
        decoded_bits, predictions, confidence_scores = self.get_predictions(
            llr_tensor, channel_noise, selected_indices_shape
        )

        # Extract results for all batch samples
        predictions_np = predictions.cpu().numpy()
        confidence_scores_np = confidence_scores.cpu().numpy()
        
        # Get predictions and confidences for all samples in the batch
        batch_predictions = []
        batch_confidences = []
        for i in range(len(predictions_np)):
            pred = predictions_np[i]
            conf = confidence_scores_np[i][pred]
            batch_predictions.append(int(pred))
            batch_confidences.append(float(conf))
        
        decoded_bits_np = decoded_bits.cpu().numpy()  # [5, 128]
        print(f"[DHSCC] 帧 {frame_index} 解码完成，分类结果: {batch_predictions}")
        #print(f"[DHSCC] 对应置信度: {[f'{c:.3f}' for c in batch_confidences]}")
        
        return {
            'frame_index': frame_index,
            'decoded_bits': decoded_bits_np,
            'predictions': batch_predictions,  # All predictions in batch
            'confidences': batch_confidences,  # All confidences in batch
            'all_confidences': confidence_scores_np  # Full confidence matrix [5, 10]
        }


def create_dhscc_receiver(polar_config: Dict[str, Any],
                         model_checkpoint: str,
                         intermediate_dim: int = 128,
                         num_classes: int = 10) -> DHSCC_Receiver:
    """
    Factory function to create a DHSCC receiver.

    Args:
        polar_config: Configuration for polar decoder
        model_checkpoint: Path to trained receiver model checkpoint
        intermediate_dim: Dimension of intermediate representation
        num_classes: Number of output classes

    Returns:
        Configured DHSCC_Receiver instance
    """
    return DHSCC_Receiver(
        polar_config=polar_config,
        model_checkpoint=model_checkpoint,
        intermediate_dim=intermediate_dim,
        num_classes=num_classes,
        use_5g_decoder=True  # Always use 5G decoder for USRP integration
    )


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    polar_config = {
        'frozen_pos': np.array([0, 1, 2, 3]),  # Example frozen positions
        'n_polar': 1024,
        'k_polar': 512,
        'channel_type': 'uplink'
    }

    # Create receiver (would need actual checkpoint path)
    try:
        receiver = create_dhscc_receiver(
            polar_config=polar_config,
            model_checkpoint="path/to/checkpoint.pth",
            intermediate_dim=128,
            num_classes=10
        )
        print("DHSCC Receiver created successfully")
    except Exception as e:
        print(f"Error creating receiver: {e}")