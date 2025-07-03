"""
Wireless communication module for VL-VFE (Variable-Length Variable Feature Encoder) model.

This module implements a wireless codec that simulates the transmission of 
quantized features through various wireless communication channels with different 
levels of noise and optional channel coding.

Key features:
- Digital modulation using QAM
- Support for both uncoded and channel-coded transmission
- Support for variable-length feature transmission
- Bit error rate (BER) calculation
"""

import torch
import numpy as np
from wireless_comm.uncoded_system import UncodedSystemAWGN
from wireless_comm.code_system import CodedSystemAWGN

class WirelessCodec:
    """
    A class for encoding, transmitting, and decoding data over simulated wireless channels
    using QAM modulation and optional channel coding.
    
    This codec supports both uncoded and polar-coded transmission modes and can adapt 
    transmission parameters based on the channel conditions.
    """
    
    def __init__(self, n, k, num_bits_per_symbol=4, device="cuda"):
        """
        Initialize the WirelessCodec.
        
        Args:
            n (int): Code length (must be a power of 2)
            k (int): Information bits (payload size)
            bits_per_index (int): Default number of bits used to represent each quantizer index
            device (str): Computing device ('cpu' or 'cuda')
        """
        self.n = n  # code length
        self.k = k  # information bits
        self.rate = k / n  # coding rate
        self.device = device
        
        # Initialize communication components
        self.num_bits_per_symbol =num_bits_per_symbol   # 16-QAM modulation
        
        # Uncoded communication system (direct modulation without channel coding)
        self.uncoded_system = UncodedSystemAWGN(
            bits_per_symbol=self.num_bits_per_symbol, 
            device=device
        )
        
        # Initialize multiple coded systems for different payload sizes
        # Each system corresponds to a different number of active feature groups
        self.system = [
            CodedSystemAWGN(
                k=self.k-i*16,  # Adjusting payload size based on number of groups
                n=self.n, 
                bits_per_symbol=self.num_bits_per_symbol, 
                device=device,
                verbose=False
            ) for i in range(8)
        ]

    def indices_to_binary(self, indices, bits_per_index=None):
        """
        Convert quantizer indices to binary representation for wireless transmission.
        Uses vectorized operations for improved performance.
        
        Args:
            indices (torch.Tensor): Tensor of quantizer indices [batch_size, num_quantizers, ...]
            bits_per_index (int, optional): Number of bits used to represent each index
            
        Returns:
            torch.Tensor: Binary representation of indices (0s and 1s)
        """
        if bits_per_index is None:
            bits_per_index = self.bits_per_index
            
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, device=self.device)
        
        # Ensure indices are within valid range for the specified bit depth
        max_value = 2**bits_per_index - 1
        if torch.max(indices).item() > max_value:
            raise ValueError(f"Index value exceeds maximum possible value for {bits_per_index} bits")
        
        # Get original shape to reshape back later
        original_shape = indices.shape
        
        # Flatten indices for batch processing
        flat_indices = indices.reshape(-1)
        
        # Create a batch of powers of 2 for binary conversion: [2^(bits-1), 2^(bits-2), ..., 2^0]
        powers = torch.pow(2, torch.arange(bits_per_index-1, -1, -1, device=indices.device))
        
        # Vectorized approach:
        # 1. Expand flat_indices to match bit representation
        expanded_indices = flat_indices.unsqueeze(-1).expand(-1, bits_per_index)
        
        # 2. Integer division and modulo operations to get binary digits
        binary = (expanded_indices // powers) % 2
        
        # Reshape to match original dimensions with added bit dimension
        return binary.reshape(original_shape + (bits_per_index,))

    def binary_to_indices(self, binary, bits_per_index=None):
        """
        Convert binary representation back to quantizer indices.
        Uses vectorized operations for improved performance.
        
        Args:
            binary (torch.Tensor): Binary tensor with shape [..., bits_per_index]
            bits_per_index (int, optional): Number of bits used to represent each index
            
        Returns:
            torch.Tensor: Reconstructed indices tensor
        """
        if bits_per_index is None:
            bits_per_index = self.bits_per_index
            
        if not isinstance(binary, torch.Tensor):
            binary = torch.tensor(binary, device=self.device)
        
        # Get original shape without the bits dimension
        original_shape = binary.shape[:-1]
        
        # Flatten binary data for batch processing
        flat_binary = binary.reshape(-1, bits_per_index)
        
        # Create power of 2 vector: [2^(bits-1), 2^(bits-2), ..., 2^0]
        powers = torch.pow(2, torch.arange(bits_per_index-1, -1, -1, device=binary.device))
        
        # Element-wise multiplication and sum for fast conversion
        indices = torch.sum(flat_binary * powers, dim=1)
        
        # Reshape back to original dimensions (without the bits_per_index dimension)
        return indices.reshape(original_shape)

    def calculate_bit_error_rate(self, original, received):
        """
        Calculate bit error rate (BER) between original and received binary data.
        
        Args:
            original (torch.Tensor): Original binary data
            received (torch.Tensor): Received binary data
            
        Returns:
            float: Bit error rate (proportion of bits that were corrupted)
        """
        if original.shape != received.shape:
            raise ValueError("Shape mismatch between original and received data")
        
        # Count bit errors
        errors = (original != received).sum().float()
        total_bits = original.numel()
        
        return errors / total_bits

    def transmit(self, indices, snr_db=None, bits_per_index=None, group_to_use=8,use_coding=True):
        """
        Complete transmission pipeline: encodes indices to binary, 
        transmits through channel, and decodes back to indices.
        
        Args:
            indices (torch.Tensor): Tensor of quantizer indices to transmit
            snr_db (float or torch.Tensor): Signal-to-noise ratio in dB
            bits_per_index (int, optional): Number of bits per index
            use_coding (bool): Whether to use channel coding or not
            
        Returns:
            dict: Dictionary containing transmission results:
                - 'decoded_indices': The reconstructed indices
                - 'binary_sent': The binary representation that was sent
                - 'binary_received': The binary data after channel effects
                - 'bit_error_rate': Measured BER
                - 'index_error_rate': Proportion of indices that were corrupted
        """
        if bits_per_index is None:
            bits_per_index = self.bits_per_index 
            
        # Ensure indices is a torch tensor on the correct device
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, device=self.device)
            
        # Convert to binary and prepare the data
        binary = self.indices_to_binary(indices, bits_per_index)

        # Get original shape for later reconstruction
        original_shape = binary.shape
        
        # Reshape binary data for transmission
        binary_reshaped = binary.reshape(original_shape[0], -1)
        
        # Pass through the appropriate system
        if use_coding:
            # Use channel coding (polar codes)
            # Select the appropriate system based on the number of groups used
            system_ind =8-group_to_use # Ensure index is within valid range
            # flip to match the reliability of 5G polar codes
            binary_reshaped = binary_reshaped.flip(-1)
            # Transmit using the selected coded system
            received_binary, noise_var = self.system[system_ind](binary_reshaped, snr_db=snr_db)
            received_binary = received_binary.flip(-1)
        else:
            # Transmit through the uncoded system (without channel coding)
            llrs, noise_var, _ = self.uncoded_system(binary_reshaped, snr_db=snr_db)  
            # Hard decision on LLRs
            received_binary = (llrs > 0).long()
            
        # Reshape to match the original binary shape
        received_binary = received_binary.reshape(original_shape)
        
        # Calculate BER
        bit_error_rate = self.calculate_bit_error_rate(binary, received_binary)
        
        # Convert back to indices
        decoded_indices = self.binary_to_indices(received_binary, bits_per_index)
        
        # Calculate index error rate
        index_errors = (indices != decoded_indices).sum().float()
        total_indices = indices.numel()
        index_error_rate = index_errors / total_indices if total_indices > 0 else 0
        
        return {
            'decoded_indices': decoded_indices,
            'binary_sent': binary,
            'binary_received': received_binary,
            'bit_error_rate': bit_error_rate,
            'index_error_rate': index_error_rate
        }
        
    def transmit_through_channel(self, indices, snr_db, bits_per_index=None, group_to_use=8,use_coding=None):
        """
        Simplified function to transmit indices through the channel and return decoded indices.
        
        Args:
            indices (torch.Tensor): Indices to transmit
            snr_db (float or torch.Tensor): SNR in dB
            bits_per_index (int, optional): Bits per index
            use_coding (bool): Whether to use channel coding
            
        Returns:
            tuple: (decoded_indices, bit_error_rate, index_error_rate)
        """
        if bits_per_index is None:
            bits_per_index = self.bits_per_index
            
        # Ensure indices are torch tensors
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, device=self.device)
        
        # Run complete transmission simulation
        result = self.transmit(indices, snr_db=snr_db, bits_per_index=bits_per_index,group_to_use=group_to_use, use_coding=use_coding)
        
        return result['decoded_indices'].long(), result['bit_error_rate'], result['index_error_rate']


def test_codec_pipeline():
    """
    Test the WirelessCodec class functionality with multiple trials at different SNR levels.
    """
    # Create a codec instance - note that n should be a power of 2 for polar codes
    codec = WirelessCodec(n=256, k=128, bits_per_index=4, device="cuda")
    
    # Test parameters
    batch_size = 16
    num_quantizers = 4
    groups = 2
    
    print(f"Testing codec with coded and uncoded transmission")
    
    # Create a sample indices tensor for initial testing
    indices = torch.randint(0, 16, (batch_size, num_quantizers, groups)).to(codec.device)
    print(f"Sample indices shape: {indices.shape}")
    
    # Test one transmission with coding to show detailed information
    print("\n--- Testing with channel coding ---")
    result_coded = codec.transmit(indices, snr_db=10.0, use_coding=True)
    print(f"Binary representation shape: {result_coded['binary_sent'].shape}")
    print(f"Measured BER: {result_coded['bit_error_rate']:.6f}")
    print(f"Decoded indices shape: {result_coded['decoded_indices'].shape}")
    print(f"Index error rate: {result_coded['index_error_rate']:.6f}")
    
    # Test one transmission without coding
    print("\n--- Testing without channel coding ---")
    result_uncoded = codec.transmit(indices, snr_db=10.0, use_coding=False)
    print(f"Measured BER: {result_uncoded['bit_error_rate']:.6f}")
    print(f"Index error rate: {result_uncoded['index_error_rate']:.6f}")
    
    # Test different SNR values
    snr_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    
    print("\nTesting different SNR values:")
    print(f"{'SNR (dB)':<10} {'Uncoded BER':<15} {'Uncoded IER':<15}")
    print("-" * 50)
    
    for snr in snr_values:
        # Generate random indices
        trial_indices = torch.randint(0, 16, (batch_size * 10, num_quantizers, groups)).to(codec.device)
        
        # Test without coding
        result_uncoded = codec.transmit(trial_indices, snr_db=snr, use_coding=False)
        uncoded_ber = result_uncoded['bit_error_rate']
        uncoded_ier = result_uncoded['index_error_rate']
        
        print(f"{snr:<10.1f} {uncoded_ber:<15.6f} {uncoded_ier:<15.6f}")


if __name__ == "__main__":
    test_codec_pipeline()