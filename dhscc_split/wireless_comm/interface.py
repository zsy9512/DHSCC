#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wireless Communication Interface Module

This module provides a unified interface for different transmission modes,
integrating the functionality from wireless_communication.py and wireless_split.py
with the modular system_integration architecture.

Supported transmission modes:
- Mode 1: Direct analog transmission with importance masking
- Mode 2: Quantized analog transmission 
- Mode 3: Digital transmission without channel coding
- Mode 4: Digital transmission with channel coding

Key features:
- Unified interface for all transmission modes
- Modular sender/receiver separation
- Index to binary conversion and reverse conversion
- Integration with wireless_comm system modules
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Union, Callable
from .system_integration import WirelessCommunicationSystem

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WirelessInterface:
    """
    Unified wireless communication interface supporting multiple transmission modes.
    
    This class provides a consistent API for different communication modes,
    handling the conversion between quantized indices and binary data, and
    interfacing with the appropriate transmission systems.
    """
    
    def __init__(self, 
                 # Communication mode configuration
                 comm_mode: int = 4,
                 
                 # Coding parameters
                 n: int = 256,
                 k: int = 128,
                 channel_type: str = "uplink",
                 decoder_type: str = "SC",
                 list_size: int = 8,
                 
                 # Modulation parameters
                 bits_per_symbol: int = 2,
                 constellation_type: str = "qam",
                 normalize: bool = True,
                 demapping_method: str = "app",
                 
                 # Channel parameters
                 channel_model: str = "awgn",
                 equalization_method: str = "mrc",
                 
                 # UDP parameters (for real hardware)
                 udp_tx_host: str = "localhost",
                 udp_tx_port: int = 12345,
                 udp_rx_host: str = "localhost", 
                 udp_rx_port: int = 12346,
                 udp_timeout: float = 10.0,
                 udp_buffer_size: int = 8192,
                 
                 # System parameters
                 device: str = "cuda",
                 verbose: bool = False):
        """
        Initialize the wireless communication interface.
        
        Args:
            comm_mode: Communication mode (1-4)
                1: Direct analog transmission with importance masking
                2: Quantized analog transmission
                3: Digital transmission without channel coding
                4: Digital transmission with channel coding
            n: Code length (must be power of 2 for polar codes)
            k: Information bits
            channel_type: "uplink" or "downlink" for 5G polar coding
            decoder_type: Decoder type ('SC', 'SCL')
            list_size: List size for SCL decoder
            bits_per_symbol: Number of bits per symbol
            constellation_type: Constellation type ("qam", "bpsk", "pam")
            normalize: Whether to normalize constellation to unit power
            demapping_method: Demapping method ('app', 'maxlog')
            channel_model: Channel model ("awgn", "rayleigh", "udp")
            equalization_method: Channel equalization method
            udp_tx_host: UDP transmission host
            udp_tx_port: UDP transmission port
            udp_rx_host: UDP reception host
            udp_rx_port: UDP reception port
            udp_timeout: UDP timeout in seconds
            udp_buffer_size: UDP buffer size in bytes
            device: Computing device
            verbose: Enable verbose output
        """
        self.comm_mode = comm_mode
        self.n = n
        self.k = k
        self.device = device
        self.verbose = verbose
        self.bits_per_symbol = bits_per_symbol
        self.channel_model = channel_model
        
        # Initialize transmission systems based on mode
        if comm_mode in [3, 4]:  # Digital modes
            # For digital modes, we need both coded and uncoded systems
            self._init_digital_systems(
                n=n, k=k, channel_type=channel_type,
                decoder_type=decoder_type, list_size=list_size,
                bits_per_symbol=bits_per_symbol,
                constellation_type=constellation_type,
                normalize=normalize, demapping_method=demapping_method,
                channel_model=channel_model,
                equalization_method=equalization_method,
                udp_tx_host=udp_tx_host, udp_tx_port=udp_tx_port,
                udp_rx_host=udp_rx_host, udp_rx_port=udp_rx_port,
                udp_timeout=udp_timeout, udp_buffer_size=udp_buffer_size,
                device=device, verbose=verbose
            )
        
        if verbose:
            print(f"WirelessInterface initialized with mode {comm_mode}")
            print(f"System parameters: n={n}, k={k}, rate={k/n:.3f}")
            print(f"Device: {device}")
    
    def _init_digital_systems(self, **kwargs):
        """Initialize digital communication systems for modes 3 and 4."""
        
        # Mode 3: Uncoded system (digital transmission without channel coding)
        self.uncoded_system = WirelessCommunicationSystem(
            use_coding=False,  # No channel coding
            k=None,            # Not needed for uncoded
            n=None,            # Not needed for uncoded
            constellation_type=kwargs['constellation_type'],
            bits_per_symbol=kwargs['bits_per_symbol'],
            channel_model=kwargs['channel_model'],
            udp_tx_host=kwargs['udp_tx_host'],
            udp_tx_port=kwargs['udp_tx_port'],
            udp_rx_host=kwargs['udp_rx_host'],
            udp_rx_port=kwargs['udp_rx_port'],
            udp_timeout=kwargs['udp_timeout'],
            device=kwargs['device'],
            verbose=kwargs.get('verbose', False)
        )
        
        # Mode 4: Coded system (digital transmission with channel coding)
        self.coded_system = WirelessCommunicationSystem(
            use_coding=True,
            k=kwargs['k'],
            n=kwargs['n'],
            channel_type=kwargs['channel_type'],
            constellation_type=kwargs['constellation_type'],
            bits_per_symbol=kwargs['bits_per_symbol'],
            channel_model=kwargs['channel_model'],
            udp_tx_host=kwargs['udp_tx_host'],
            udp_tx_port=kwargs['udp_tx_port'], 
            udp_rx_host=kwargs['udp_rx_host'],
            udp_rx_port=kwargs['udp_rx_port'],
            udp_timeout=kwargs['udp_timeout'],
            device=kwargs['device'],
            verbose=kwargs.get('verbose', False)
        )
        
        # Multiple coded systems for different payload sizes (adaptive transmission)
        # This matches the original WirelessCodec logic: k = self.k - i*16
        self.adaptive_systems = []
        for i in range(13):
            # Calculate adaptive payload size (same as original implementation)
            #adaptive_k = kwargs['k'] - i * 16
            adaptive_k = 128-  (i+2)* 8
            print(f"Adaptive polar code system {i}: k={adaptive_k}, n={kwargs['n']}, k={kwargs['k']},, rate={adaptive_k / kwargs['n']:.3f}")
            # Skip if k becomes too small (maintain minimum viable code rate)
            if adaptive_k < 8: #16
                if kwargs.get('verbose', False):
                    print(f"Skipping adaptive system {i}: k={adaptive_k} too small")
                continue
                
            system = WirelessCommunicationSystem(
                use_coding=True,
                k=adaptive_k,
                n=kwargs['n'],
                channel_type=kwargs['channel_type'],
                decoder_type=kwargs.get('decoder_type', 'SC'),
                list_size=kwargs.get('list_size', 8),
                constellation_type=kwargs['constellation_type'],
                bits_per_symbol=kwargs['bits_per_symbol'],
                normalize=kwargs.get('normalize', True),
                demapping_method=kwargs.get('demapping_method', 'app'),
                channel_model=kwargs['channel_model'],
                equalization_method=kwargs.get('equalization_method', 'mrc'),
                udp_tx_host=kwargs['udp_tx_host'],
                udp_tx_port=kwargs['udp_tx_port'],
                udp_rx_host=kwargs['udp_rx_host'],
                udp_rx_port=kwargs['udp_rx_port'],
                udp_timeout=kwargs['udp_timeout'],
                udp_buffer_size=kwargs.get('udp_buffer_size', 8192),
                device=kwargs['device'],
                verbose=False  # Reduce verbosity for multiple systems
            )
            self.adaptive_systems.append(system)
            
            if kwargs.get('verbose', False):
                print(f"Adaptive polar code system {i}: k={adaptive_k}, n={kwargs['n']}, rate={adaptive_k/kwargs['n']:.3f}")
        
        if kwargs.get('verbose', False):
            print(f"Initialized {len(self.adaptive_systems)} adaptive polar code systems for group-based transmission")
    
    def indices_to_binary(self, indices: torch.Tensor, bits_per_index: int = 4) -> torch.Tensor:
        """
        Convert quantizer indices to binary representation.
        
        Supports different index shapes:
        - [batch_size, n_q] for residual VQ (flattened)
        - [batch_size, n_groups, group_size] for grouped indices
        
        Args:
            indices: Quantizer indices 
            bits_per_index: Number of bits per index
            
        Returns:
            Binary representation [batch_size, total_bits]
        """
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, device=self.device)
        else:
            indices = indices.to(self.device)
        
        # Validate index range
        max_value = 2**bits_per_index - 1
        if torch.max(indices).item() > max_value:
            raise ValueError(f"Index value exceeds maximum for {bits_per_index} bits")
        
        # Store original shape for reconstruction
        original_shape = indices.shape
        
        # Handle different input shapes
        if len(original_shape) == 2:
            # Shape: [batch_size, n_q] - residual VQ case
            batch_size, n_indices = original_shape
            flat_indices = indices.reshape(-1)
        elif len(original_shape) == 3:
            # Shape: [batch_size, n_groups, group_size] - grouped case
            batch_size = original_shape[0]
            flat_indices = indices.reshape(-1)
            n_indices = np.prod(original_shape[1:])
        else:
            raise ValueError(f"Unsupported indices shape: {original_shape}. Expected 2D or 3D tensor.")
        
        # Create powers of 2 for binary conversion - ensure same device
        powers = torch.pow(2, torch.arange(bits_per_index-1, -1, -1, device=self.device))
        
        # Vectorized binary conversion
        expanded_indices = flat_indices.unsqueeze(-1).expand(-1, bits_per_index)
        binary = (expanded_indices // powers) % 2
        
        # Reshape to [batch_size, total_bits]
        total_bits = n_indices * bits_per_index
        
        return binary.reshape(batch_size, total_bits).float()
    
    def binary_to_indices(self, binary: torch.Tensor, 
                         original_shape: Tuple, 
                         bits_per_index: int = 4) -> torch.Tensor:
        """
        Convert binary representation back to quantizer indices.
        
        Args:
            binary: Binary data [batch_size, total_bits]
            original_shape: Original shape of indices tensor
            bits_per_index: Number of bits per index
            
        Returns:
            Reconstructed indices tensor
        """
        if not isinstance(binary, torch.Tensor):
            binary = torch.tensor(binary, device=self.device)
        else:
            binary = binary.to(self.device)
        
        # Reshape binary to [batch_size * num_indices, bits_per_index]
        batch_size = binary.shape[0]
        num_indices = np.prod(original_shape[1:])
        binary_reshaped = binary.reshape(-1, bits_per_index)
        
        # Create powers of 2 for index reconstruction - ensure same device
        powers = torch.pow(2, torch.arange(bits_per_index-1, -1, -1, device=self.device))
        
        # Convert binary to indices
        indices = torch.sum(binary_reshaped * powers, dim=1)
        
        # Reshape to original shape
        return indices.reshape(original_shape).long()
    
    def prepare_transmission_data(self, indices: torch.Tensor, 
                                bits_per_index: int = 4,
                                group_to_use: int = 8) -> Dict:
        """
        Prepare quantized indices for transmission.
        
        Args:
            indices: Quantized indices [batch_size, n_q] or [batch_size, n_groups, group_size]
            bits_per_index: Number of bits per index
            group_to_use: Number of groups/indices to transmit
            
        Returns:
            Dictionary containing prepared data and metadata
        """
        original_shape = indices.shape
        
        # Handle different input shapes
        if len(original_shape) == 2:
            # Shape: [batch_size, n_q] - residual VQ case
            batch_size, n_q = original_shape
            if group_to_use < n_q:
                # Select first group_to_use indices
                transmit_indices = indices[:, :group_to_use]
            else:
                transmit_indices = indices
            total_groups = n_q
        elif len(original_shape) == 3:
            # Shape: [batch_size, n_groups, group_size] - grouped case
            if group_to_use < original_shape[1]:
                transmit_indices = indices[:, :group_to_use, :]
            else:
                transmit_indices = indices
            total_groups = original_shape[1]
        else:
            raise ValueError(f"Unsupported indices shape: {original_shape}")
            
        # Convert to binary
        binary_data = self.indices_to_binary(transmit_indices, bits_per_index)
        
        # Store metadata for reconstruction
        metadata = {
            'original_shape': transmit_indices.shape,
            'bits_per_index': bits_per_index,
            'group_to_use': group_to_use,
            'total_groups': total_groups
        }
        
        return {
            'binary_data': binary_data,
            'transmit_indices': transmit_indices,
            'metadata': metadata
        }
    
    def transmit_mode3(self, indices: torch.Tensor, 
                      snr_db: float,
                      bits_per_index: int = 4,
                      group_to_use: int = 8) -> Dict:
        """
        Transmit using Mode 3: Digital transmission without channel coding.
        
        Args:
            indices: Quantized indices to transmit
            snr_db: Signal-to-noise ratio in dB
            bits_per_index: Number of bits per index
            group_to_use: Number of groups to transmit
            
        Returns:
            Transmission results dictionary
        """
        if self.comm_mode != 3:
            raise ValueError("transmit_mode3 can only be used with comm_mode=3")
        # Prepare transmission data
        prep_data = self.prepare_transmission_data(indices, bits_per_index, group_to_use)
        binary_data = prep_data['binary_data']
        metadata = prep_data['metadata']
        
        # Ensure binary data is on the correct device
        binary_data = binary_data.to(self.device)
        received_binary= torch.zeros_like(binary_data, device=self.device)
        # Transmit through uncoded system using system_integration
        #print("Transmitting using uncoded system... at snr=",snr_db)
        llrs, noise_var, h = self.uncoded_system(binary_data, snr_db=snr_db)
        # Convert LLRs to hard decisions (like mode 4)
        received_binary = (llrs > 0).float()  # Hard decision: LLR > 0 -> bit = 1
        
        # Calculate bit error rate
        bit_errors = (binary_data != received_binary).sum().float()
        total_bits = binary_data.numel()
        ber = bit_errors / total_bits


        # Reconstruct indices
        received_indices = self.binary_to_indices(
            received_binary, metadata['original_shape'], bits_per_index
        )
        
        # Calculate index error rate
        index_errors = (prep_data['transmit_indices'] != received_indices).sum().float()
        total_indices = prep_data['transmit_indices'].numel()
        index_error_rate = index_errors / total_indices
        
        return {
            'received_indices': received_indices,
            'binary_sent': binary_data,
            'binary_received': received_binary,
            'bit_error_rate': ber,
            'index_error_rate': index_error_rate,
            'metadata': metadata
        }
    
    def transmit_mode4(self, indices: torch.Tensor,
                      snr_db: float,
                      bits_per_index: int = 4, 
                      group_to_use: int = 8,
                      use_coding: bool = True) -> Dict:
        """
        Transmit using Mode 4: Digital transmission with adaptive channel coding.
        
        Args:
            indices: Quantized indices to transmit
            snr_db: Signal-to-noise ratio in dB
            bits_per_index: Number of bits per index
            group_to_use: Number of groups to transmit
            use_coding: Whether to use polar coding (True) or uncoded transmission (False)
            
        Returns:
            Transmission results dictionary
        """
        if self.comm_mode != 4:
            raise ValueError("transmit_mode4 can only be used with comm_mode=4")
        
        # Prepare transmission data
        prep_data = self.prepare_transmission_data(indices, bits_per_index, group_to_use)
        binary_data = prep_data['binary_data']
        metadata = prep_data['metadata']
        
        # Ensure binary data is on the correct device
        binary_data = binary_data.to(self.device)
        #print(f"Transmitting {binary_data.shape[1]} bits using {'polar coding' if use_coding else 'uncoded transmission'} at SNR={snr_db} dB")
        #binary_data = binary_data.flip(-1)  # Flip for polar coding
        if use_coding:
            # Use polar coding transmission
            # Select appropriate coded system based on actual bit count
            system = self.coded_system
            current_bits = binary_data.shape[1]
            
            if hasattr(self, 'adaptive_systems') and len(self.adaptive_systems) > 0:
                # Find the system with matching k value
                selected_system = None
                for sys in self.adaptive_systems:
                    if sys.k == current_bits:
                        selected_system = sys
                        #print(f"Using adaptive polar code system with k={sys.k} for {current_bits} bits")
                        break
                
                if selected_system is not None:
                    system = selected_system
                    if self.verbose:
                        print(f"Selected adaptive polar code system: k={system.k}, n={system.n}, rate={system.k/system.n:.3f}")
                else:
                    if self.verbose:
                        print(f"No adaptive system found for k={current_bits}, using default coded system k={system.k}")
                        
                    # Check if default system can handle the data
                    if current_bits != system.k:
                        if self.verbose:
                            print(f"Warning: Data size mismatch - input {current_bits} bits, system expects {system.k} bits")
            
            else:
                # No adaptive systems, use default
                if current_bits != system.k:
                    if self.verbose:
                        print(f"Warning: Data size mismatch - input {current_bits} bits, system expects {system.k} bits")
            
            # Transmit through coded system
            try:
                
                received_bits, original_bits = system(binary_data, snr_db=snr_db)
                #received_bits = received_bits.flip(-1)
                # Calculate bit error rate using the restored original bits
                bit_errors = (binary_data != received_bits).sum().float()
                total_bits = binary_data.numel()
                ber = bit_errors / total_bits
                
                received_binary = received_bits
                
            except Exception as e:
                if self.verbose:
                    print(f"Coded transmission failed, falling back to uncoded: {e}")
                
                # Fallback to uncoded transmission
                try:
                    llrs_fallback, noise_var_fallback, h_fallback = self.uncoded_system(binary_data, snr_db=snr_db)
                    received_binary = (llrs_fallback > 0).float()  # Correct LLR to bit conversion: LLR < 0 -> bit = 1
                    
                    bit_errors = (binary_data != received_binary).sum().float()
                    total_bits = binary_data.numel()
                    ber = bit_errors / total_bits
                    
                except Exception as e2:
                    if self.verbose:
                        print(f"Fallback transmission also failed: {e2}")
                    received_binary = binary_data.clone()
                    ber = torch.tensor(0.0)
        else:
            # Pretrain phase: Use uncoded transmission (bypassing polar coding)
            try:
                llrs, noise_var, h = self.uncoded_system(binary_data, snr_db=snr_db)
                received_binary = (llrs > 0).float()  # Correct LLR to bit conversion: LLR > 0 -> bit = 1
                
                bit_errors = (binary_data != received_binary).sum().float()
                total_bits = binary_data.numel()
                ber = bit_errors / total_bits
                
            except Exception as e:
                if self.verbose:
                    print(f"Uncoded transmission failed: {e}")
                received_binary = binary_data.clone()
                ber = torch.tensor(0.0)
        
        # Transmission successful (removed verbose output)
        
        # Reconstruct indices
        received_indices = self.binary_to_indices(
            received_binary, metadata['original_shape'], bits_per_index
        )
        
        # Calculate index error rate
        index_errors = (prep_data['transmit_indices'] != received_indices).sum().float()
        total_indices = prep_data['transmit_indices'].numel()
        index_error_rate = index_errors / total_indices
        
        return {
            'received_indices': received_indices,
            'binary_sent': binary_data,
            'binary_received': received_binary,
            'bit_error_rate': ber,
            'index_error_rate': index_error_rate,
            'metadata': metadata
        }
    
    def transmit(self, indices: torch.Tensor,
                snr_db: float,
                bits_per_index: int = 4,
                group_to_use: int = 8,
                **kwargs) -> Dict:
        """
        Universal transmission interface supporting all digital modes.
        
        Args:
            indices: Quantized indices to transmit
            snr_db: Signal-to-noise ratio in dB
            bits_per_index: Number of bits per index
            group_to_use: Number of groups to transmit
            **kwargs: Additional mode-specific parameters
            
        Returns:
            Transmission results dictionary
        """
        if self.comm_mode == 3:
            return self.transmit_mode3(indices, snr_db, bits_per_index, group_to_use)
        elif self.comm_mode == 4:
            return self.transmit_mode4(indices, snr_db, bits_per_index, group_to_use)
        else:
            raise ValueError(f"Digital transmission not supported for mode {self.comm_mode}")
    
    def get_sender_interface(self):
        """
        Get sender-only interface for distributed deployment.
        
        Returns:
            WirelessSender instance
        """
        return WirelessSender(self)
    
    def get_receiver_interface(self):
        """
        Get receiver-only interface for distributed deployment.
        
        Returns:
            WirelessReceiver instance  
        """
        return WirelessReceiver(self)
    
    def get_transmission_stats(self) -> Dict:
        """
        Get transmission statistics and system information.
        
        Returns:
            Dictionary with system statistics
        """
        stats = {
            'comm_mode': self.comm_mode,
            'mode_description': self._get_mode_description(),
            'n': self.n,
            'k': self.k,
            'code_rate': self.k / self.n,
            'bits_per_symbol': self.bits_per_symbol,
            'device': self.device
        }
        
        return stats
    
    def _get_mode_description(self) -> str:
        """Get human-readable description of communication mode."""
        descriptions = {
            1: "Direct analog transmission with importance masking",
            2: "Quantized analog transmission with Gaussian noise",
            3: "Digital transmission without channel coding", 
            4: "Digital transmission with channel coding"
        }
        return descriptions.get(self.comm_mode, "Unknown mode")


class WirelessSender:
    """
    Sender-only interface for distributed wireless communication.
    
    This class encapsulates only the sender-side functionality,
    useful for distributed deployment scenarios.
    """
    
    def __init__(self, interface: WirelessInterface):
        """
        Initialize sender interface.
        
        Args:
            interface: Parent WirelessInterface instance
        """
        self.interface = interface
        self.comm_mode = interface.comm_mode
        self.device = interface.device
    
    def prepare_data(self, indices: torch.Tensor, 
                    bits_per_index: int = 4,
                    group_to_use: int = 8) -> Dict:
        """
        Prepare data for transmission (sender side).
        
        Args:
            indices: Quantized indices
            bits_per_index: Bits per index
            group_to_use: Number of groups to use
            
        Returns:
            Prepared transmission data
        """
        return self.interface.prepare_transmission_data(indices, bits_per_index, group_to_use)
    
    def encode_for_transmission(self, binary_data: torch.Tensor, 
                              metadata: Dict) -> torch.Tensor:
        """
        Encode binary data for transmission (includes channel coding if applicable).
        
        Args:
            binary_data: Binary data to encode
            metadata: Transmission metadata
            
        Returns:
            Encoded data ready for transmission
        """
        if self.comm_mode == 3:
            # No encoding for uncoded transmission
            return binary_data
        elif self.comm_mode == 4:
            # Apply polar encoding
            group_to_use = metadata.get('group_to_use', 8)
            system_idx = min(group_to_use - 1, len(self.interface.adaptive_systems) - 1)
            system = (self.interface.adaptive_systems[system_idx] 
                     if system_idx >= 0 else self.interface.coded_system)
            
            # Use transmitter only
            binary_flipped = binary_data.flip(-1)
            encoded_bits, symbols = system.transmitter(binary_flipped)
            return encoded_bits
        else:
            raise ValueError(f"Encoding not supported for mode {self.comm_mode}")


class WirelessReceiver:
    """
    Receiver-only interface for distributed wireless communication.
    
    This class encapsulates only the receiver-side functionality,
    useful for distributed deployment scenarios.
    """
    
    def __init__(self, interface: WirelessInterface):
        """
        Initialize receiver interface.
        
        Args:
            interface: Parent WirelessInterface instance
        """
        self.interface = interface
        self.comm_mode = interface.comm_mode
        self.device = interface.device
    
    def decode_from_transmission(self, received_data: torch.Tensor,
                               metadata: Dict,
                               noise_var: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode received data (includes channel decoding if applicable).
        
        Args:
            received_data: Received encoded data
            metadata: Transmission metadata
            noise_var: Noise variance (for soft decoding)
            
        Returns:
            Decoded binary data
        """
        if self.comm_mode == 3:
            # Hard decision for uncoded transmission
            if received_data.dtype == torch.float32:
                return (received_data > 0).float()
            else:
                return received_data.float()
                
        elif self.comm_mode == 4:
            # Apply polar decoding
            group_to_use = metadata.get('group_to_use', 8)
            system_idx = min(group_to_use - 1, len(self.interface.adaptive_systems) - 1)
            system = (self.interface.adaptive_systems[system_idx] 
                     if system_idx >= 0 else self.interface.coded_system)
            
            # Use receiver only
            if noise_var is not None:
                llrs, decoded_bits = system.receiver(received_data, noise_var, h=None)
            else:
                # Hard decision decoding
                decoded_bits = system.receiver.decode_bits(received_data)
            
            return decoded_bits.flip(-1)  # Flip back from polar code ordering
        else:
            raise ValueError(f"Decoding not supported for mode {self.comm_mode}")
    
    def reconstruct_indices(self, binary_data: torch.Tensor, 
                          metadata: Dict) -> torch.Tensor:
        """
        Reconstruct quantized indices from binary data.
        
        Args:
            binary_data: Decoded binary data
            metadata: Transmission metadata
            
        Returns:
            Reconstructed indices
        """
        return self.interface.binary_to_indices(
            binary_data, 
            metadata['original_shape'], 
            metadata['bits_per_index']
        )


def create_wireless_interface(args) -> WirelessInterface:
    """
    Factory function to create WirelessInterface from configuration.
    
    Args:
        args: Configuration object with wireless communication parameters
        
    Returns:
        Configured WirelessInterface instance
    """
    # Enable verbose output for debugging polar code issues if requested
    verbose = getattr(args, 'verbose', False)
    
    interface = WirelessInterface(
        comm_mode=getattr(args, 'communication_mode', 4),  # 使用正确的参数名
        n=getattr(args, 'polar_n', 128),  # 修正参数名
        k=getattr(args, 'polar_k', 64),  # 修正参数名
        #k=getattr(args, 'polar_k', 240),  # 修正参数名
        channel_type=getattr(args, 'channel_type', 'uplink'),
        decoder_type=getattr(args, 'decoder_type', 'SC'),
        list_size=getattr(args, 'list_size', 8),
        bits_per_symbol=getattr(args, 'bits_per_symbol', 2),
        constellation_type=getattr(args, 'constellation_type', 'qam'),
        normalize=getattr(args, 'normalize', True),
        demapping_method=getattr(args, 'demapping_method', 'app'),
        channel_model=getattr(args, 'channel_model', 'awgn'),
        equalization_method=getattr(args, 'equalization_method', 'mrc'),
        udp_tx_host=getattr(args, 'udp_tx_host', 'localhost'),
        udp_tx_port=getattr(args, 'udp_tx_port', 12345),  
        udp_rx_host=getattr(args, 'udp_rx_host', 'localhost'),
        udp_rx_port=getattr(args, 'udp_rx_port', 12346),
        udp_timeout=getattr(args, 'udp_timeout', 10.0),
        udp_buffer_size=getattr(args, 'udp_buffer_size', 8192),
        device=getattr(args, 'device', str(device).split(':')[0]),  # 确保设备参数正确
        verbose=verbose
    )
    
    if verbose:
        print(f"\nCreated WirelessInterface with the following configuration:")
        print(f"- Communication mode: {interface.comm_mode}")
        print(f"- Polar code parameters: n={interface.n}, k={interface.k}")
        print(f"- Number of adaptive systems: {len(interface.adaptive_systems) if hasattr(interface, 'adaptive_systems') else 0}")
        if hasattr(interface, 'adaptive_systems'):
            for i, sys in enumerate(interface.adaptive_systems):
                print(f"  System {i}: k={sys.k}, n={sys.n}, rate={sys.k/sys.n:.3f}")
    
    return interface
