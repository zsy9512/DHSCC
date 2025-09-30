#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wireless Channel Module

This module implements various wireless channel models including:
- AWGN (Additive White Gaussian Noise) channel
- Rayleigh fading channel
- UDP channel for real USRP hardware transmission
- Channel parameter estimation and equalization
"""

import torch
import numpy as np
import socket
import struct
import time
import threading
import queue
from typing import Tuple, Optional, Union


class WirelessChannel:
    """
    Unified wireless channel supporting AWGN, Rayleigh fading, and UDP hardware transmission
    """
    
    def __init__(self,
                 channel_model: str = "awgn",
                 device: str = "cpu",
                 # UDP channel parameters
                 udp_tx_host: str = "localhost",
                 udp_tx_port: int = 12345,
                 udp_rx_host: str = "localhost", 
                 udp_rx_port: int = 12346,
                 udp_timeout: float = 10.0,
                 udp_buffer_size: int = 8192,
                 verbose: bool = False):
        """
        Initialize the wireless channel
        
        Args:
            channel_model: Channel model type ('awgn', 'rayleigh', or 'udp')
            device: Device to run computations on
            udp_tx_host: UDP transmission host address
            udp_tx_port: UDP transmission port
            udp_rx_host: UDP reception host address  
            udp_rx_port: UDP reception port
            udp_timeout: UDP socket timeout in seconds
            udp_buffer_size: UDP buffer size in bytes
            verbose: Whether to display verbose output
        """
        self.channel_model = channel_model.lower()
        self.device = device
        self.verbose = verbose
        
        if self.channel_model not in ['awgn', 'rayleigh', 'udp']:
            raise ValueError(f"Unsupported channel model: {channel_model}")
        # UDP channel parameters
        if self.channel_model == 'udp':
            self.udp_tx_host = udp_tx_host
            self.udp_tx_port = udp_tx_port
            self.udp_rx_host = udp_rx_host
            self.udp_rx_port = udp_rx_port
            self.udp_timeout = udp_timeout
            self.udp_buffer_size = udp_buffer_size
            
            # Initialize UDP sockets
            self._init_udp_sockets()
    
    def _init_udp_sockets(self):
        """Initialize UDP sockets for transmission and reception"""
        try:
            # Transmission socket
            self.tx_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.tx_socket.settimeout(self.udp_timeout)
            
            # Reception socket
            self.rx_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.rx_socket.settimeout(self.udp_timeout)
            self.rx_socket.bind((self.udp_rx_host, self.udp_rx_port))
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize UDP sockets: {e}")
    
    def _close_udp_sockets(self):
        """Close UDP sockets"""
        if hasattr(self, 'tx_socket'):
            self.tx_socket.close()
        if hasattr(self, 'rx_socket'):
            self.rx_socket.close()
    
    def __del__(self):
        """Destructor to clean up UDP sockets"""
        if self.channel_model == 'udp':
            self._close_udp_sockets()
    
    def add_awgn(self, 
                 symbols: torch.Tensor, 
                 snr_db: Optional[float] = None, 
                 ebn0_db: Optional[float] = None, 
                 bits_per_symbol: Optional[int] = None,
                 code_rate: Optional[float] = None) -> Tuple[torch.Tensor, float]:
        """
        Add Additive White Gaussian Noise to symbols
        
        Args:
            symbols: Real or complex symbols
            snr_db: Signal-to-Noise ratio in dB
            ebn0_db: Energy per bit to noise power spectral density ratio in dB
            bits_per_symbol: Number of bits per symbol (required for Eb/N0)
            code_rate: Code rate (required for coded systems with Eb/N0)
            
        Returns:
            tuple: (noisy_symbols, noise_power)
        """
        # Check if symbols are complex or real
        is_complex = symbols.is_complex()
        
        # Calculate symbol energy (assuming normalized constellation)
        symbol_energy = torch.mean(torch.abs(symbols)**2).item()
        
        # Calculate noise power based on provided parameter
        if snr_db is not None:
            # Convert SNR from dB to linear scale
            snr_linear = 10.0**(snr_db/10.0)
            noise_power = symbol_energy / snr_linear
        elif ebn0_db is not None:
            if bits_per_symbol is None:
                raise ValueError("bits_per_symbol must be provided when using ebn0_db")
            
            # Convert Eb/N0 from dB to linear scale
            ebn0_linear = 10.0**(ebn0_db/10.0)
            
            # Calculate bit energy
            bit_energy = symbol_energy / bits_per_symbol
            
            # Apply code rate if provided (for coded systems)
            if code_rate is not None:
                bit_energy = bit_energy * code_rate
            
            # Calculate noise power based on bit energy and Eb/N0
            noise_power = bit_energy / ebn0_linear
        else:
            raise ValueError("Either snr_db or ebn0_db must be provided")
        # Generate appropriate noise for complex and real symbols
        if is_complex:
            # Complex signal noise components - each dimension gets half the power
            noise_var_per_dim = noise_power / 2
            if not torch.is_tensor(noise_var_per_dim):
                noise_var_per_dim = torch.tensor(noise_var_per_dim, device=symbols.real.device)
            else:
                noise_var_per_dim = noise_var_per_dim.to(symbols.real.device)
            noise_real = torch.randn_like(symbols.real) * torch.sqrt(noise_var_per_dim)
            noise_imag = torch.randn_like(symbols.imag) * torch.sqrt(noise_var_per_dim)
            noise = torch.complex(noise_real, noise_imag)

        else:
            # Real signal noise components - BPSK uses all noise power for one dimension
            noise = torch.randn_like(symbols) * np.sqrt(noise_power)
        
        # Add noise to symbols
        noisy_symbols = symbols + noise
        
        return noisy_symbols, noise_power
    
    def add_rayleigh_channel(self, 
                           symbols: torch.Tensor, 
                           snr_db: Optional[float] = None, 
                           ebn0_db: Optional[float] = None, 
                           bits_per_symbol: Optional[int] = None,
                           code_rate: Optional[float] = None) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Add Rayleigh fading channel with AWGN
        
        Args:
            symbols: Real or complex symbols
            snr_db: Signal-to-Noise ratio in dB
            ebn0_db: Energy per bit to noise power spectral density ratio in dB
            bits_per_symbol: Number of bits per symbol (required for Eb/N0)
            code_rate: Code rate (required for coded systems with Eb/N0)
            
        Returns:
            tuple: (received_symbols, noise_power, channel_coefficients)
        """
        # Check if symbols are complex or real
        is_complex = symbols.is_complex()
        #print(f"Adding Rayleigh fading channel to symbols: {symbols.shape}, is_complex: {is_complex}")
        # Convert real symbols to complex for Rayleigh fading processing
        if not is_complex:
            symbols_complex = torch.complex(symbols, torch.zeros_like(symbols))
        else:
            symbols_complex = symbols

        # Generate Rayleigh fading coefficients (complex Gaussian random variables, variance=1)
        h_real = torch.randn_like(symbols_complex.real) / np.sqrt(2)
        h_imag = torch.randn_like(symbols_complex.imag) / np.sqrt(2)
        h = torch.complex(h_real, h_imag)
        
        # Apply Rayleigh fading
        faded_symbols = h * symbols_complex
        
        # Calculate symbol energy (assuming normalized constellation)
        symbol_energy = 1.0  # Maintain unit energy assumption
        
        # Calculate noise power based on provided parameter
        if snr_db is not None:
            # Convert SNR from dB to linear scale
            snr_linear = 10.0**(snr_db/10.0)
            noise_power = symbol_energy / snr_linear
        elif ebn0_db is not None:
            if bits_per_symbol is None:
                raise ValueError("bits_per_symbol must be provided when using ebn0_db")
            
            # Convert Eb/N0 from dB to linear scale
            ebn0_linear = 10.0**(ebn0_db/10.0)
            
            # Calculate bit energy
            bit_energy = symbol_energy / bits_per_symbol
            
            # Apply code rate if provided
            if code_rate is not None:
                bit_energy = bit_energy * code_rate
            
            # Calculate noise power
            noise_power = bit_energy / ebn0_linear
        else:
            raise ValueError("Either snr_db or ebn0_db must be provided")
        
        # Generate AWGN noise for complex symbols
        noise_var_per_dim = noise_power / 2
        # 保证 noise_var_per_dim 是 tensor 并在正确设备
        if not torch.is_tensor(noise_var_per_dim):
            noise_var_per_dim = torch.tensor(noise_var_per_dim, device=faded_symbols.device, dtype=faded_symbols.dtype)
        else:
            noise_var_per_dim = noise_var_per_dim.to(faded_symbols.device)
        noise_std = torch.sqrt(noise_var_per_dim.real)  # 保证 noise_std 是实数
        noise_real = torch.randn_like(faded_symbols.real) * noise_std
        noise_imag = torch.randn_like(faded_symbols.imag) * noise_std
        noise = torch.complex(noise_real, noise_imag)
        
        # Add noise to faded symbols
        received_symbols = faded_symbols + noise
        
        # If original symbols were real, extract real part
        if not is_complex:
            received_symbols = received_symbols.real
            h = h.real  # Return real channel coefficients for real symbols
        
        return received_symbols, noise_power, h
    
    def apply_channel(self, 
                     symbols: torch.Tensor, 
                     snr_db: Optional[float] = None, 
                     ebn0_db: Optional[float] = None, 
                     bits_per_symbol: Optional[int] = None,
                     code_rate: Optional[float] = None) -> Tuple[torch.Tensor, float, Optional[torch.Tensor]]:
        """
        Apply the selected channel model to symbols
        
        Args:
            symbols: Input symbols
            snr_db: Signal-to-Noise ratio in dB
            ebn0_db: Energy per bit to noise power spectral density ratio in dB
            bits_per_symbol: Number of bits per symbol
            code_rate: Code rate for coded systems
            
        Returns:
            tuple: (received_symbols, noise_power, channel_coefficients)
        """
        if self.channel_model == "awgn":
            received_symbols, noise_power = self.add_awgn(
                symbols, snr_db=snr_db, ebn0_db=ebn0_db, 
                bits_per_symbol=bits_per_symbol, code_rate=code_rate
            )
            return received_symbols, noise_power, None
        
        elif self.channel_model == "rayleigh":
            #print(f"Applying Rayleigh fading channel to symbols: {symbols.shape}")
            received_symbols, noise_power, h = self.add_rayleigh_channel(
                symbols, snr_db=snr_db, ebn0_db=ebn0_db,
                bits_per_symbol=bits_per_symbol, code_rate=code_rate
            )
            return received_symbols, noise_power, h
        
        elif self.channel_model == "udp":
            # UDP channel expects bits, not symbols - this should not be called for UDP
            raise ValueError("UDP channel should be called with apply_udp_channel method")
        
        else:
            raise ValueError(f"Unsupported channel model: {self.channel_model}")
    
    def channel_equalization(self, 
                           received_symbols: torch.Tensor, 
                           h: torch.Tensor, 
                           equalization_method: str = "mrc",
                           noise_var: Optional[float] = None) -> torch.Tensor:
        """
        Apply channel equalization for both AWGN and Rayleigh fading channels
        
        Args:
            received_symbols: Received symbols after channel
            h: Channel coefficients
            equalization_method: Equalization method ('zf', 'mmse', 'mrc', or 'identity')
            noise_var: Noise variance (required for MMSE)
            
        Returns:
            Equalized symbols
        """
        if h is None:
            return received_symbols
        
        if equalization_method.lower() == "identity":
            # Identity equalization (no equalization)
            return received_symbols
        
        elif equalization_method.lower() == "zf":
            # Zero-Forcing equalization: y_eq = y / h
            h_magnitude = torch.abs(h)
            h_safe = torch.where(h_magnitude < 1e-8, 
                               torch.sgn(h) * 1e-8, h)
            equalized_symbols = received_symbols / h_safe
        
        elif equalization_method.lower() == "mrc":
            # Maximum Ratio Combining: y_eq = h* * y / |h|^2
            h_conj = torch.conj(h)
            h_magnitude_squared = torch.abs(h)**2
            h_mag_sq_safe = torch.where(h_magnitude_squared < 1e-8, 
                                      1e-8, h_magnitude_squared)
            equalized_symbols = (h_conj * received_symbols) / h_mag_sq_safe
        
        elif equalization_method.lower() == "mmse":
            # MMSE equalization: y_eq = h* * y / (|h|^2 + sigma^2)
            if noise_var is None:
                noise_var = 0.1  # Default noise variance estimate
            
            h_conj = torch.conj(h)
            h_magnitude_squared = torch.abs(h)**2
            denominator = h_magnitude_squared + noise_var
            equalized_symbols = (h_conj * received_symbols) / denominator
        
        else:
            raise ValueError(f"Unknown equalization method: {equalization_method}")
        
        return equalized_symbols
    
    def estimate_channel_from_pilots(self, 
                                   received_pilots: torch.Tensor, 
                                   transmitted_pilots: torch.Tensor) -> torch.Tensor:
        """
        Estimate channel coefficients from pilot symbols (for advanced systems)
        
        Args:
            received_pilots: Received pilot symbols
            transmitted_pilots: Known transmitted pilot symbols
            
        Returns:
            Estimated channel coefficients
        """
        # Simple LS channel estimation: h_est = y_pilot / x_pilot
        h_est = received_pilots / (transmitted_pilots + 1e-8)  # Avoid division by zero
        return h_est
    
    def get_channel_capacity(self, snr_db: float) -> float:
        """
        Calculate theoretical channel capacity
        
        Args:
            snr_db: SNR in dB
            
        Returns:
            Channel capacity in bits/s/Hz
        """
        snr_linear = 10**(snr_db/10)
        
        if self.channel_model == "awgn":
            # AWGN channel capacity: C = log2(1 + SNR)
            capacity = np.log2(1 + snr_linear)
        elif self.channel_model == "rayleigh":
            # Rayleigh fading channel capacity (ergodic)
            # This is an approximation
            capacity = np.log2(1 + snr_linear) * 0.7  # Rough approximation
        else:
            capacity = 0.0
        
        return capacity
    
    def apply_udp_channel(self, 
                         bits: torch.Tensor,
                         packet_header: Optional[bytes] = None,
                         max_retries: int = 3) -> Tuple[torch.Tensor, float, Optional[dict]]:
        """
        Apply UDP channel transmission to real USRP hardware
        
        Core UDP channel functionality:
        1. Converts 0/1 bit arrays to bytes for UDP transmission
        2. Sends data through UDP to USRP transmission system
        3. Receives response bytes from USRP reception system  
        4. Converts received bytes back to 0/1 bit arrays
        
        Bypasses modulation/demodulation as USRP handles that,
        directly interfacing with channel coding modules.
        
        Args:
            bits: Input bits tensor [batch_size, num_bits] (0s and 1s)
            packet_header: Optional packet header bytes
            max_retries: Maximum transmission retries
            
        Returns:
            tuple: (received_bits, estimated_noise_power, channel_info)
        """
        if self.channel_model != 'udp':
            raise ValueError("apply_udp_channel called but channel_model is not 'udp'")
        
        # Convert tensor to numpy array
        bits_np = bits.cpu().numpy() if isinstance(bits, torch.Tensor) else np.array(bits)
        bits_np = (bits_np > 0.5).astype(np.uint8)  # Ensure binary values
        batch_size, num_bits = bits_np.shape
        
        # UDP transmission with retry mechanism
        tx_success = False
        rx_data = None
        
        for attempt in range(max_retries):
            try:
                # Convert bits to bytes and transmit
                tx_data = self._bits_to_bytes(bits_np, packet_header)
                self.tx_socket.sendto(tx_data, (self.udp_tx_host, self.udp_tx_port))
                
                # Receive response from USRP
                rx_data, _ = self.rx_socket.recvfrom(self.udp_buffer_size)
                tx_success = True
                break
                
            except socket.timeout:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"UDP transmission timeout after {max_retries} attempts")
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"UDP transmission failed: {e}")
        
        # Convert received bytes back to bits
        if rx_data is not None:
            received_bits_np = self._bytes_to_bits(rx_data, batch_size, num_bits)
            received_bits = torch.tensor(received_bits_np, dtype=torch.float32, device=self.device)
        else:
            received_bits = torch.zeros_like(bits, device=self.device)
        
        # For UDP channel, set noise power and BER to 0.0 as they are not meaningful
        # Real performance depends on actual wireless channel conditions
        estimated_noise_power = 0.0
        estimated_ber = 0.0
        
        # Channel statistics (transmission success info only)
        channel_info = {
            'transmission_successful': tx_success,
            'estimated_ber': estimated_ber,
            'packet_size': len(tx_data) if 'tx_data' in locals() else 0,
            'response_size': len(rx_data) if rx_data else 0
        }
        
        return received_bits, estimated_noise_power, channel_info

    def _bits_to_bytes(self, bits_array: np.ndarray, header: Optional[bytes] = None) -> bytes:
        """
        Convert 2D bits array to bytes for UDP transmission
        
        Args:
            bits_array: numpy array of shape [batch_size, num_bits] with 0s and 1s
            header: Optional header bytes
            
        Returns:
            bytes: Packed data for UDP transmission
        """
        batch_size, num_bits = bits_array.shape
        
        # Create packet header with metadata
        packet_data = bytearray()
        
        # Add custom header if provided
        if header is not None:
            packet_data.extend(header)
        
        # Add metadata: batch_size, num_bits
        packet_data.extend(struct.pack('!II', batch_size, num_bits))
        
        # Pack bits into bytes (8 bits per byte)
        # Flatten the array for packing
        flat_bits = bits_array.flatten()
        
        # Pad to multiple of 8 bits
        padding_bits = (8 - (len(flat_bits) % 8)) % 8
        if padding_bits > 0:
            flat_bits = np.concatenate([flat_bits, np.zeros(padding_bits, dtype=np.uint8)])
        
        # Pack 8 bits into each byte
        num_bytes = len(flat_bits) // 8
        packed_bytes = np.zeros(num_bytes, dtype=np.uint8)
        
        for i in range(num_bytes):
            byte_val = 0
            for j in range(8):
                bit_idx = i * 8 + j
                if bit_idx < len(flat_bits):
                    byte_val |= (flat_bits[bit_idx] << (7 - j))
            packed_bytes[i] = byte_val
        
        packet_data.extend(packed_bytes.tobytes())
        
        # Add packet size at the beginning for validation
        final_packet = struct.pack('!I', len(packet_data)) + packet_data
        
        return bytes(final_packet)
    
    def _bytes_to_bits(self, data: bytes, expected_batch_size: int, expected_num_bits: int) -> np.ndarray:
        """
        Convert received bytes back to 2D bits array
        
        Args:
            data: Received bytes
            expected_batch_size: Expected batch size
            expected_num_bits: Expected number of bits per batch
            
        Returns:
            numpy array of shape [batch_size, num_bits] with 0s and 1s
        """
        try:
            # Extract packet size
            if len(data) < 4:
                raise ValueError("Received data too short")
            
            packet_size = struct.unpack('!I', data[:4])[0]
            packet_data = data[4:4+packet_size]
            
            # Extract metadata (skip custom header for now)
            if len(packet_data) < 8:
                raise ValueError("Packet data too short for metadata")
            
            batch_size, num_bits = struct.unpack('!II', packet_data[:8])
            bits_data = packet_data[8:]
            
            # Use expected dimensions for safety
            if batch_size != expected_batch_size or num_bits != expected_num_bits:
                batch_size, num_bits = expected_batch_size, expected_num_bits
            
            # Unpack bytes back to bits
            total_bits = batch_size * num_bits
            padding_bits = (8 - (total_bits % 8)) % 8
            expected_bytes = (total_bits + padding_bits) // 8
            
            if len(bits_data) < expected_bytes:
                # Pad with zeros if insufficient data
                bits_data += b'\x00' * (expected_bytes - len(bits_data))
            
            # Unpack bits from bytes
            unpacked_bits = []
            for byte_val in bits_data[:expected_bytes]:
                for j in range(8):
                    bit = (byte_val >> (7 - j)) & 1
                    unpacked_bits.append(bit)
            
            # Take only required bits and reshape
            unpacked_bits = unpacked_bits[:total_bits]
            bits_array = np.array(unpacked_bits, dtype=np.uint8).reshape(batch_size, num_bits)
            
            return bits_array
            
        except Exception as e:
            # Return zeros as fallback
            return np.zeros((expected_batch_size, expected_num_bits), dtype=np.uint8)
