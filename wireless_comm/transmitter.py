#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wireless Communication Transmitter Module

This module contains all transmitter-side processing including:
- Channel coding (Polar coding)
- Modulation (BPSK, QAM, PAM)
- Signal processing for transmission
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from .polar_encoding import Polar5GEncoder
from .mapping import Constellation, Mapper


class WirelessTransmitter(nn.Module):
    """
    Unified wireless transmitter supporting both coded and uncoded transmission
    """
    
    def __init__(self,
                 # Coding parameters
                 k: Optional[int] = None,
                 n: Optional[int] = None,
                 channel_type: str = "uplink",
                 use_coding: bool = True,
                 
                 # Modulation parameters
                 constellation_type: str = "qam",
                 bits_per_symbol: int = 4,
                 normalize: bool = True,
                 
                 # System parameters
                 device: str = "cpu",
                 verbose: bool = False):
        """
        Initialize the wireless transmitter
        
        Args:
            k: Number of information bits (required if use_coding=True)
            n: Codeword length (required if use_coding=True)
            channel_type: "uplink" or "downlink" for 5G polar coding
            use_coding: Whether to use channel coding
            constellation_type: Type of constellation ('qam', 'bpsk', 'pam')
            bits_per_symbol: Number of bits per symbol
            normalize: Whether to normalize constellation to unit power
            device: Device to run computations on
            verbose: Whether to display verbose output
        """
        super().__init__()
        
        self.device = device
        self.use_coding = use_coding
        self.bits_per_symbol = bits_per_symbol
        self.verbose = verbose
        
        # Channel coding setup
        if use_coding:
            if k is None or n is None:
                raise ValueError("k and n must be specified when use_coding=True")
            
            self.k = k
            self.n = n
            self.code_rate = k / n
            
            # Create polar encoder
            self.encoder = Polar5GEncoder(
                k=k,
                n=n,
                channel_type=channel_type,
                verbose=verbose,
                dtype=torch.float32
            ).to(device)
        else:
            self.k = None
            self.n = None
            self.code_rate = 1.0
            self.encoder = None
        
        # Modulation setup
        if constellation_type.lower() == "bpsk" or bits_per_symbol == 1:
            self.modulation = "bpsk"
            self.constellation = None
            self.mapper = self._bpsk_mapper
        else:
            self.modulation = constellation_type.lower()
            
            # Create constellation
            self.constellation = Constellation(
                constellation_type=constellation_type,
                num_bits_per_symbol=bits_per_symbol,
                normalize=normalize
            )
            
            # Create mapper
            self.mapper = Mapper(constellation=self.constellation)
            
            # Move to device
            if self.constellation is not None:
                self.constellation.points = self.constellation.points.to(device)
    
    def _bpsk_mapper(self, bits: torch.Tensor) -> torch.Tensor:
        """Simple BPSK mapper: 0 -> +1, 1 -> -1"""
        return 1 - 2 * bits.float()
    
    def encode(self, bits: torch.Tensor) -> torch.Tensor:
        """
        Apply channel coding to information bits
        
        Args:
            bits: Information bits to encode
            
        Returns:
            Encoded bits (or original bits if uncoded)
        """
        if self.use_coding and self.encoder is not None:
            if self.verbose:
                print(f"Polar encoding: input bits shape {bits.shape}, k={self.k}, n={self.n}")
            
            # Ensure input size matches encoder requirements
            if bits.shape[1] != self.k:
                if bits.shape[1] < self.k:
                    # Pad with zeros
                    padding_size = self.k - bits.shape[1]
                    padding = torch.zeros(bits.shape[0], padding_size, device=bits.device, dtype=bits.dtype)
                    padded_bits = torch.cat([bits, padding], dim=1)
                    if self.verbose:
                        print(f"Padding input from {bits.shape[1]} to {self.k} bits")
                    encoded = self.encoder(padded_bits)
                else:
                    # Truncate 
                    truncated_bits = bits[:, :self.k]
                    if self.verbose:
                        print(f"Truncating input from {bits.shape[1]} to {self.k} bits")
                    encoded = self.encoder(truncated_bits)
            else:
                encoded = self.encoder(bits)
            
            if self.verbose:
                print(f"Polar encoding: output bits shape {encoded.shape}")
            return encoded
        else:
            if self.verbose:
                print(f"No encoding applied: input/output shape {bits.shape}")
            return bits
    
    def modulate(self, bits: torch.Tensor) -> torch.Tensor:
        """
        Modulate bits to symbols
        
        Args:
            bits: Bits to modulate
            
        Returns:
            Modulated symbols
        """
        return self.mapper(bits)
    
    def transmit_udp(self, bits: torch.Tensor, channel: object) -> Tuple[torch.Tensor, float, Optional[torch.Tensor]]:
        """
        Transmit bits through UDP channel (bypassing modulation)
        
        This method is used when the channel is a UDP channel connected to real USRP hardware.
        It bypasses the modulation step since USRP handles modulation/demodulation.
        
        Args:
            bits: Information bits to transmit
            channel: UDP channel object
            
        Returns:
            tuple: (received_bits, estimated_noise_power, channel_info)
        """
        # Move bits to device
        bits = bits.to(self.device)
        
        # Channel coding (if enabled)
        encoded_bits = self.encode(bits)
        
        # Direct transmission through UDP channel (bypassing modulation)
        received_bits, noise_power, channel_info = channel.apply_udp_channel(encoded_bits)
        
        return received_bits, noise_power, channel_info

    def forward(self, bits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete transmitter processing: encoding + modulation
        
        Args:
            bits: Information bits
            
        Returns:
            tuple: (encoded_bits, modulated_symbols)
        """
        # Move bits to device
        bits = bits.to(self.device)
        
        # Channel coding
        encoded_bits = self.encode(bits)
        
        # Modulation
        symbols = self.modulate(encoded_bits)
        
        return encoded_bits, symbols
    
    def get_code_rate(self) -> float:
        """Get the code rate"""
        return self.code_rate
    
    def get_bits_per_symbol(self) -> int:
        """Get bits per symbol"""
        return self.bits_per_symbol
    
    def save_state_dict(self, filepath: str):
        """Save transmitter parameters"""
        state_dict = {
            'use_coding': self.use_coding,
            'k': self.k,
            'n': self.n,
            'bits_per_symbol': self.bits_per_symbol,
            'modulation': self.modulation,
            'code_rate': self.code_rate
        }
        
        if self.encoder is not None:
            state_dict['encoder'] = self.encoder.state_dict()
        if self.constellation is not None:
            state_dict['constellation_points'] = self.constellation.points
            
        torch.save(state_dict, filepath)
        
    def load_state_dict(self, filepath: str):
        """Load transmitter parameters"""
        state_dict = torch.load(filepath, map_location=self.device)
        
        # Load basic parameters
        self.use_coding = state_dict['use_coding']
        self.k = state_dict['k']
        self.n = state_dict['n']
        self.bits_per_symbol = state_dict['bits_per_symbol']
        self.modulation = state_dict['modulation']
        self.code_rate = state_dict['code_rate']
        
        # Load encoder state if present
        if 'encoder' in state_dict and self.encoder is not None:
            self.encoder.load_state_dict(state_dict['encoder'])
            
        # Load constellation points if present
        if 'constellation_points' in state_dict and self.constellation is not None:
            self.constellation.points = state_dict['constellation_points'].to(self.device)
