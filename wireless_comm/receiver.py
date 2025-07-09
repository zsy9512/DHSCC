#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wireless Communication Receiver Module

This module contains all receiver-side processing including:
- Channel equalization
- Demodulation with channel information
- Channel decoding (Polar decoding)
- Performance evaluation utilities
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from .polar_decoding import Polar5GDecoder
from .mapping import Constellation, Demapper


class WirelessReceiver(nn.Module):
    """
    Unified wireless receiver supporting both coded and uncoded reception
    """
    
    def __init__(self,
                 # Coding parameters
                 encoder_ref: Optional[object] = None,
                 decoder_type: str = "SC",
                 list_size: int = 8,
                 use_coding: bool = True,
                 
                 # Demodulation parameters
                 constellation_type: str = "qam",
                 bits_per_symbol: int = 4,
                 normalize: bool = True,
                 demapping_method: str = "app",
                 
                 # Equalization parameters
                 equalization_method: str = "mrc",
                 
                 # System parameters
                 device: str = "cpu",
                 verbose: bool = False):
        """
        Initialize the wireless receiver
        
        Args:
            encoder_ref: Reference to encoder (required for coded systems)
            decoder_type: Decoder type ('SC', 'SCL')
            list_size: List size for SCL decoder
            use_coding: Whether to use channel decoding
            constellation_type: Type of constellation ('qam', 'bpsk', 'pam')
            bits_per_symbol: Number of bits per symbol
            normalize: Whether constellation is normalized
            demapping_method: Demapping method ('app', 'maxlog')
            equalization_method: Channel equalization method
            device: Device to run computations on
            verbose: Whether to display verbose output
        """
        super().__init__()
        
        self.device = device
        self.use_coding = use_coding
        self.bits_per_symbol = bits_per_symbol
        self.demapping_method = demapping_method
        self.equalization_method = equalization_method
        self.verbose = verbose
        
        # Channel decoding setup
        if use_coding:
            if encoder_ref is None:
                raise ValueError("encoder_ref must be provided when use_coding=True")
            
            self.decoder = Polar5GDecoder(
                enc_polar=encoder_ref,
                dec_type=decoder_type,
                list_size=list_size if decoder_type == "SCL" else None,
                return_crc_status=False,
                dtype=torch.float32
            ).to(device)
        else:
            self.decoder = None
        
        # Demodulation setup
        if constellation_type.lower() == "bpsk" or bits_per_symbol == 1:
            self.modulation = "bpsk"
            self.constellation = None
            self.demapper = self._bpsk_demapper
        else:
            self.modulation = constellation_type.lower()
            
            # Create constellation
            self.constellation = Constellation(
                constellation_type=constellation_type,
                num_bits_per_symbol=bits_per_symbol,
                normalize=normalize
            )
            
            # Create demapper
            self.demapper = Demapper(
                demapping_method=demapping_method,
                constellation=self.constellation
            )
            
            # Move to device
            if self.constellation is not None:
                self.constellation.points = self.constellation.points.to(device)
    
    def _bpsk_demapper(self, symbols: torch.Tensor, noise_var: float, h: torch.Tensor = None) -> torch.Tensor:
        """
        Simple BPSK demapper that converts symbols to LLRs
        
        Args:
            symbols: Received symbols
            noise_var: Noise variance
            h: Channel coefficients (for fading channels)
            
        Returns:
            LLRs for each bit
        """
        if h is not None:
            # Fading channel: LLR = 2 * Re(h* * y) / (|h|^2 * sigma^2)
            if h.is_complex():
                h_conj = torch.conj(h)
                h_magnitude_squared = torch.abs(h)**2
                llrs = 2 * torch.real(h_conj * symbols) / (h_magnitude_squared * noise_var)
            else:
                # Real channel coefficients
                llrs = 2 * h * symbols / (h**2 * noise_var)
        else:
            # AWGN channel: LLR = 2 * y / sigma^2
            if symbols.is_complex():
                llrs = 2 * symbols.real / noise_var
            else:
                llrs = 2 * symbols / noise_var
        
        return llrs
    
    def equalize_channel(self, 
                        received_symbols: torch.Tensor, 
                        h: torch.Tensor,
                        noise_var: float) -> Tuple[torch.Tensor, float]:
        """
        Apply channel equalization
        
        Args:
            received_symbols: Received symbols after channel
            h: Channel coefficients
            noise_var: Noise variance
            
        Returns:
            tuple: (equalized_symbols, adjusted_noise_var)
        """
        if h is None:
            return received_symbols, noise_var
        
        if self.equalization_method.lower() == "identity":
            return received_symbols, noise_var
        
        elif self.equalization_method.lower() == "zf":
            # Zero-Forcing equalization: y_eq = y / h
            h_magnitude = torch.abs(h)
            h_safe = torch.where(h_magnitude < 1e-8, 
                               torch.sgn(h) * 1e-8, h)
            equalized_symbols = received_symbols / h_safe
            
            # ZF amplifies noise by |h|^(-2) for each symbol
            h_magnitude_squared = torch.abs(h)**2
            h_mag_sq_safe = torch.where(h_magnitude_squared < 1e-8, 
                                       1e-8, h_magnitude_squared)
            # Use harmonic mean of noise amplification factors
            adjusted_noise_var = noise_var * torch.mean(1.0 / h_mag_sq_safe).item()
            
            return equalized_symbols, adjusted_noise_var
        
        elif self.equalization_method.lower() == "mrc":
            # Maximum Ratio Combining: y_eq = h* * y / |h|^2
            h_conj = torch.conj(h)
            h_magnitude_squared = torch.abs(h)**2
            h_mag_sq_safe = torch.where(h_magnitude_squared < 1e-8, 
                                       1e-8, h_magnitude_squared)
            equalized_symbols = (h_conj * received_symbols) / h_magnitude_squared
            
            # MRC provides diversity gain - use average channel gain
            adjusted_noise_var = noise_var / h_magnitude_squared.mean().item()
            
            return equalized_symbols, adjusted_noise_var
    
        elif self.equalization_method.lower() == "mmse":
            # MMSE equalization: y_eq = h* * y / (|h|^2 + sigma^2)
            h_conj = torch.conj(h)
            h_magnitude_squared = torch.abs(h)**2
            denominator = h_magnitude_squared + noise_var
            equalized_symbols = (h_conj * received_symbols) / denominator
            
            # For MMSE, use original noise variance
            adjusted_noise_var = noise_var
            
            return equalized_symbols, adjusted_noise_var
        
        else:
            raise ValueError(f"Unknown equalization method: {self.equalization_method}")
    
    def demodulate(self, 
                  received_symbols: torch.Tensor, 
                  noise_var: float,
                  h: torch.Tensor = None) -> torch.Tensor:
        """
        Demodulate received symbols to LLRs with channel information
        
        Args:
            received_symbols: Received symbols
            noise_var: Noise variance
            h: Channel coefficients (for fading channels)
            
        Returns:
            LLRs for each bit
        """
        if h is not None:
            # Apply channel equalization
            equalized_symbols, adjusted_noise_var = self.equalize_channel(
                received_symbols, h, noise_var
            )
            
            # Use demapper with equalized symbols
            if self.modulation == "bpsk":
                return self._bpsk_demapper(equalized_symbols, adjusted_noise_var)
            else:
                return self.demapper(equalized_symbols, adjusted_noise_var)
        else:
            # AWGN channel - direct demodulation
            if self.modulation == "bpsk":
                return self._bpsk_demapper(received_symbols, noise_var)
            else:
                return self.demapper(received_symbols, noise_var)
    
    def decode(self, llrs: torch.Tensor) -> torch.Tensor:
        """
        Apply channel decoding to LLRs
        
        Args:
            llrs: Log-likelihood ratios
            
        Returns:
            Decoded bits (or hard decisions if uncoded)
        """
        if self.use_coding and self.decoder is not None:
            if self.verbose:
                print(f"Polar decoding: input LLRs shape {llrs.shape}")
            
            decoded = self.decoder(llrs)
            
            if self.verbose:
                print(f"Polar decoding: output bits shape {decoded.shape}")
            return decoded
        else:
            # Hard decision decoding for uncoded systems
            hard_bits = (llrs < 0).float()
            if self.verbose:
                print(f"Hard decision decoding: input LLRs shape {llrs.shape}, output bits shape {hard_bits.shape}")
            return hard_bits
    
    def decode_bits(self, received_bits: torch.Tensor) -> torch.Tensor:
        """
        Directly decode received bits (for UDP channel where USRP handles modulation)
        
        Args:
            received_bits: Received bits from UDP channel
            
        Returns:
            Decoded information bits
        """
        received_bits = received_bits.to(self.device)
        
        if self.use_coding and self.decoder is not None:
            # Convert bits to LLRs for decoder input
            # Bits are 0/1, convert to LLRs: 0 -> +inf, 1 -> -inf
            llrs = torch.where(received_bits > 0.5, 
                             torch.tensor(-10.0, device=self.device), 
                             torch.tensor(10.0, device=self.device))
            
            # Apply channel decoding
            decoded_bits = self.decoder(llrs)
        else:
            # For uncoded systems, return bits as-is
            decoded_bits = received_bits
        
        return decoded_bits

    def forward(self, 
               received_symbols: torch.Tensor, 
               noise_var: float,
               h: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete receiver processing: demodulation + decoding
        
        Args:
            received_symbols: Received symbols from channel
            noise_var: Noise variance
            h: Channel coefficients (optional)
            
        Returns:
            tuple: (llrs, decoded_bits)
        """
        # Move symbols to device
        received_symbols = received_symbols.to(self.device)
        if h is not None:
            h = h.to(self.device)
        
        # Demodulation
        llrs = self.demodulate(received_symbols, noise_var, h)
        
        # Handle reshaping for multi-bit modulations
        if self.bits_per_symbol > 1 and llrs.ndim > 2:
            # Reshape from [batch, symbols, bits_per_symbol] to [batch, total_bits]
            batch_size = llrs.size(0)
            llrs_reshaped = llrs.reshape(batch_size, -1)
        else:
            llrs_reshaped = llrs
        
        # Channel decoding
        decoded_bits = self.decode(llrs_reshaped)
        
        return llrs_reshaped, decoded_bits
    
    def save_state_dict(self, filepath: str):
        """Save receiver parameters"""
        state_dict = {
            'use_coding': self.use_coding,
            'bits_per_symbol': self.bits_per_symbol,
            'modulation': self.modulation,
            'demapping_method': self.demapping_method,
            'equalization_method': self.equalization_method
        }
        
        if self.decoder is not None:
            state_dict['decoder'] = self.decoder.state_dict()
        if self.constellation is not None:
            state_dict['constellation_points'] = self.constellation.points
            
        torch.save(state_dict, filepath)
        
    def load_state_dict(self, filepath: str):
        """Load receiver parameters"""
        state_dict = torch.load(filepath, map_location=self.device)
        
        # Load basic parameters
        self.use_coding = state_dict['use_coding']
        self.bits_per_symbol = state_dict['bits_per_symbol']
        self.modulation = state_dict['modulation']
        self.demapping_method = state_dict['demapping_method']
        self.equalization_method = state_dict['equalization_method']
        
        # Load decoder state if present
        if 'decoder' in state_dict and self.decoder is not None:
            self.decoder.load_state_dict(state_dict['decoder'])
            
        # Load constellation points if present
        if 'constellation_points' in state_dict and self.constellation is not None:
            self.constellation.points = state_dict['constellation_points'].to(self.device)
