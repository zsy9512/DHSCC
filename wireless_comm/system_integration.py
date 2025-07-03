#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wireless Communication System Integration Module

This module integrates transmitter, channel, and receiver modules to create
complete communication systems with performance evaluation capabilities.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Union, Optional
import time

from .transmitter import WirelessTransmitter
from .channel import WirelessChannel
from .receiver import WirelessReceiver


class WirelessCommunicationSystem:
    """
    Complete wireless communication system integrating all modules
    """
    
    def __init__(self,
                 # System type
                 use_coding: bool = True,
                 
                 # Coding parameters (for coded systems)
                 k: Optional[int] = None,
                 n: Optional[int] = None,
                 channel_type: str = "uplink",
                 decoder_type: str = "SC",
                 list_size: int = 8,
                 
                 # Modulation parameters
                 constellation_type: str = "qam",
                 bits_per_symbol: int = 4,
                 normalize: bool = True,
                 demapping_method: str = "app",
                 
                 # Channel parameters
                 channel_model: str = "awgn",
                 equalization_method: str = "mrc",
                 
                 # UDP channel parameters (for real hardware transmission)
                 udp_tx_host: str = "localhost",
                 udp_tx_port: int = 12345,
                 udp_rx_host: str = "localhost",
                 udp_rx_port: int = 12346,
                 udp_timeout: float = 10.0,
                 udp_buffer_size: int = 8192,
                 
                 # System parameters
                 device: str = "cpu",
                 verbose: bool = False):
        """
        Initialize the complete communication system
        
        Args:
            use_coding: Whether to use channel coding
            k: Number of information bits (required for coded systems)
            n: Codeword length (required for coded systems)
            channel_type: "uplink" or "downlink" for 5G polar coding
            decoder_type: Decoder type ('SC', 'SCL')
            list_size: List size for SCL decoder
            constellation_type: Type of constellation ('qam', 'bpsk', 'pam')
            bits_per_symbol: Number of bits per symbol
            normalize: Whether to normalize constellation
            demapping_method: Demapping method ('app', 'maxlog')
            channel_model: Channel model type ('awgn', 'rayleigh', 'udp')
            equalization_method: Channel equalization method
            udp_tx_host: UDP transmission host (for 'udp' channel)
            udp_tx_port: UDP transmission port (for 'udp' channel)
            udp_rx_host: UDP reception host (for 'udp' channel)
            udp_rx_port: UDP reception port (for 'udp' channel)
            udp_timeout: UDP socket timeout in seconds (for 'udp' channel)
            udp_buffer_size: UDP buffer size in bytes (for 'udp' channel)
            device: Device to run computations on
            verbose: Whether to display verbose output
        """
        self.device = device
        self.use_coding = use_coding
        self.bits_per_symbol = bits_per_symbol
        self.channel_model = channel_model
        self.verbose = verbose
        
        # Create transmitter
        self.transmitter = WirelessTransmitter(
            k=k,
            n=n,
            channel_type=channel_type,
            use_coding=use_coding,
            constellation_type=constellation_type,
            bits_per_symbol=bits_per_symbol,
            normalize=normalize,
            device=device,
            verbose=verbose
        )
        
        # Create channel
        self.channel = WirelessChannel(
            channel_model=channel_model,
            device=device,
            udp_tx_host=udp_tx_host,
            udp_tx_port=udp_tx_port,
            udp_rx_host=udp_rx_host,
            udp_rx_port=udp_rx_port,
            udp_timeout=udp_timeout,
            udp_buffer_size=udp_buffer_size
        )
        
        # Create receiver
        self.receiver = WirelessReceiver(
            encoder_ref=self.transmitter.encoder if use_coding else None,
            decoder_type=decoder_type,
            list_size=list_size,
            use_coding=use_coding,
            constellation_type=constellation_type,
            bits_per_symbol=bits_per_symbol,
            normalize=normalize,
            demapping_method=demapping_method,
            equalization_method=equalization_method,
            device=device,
            verbose=verbose
        )
        
        # System parameters
        self.k = k
        self.n = n
        self.code_rate = self.transmitter.get_code_rate()
        
        if verbose:
            self._print_system_info()
    
    def _print_system_info(self):
        """Print system configuration information"""
        print(f"Wireless Communication System Configuration:")
        print(f"- System type: {'Coded' if self.use_coding else 'Uncoded'}")
        if self.use_coding:
            print(f"- Info bits (k): {self.k}")
            print(f"- Code bits (n): {self.n}")
            print(f"- Code rate: {self.code_rate:.3f}")
        if self.bits_per_symbol == 1:
            print(f"- Modulation: BPSK")
        else:
            print(f"- Modulation: {2**self.bits_per_symbol}-{self.transmitter.modulation.upper()}")
        print(f"- Channel model: {self.channel_model.upper()}")
        print(f"- Device: {self.device}")
    
    def __call__(self, 
                bits: torch.Tensor, 
                ebno_db: Optional[float] = None, 
                snr_db: Optional[float] = None) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Process bits through the complete communication system
        
        Args:
            bits: Information bits to transmit
            ebno_db: Eb/N0 in dB (not used for UDP channel)
            snr_db: SNR in dB (not used for UDP channel)
            
        Returns:
            For coded systems: (decoded_bits, original_bits)
            For uncoded systems: (llrs, noise_var, channel_info)
        """
        # For UDP channel, SNR parameters are not needed
        if self.channel_model != 'udp' and ebno_db is None and snr_db is None:
            raise ValueError("Either ebno_db or snr_db must be provided for simulated channels")
        
        # Transmitter processing
        encoded_bits, symbols = self.transmitter(bits)
        
        # Channel processing
        if self.channel_model == 'udp':
            # For UDP channel, work directly with encoded bits
            # bypassing modulation/demodulation as USRP handles it
            received_bits, noise_var, channel_info = self.channel.apply_udp_channel(encoded_bits)
            
            # For UDP channel, go directly to decoding
            if self.use_coding:
                decoded_bits = self.receiver.decode_bits(received_bits)
                return decoded_bits, bits  # Return decoded_bits and original for coded systems
            else:
                # Convert bits to LLR format for consistency with uncoded interface
                llrs = torch.where(received_bits == 1, torch.tensor(1.0), torch.tensor(-1.0))
                return llrs, noise_var, channel_info
        else:
            # For simulated channels, use standard processing
            received_symbols, noise_var, h = self.channel.apply_channel(
                symbols,
                snr_db=snr_db,
                ebn0_db=ebno_db,
                bits_per_symbol=self.bits_per_symbol,
                code_rate=self.code_rate
            )
            
            # Receiver processing
            llrs, decoded_bits = self.receiver(received_symbols, noise_var, h)
            
            if self.use_coding:
                # For coded systems, return decoded bits and original bits
                return decoded_bits, bits
            else:
                # For uncoded systems, return LLRs, noise_var, and channel info
                return llrs, noise_var, h
    
    def evaluate_ber(self, 
                    batch_size: int, 
                    ebno_db_range: List[float], 
                    max_errors: int = 100, 
                    max_bits: int = 10**6, 
                    show_progress: bool = True) -> Dict:
        """
        Evaluate BER and BLER performance over a range of Eb/N0 values
        
        Note: UDP channel evaluation is not supported as it uses real hardware.
        For UDP channel, use evaluate_udp_channel() method instead.
        
        Args:
            batch_size: Number of codewords per batch
            ebno_db_range: Range of Eb/N0 values in dB
            max_errors: Early stopping based on number of errors
            max_bits: Maximum number of bits to process
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary containing BER and BLER results
        """
        # Check if UDP channel is being used
        if self.channel_model == 'udp':
            print("Warning: UDP channel evaluation not supported in evaluate_ber().")
            print("UDP channel uses real hardware and doesn't depend on Eb/N0 parameters.")
            print("Use evaluate_udp_channel() method for UDP channel evaluation.")
            
            # Return dummy results for UDP channel
            results = {
                'ebno_dbs': ebno_db_range,
                'bers': [0.0] * len(ebno_db_range)
            }
            if self.use_coding:
                results['blers'] = [0.0] * len(ebno_db_range)
            return results
        
        bers = []
        blers = [] if self.use_coding else None
        
        # Print header
        if self.use_coding:
            print(f"{'Eb/N0 (dB)':<10} {'BER':<15} {'BLER':<15} {'Bits':<15}")
            print("-" * 60)
        else:
            print(f"{'Eb/N0 (dB)':<10} {'BER':<15} {'Bits':<15}")
            print("-" * 45)
        
        for ebno_db in ebno_db_range:
            bit_errors = 0
            block_errors = 0
            total_bits = 0
            total_blocks = 0
            
            # Progress bar
            pbar = tqdm(total=max_errors, disable=not show_progress,
                       desc=f"Eb/N0 = {ebno_db:.1f} dB")
            
            while (bit_errors < max_errors and total_bits < max_bits):
                # Generate random information bits
                if self.use_coding:
                    info_bits = torch.randint(0, 2, (batch_size, self.k), 
                                            device=self.device, dtype=torch.float32)
                else:
                    # For uncoded systems, generate bits for modulation
                    bits_per_batch = batch_size * 100  # Arbitrary length
                    info_bits = torch.randint(0, 2, (batch_size, bits_per_batch), 
                                            device=self.device, dtype=torch.float32)
                
                # Process through communication system
                decoded_bits, original_bits = self(info_bits, ebno_db=ebno_db)
                
                # Calculate errors
                bit_error_mask = (decoded_bits != original_bits)
                errors = bit_error_mask.sum().item()
                bit_errors += errors
                total_bits += original_bits.numel()
                
                if self.use_coding:
                    # Calculate block errors
                    block_error_mask = bit_error_mask.any(dim=1)
                    block_errors += block_error_mask.sum().item()
                    total_blocks += batch_size
                
                # Update progress
                pbar.update(errors)
                
                # Early stopping for high SNR
                if ebno_db > 10.0 and total_bits >= 1e6 and bit_errors == 0:
                    break
            
            pbar.close()
            
            # Calculate rates
            ber = bit_errors / total_bits
            bers.append(ber)
            
            if self.use_coding:
                bler = block_errors / total_blocks
                blers.append(bler)
                print(f"{ebno_db:<10.1f} {ber:<15.6e} {bler:<15.6e} {total_bits:<15d}")
            else:
                print(f"{ebno_db:<10.1f} {ber:<15.6e} {total_bits:<15d}")
        
        # Return results
        results = {
            'ebno_dbs': ebno_db_range,
            'bers': bers
        }
        if self.use_coding:
            results['blers'] = blers
        
        return results
    
    def plot_results(self, 
                    results: Dict, 
                    title: Optional[str] = None, 
                    filename: Optional[str] = None, 
                    show_bler: bool = True):
        """
        Plot BER/BLER results
        
        Args:
            results: Results from evaluate_ber
            title: Plot title
            filename: Save figure to file
            show_bler: Whether to show BLER curve
        """
        plt.figure(figsize=(10, 7))
        
        ebno_dbs = results['ebno_dbs']
        bers = results['bers']
        
        # Plot BER
        system_label = f"{'Coded' if self.use_coding else 'Uncoded'} System ({self.channel_model.upper()})"
        plt.semilogy(ebno_dbs, bers, 'bo-', label=f'BER - {system_label}', linewidth=2)
        
        # Plot BLER if available and requested
        if show_bler and 'blers' in results:
            blers = results['blers']
            plt.semilogy(ebno_dbs, blers, 'rs-', label=f'BLER - {system_label}', linewidth=2)
        
        # Add theoretical curves for reference
        if self.channel_model == "awgn":
            try:
                from scipy.special import erfc
                ebno_range = np.linspace(min(ebno_dbs), max(ebno_dbs), 100)
                
                # Theoretical BER for BPSK over AWGN
                if not self.use_coding:  # Only for uncoded systems
                    ber_theory = 0.5 * erfc(np.sqrt(2 * 10**(ebno_range/10)))
                    plt.semilogy(ebno_range, ber_theory, 'g-.', 
                               label='BPSK Theory (AWGN)', linewidth=1.5, alpha=0.7)
            except ImportError:
                pass
        
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.xlabel('Eb/No (dB)', fontsize=12)
        plt.ylabel('Error Rate', fontsize=12)
        
        if title:
            plt.title(title, fontsize=14)
        else:
            if self.bits_per_symbol == 1:
                mod_name = "BPSK"
            else:
                mod_name = f"{2**self.bits_per_symbol}-{self.transmitter.modulation.upper()}"
            
            system_type = "Coded" if self.use_coding else "Uncoded"
            plt.title(f'{system_type} {mod_name} Performance ({self.channel_model.upper()} Channel)', 
                     fontsize=14)
        
        plt.legend(loc='best')
        plt.ylim(bottom=1e-6)
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def save_system(self, base_path: str):
        """Save complete system configuration"""
        self.transmitter.save_state_dict(f"{base_path}_transmitter.pth")
        self.receiver.save_state_dict(f"{base_path}_receiver.pth")
        
        # Save system configuration
        system_config = {
            'use_coding': self.use_coding,
            'k': self.k,
            'n': self.n,
            'code_rate': self.code_rate,
            'bits_per_symbol': self.bits_per_symbol,
            'channel_model': self.channel_model
        }
        torch.save(system_config, f"{base_path}_config.pth")
    
    def load_system(self, base_path: str):
        """Load complete system configuration"""
        # Load system configuration
        system_config = torch.load(f"{base_path}_config.pth", map_location=self.device)
        
        self.use_coding = system_config['use_coding']
        self.k = system_config['k']
        self.n = system_config['n']
        self.code_rate = system_config['code_rate']
        self.bits_per_symbol = system_config['bits_per_symbol']
        self.channel_model = system_config['channel_model']
        
        # Load transmitter and receiver
        self.transmitter.load_state_dict(f"{base_path}_transmitter.pth")
        self.receiver.load_state_dict(f"{base_path}_receiver.pth")
    
    def evaluate_udp_channel(self, 
                           batch_size: int, 
                           num_iterations: int = 100,
                           max_errors: int = 1000,
                           show_progress: bool = True) -> Dict:
        """
        Evaluate UDP channel performance through real USRP hardware
        
        Args:
            batch_size: Number of codewords per batch
            num_iterations: Number of transmission iterations
            max_errors: Maximum errors before stopping
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary containing BER and BLER results
        """
        if self.channel_model != 'udp':
            raise ValueError("evaluate_udp_channel can only be used with UDP channel")
        
        bit_errors = 0
        block_errors = 0
        total_bits = 0
        total_blocks = 0
        transmission_times = []
        
        # Progress bar
        pbar = tqdm(total=num_iterations, disable=not show_progress,
                   desc="UDP Channel Evaluation")
        
        for iteration in range(num_iterations):
            if bit_errors >= max_errors:
                break
                
            try:
                # Generate random information bits
                if self.use_coding:
                    info_bits = torch.randint(0, 2, (batch_size, self.k), 
                                            device=self.device, dtype=torch.float32)
                else:
                    bits_per_batch = batch_size * 100  # Arbitrary length
                    info_bits = torch.randint(0, 2, (batch_size, bits_per_batch), 
                                            device=self.device, dtype=torch.float32)
                
                # Process through UDP communication system
                start_time = time.time()
                decoded_bits, original_bits = self(info_bits, ebno_db=0.0)  # Dummy SNR for UDP
                transmission_time = time.time() - start_time
                transmission_times.append(transmission_time)
                
                # Calculate errors
                bit_error_mask = (decoded_bits != original_bits)
                errors = bit_error_mask.sum().item()
                bit_errors += errors
                total_bits += original_bits.numel()
                
                if self.use_coding:
                    # Calculate block errors
                    block_error_mask = bit_error_mask.any(dim=1)
                    block_errors += block_error_mask.sum().item()
                    total_blocks += batch_size
                
                pbar.update(1)
                
            except Exception as e:
                print(f"UDP transmission failed in iteration {iteration}: {e}")
                break
        
        pbar.close()
        
        # Calculate performance metrics
        ber = bit_errors / total_bits if total_bits > 0 else 0.0
        avg_transmission_time = np.mean(transmission_times) if transmission_times else 0.0
        
        results = {
            'ber': ber,
            'bit_errors': bit_errors,
            'total_bits': total_bits,
            'iterations_completed': len(transmission_times),
            'avg_transmission_time': avg_transmission_time,
            'throughput_bps': total_bits / sum(transmission_times) if transmission_times else 0.0
        }
        
        if self.use_coding:
            bler = block_errors / total_blocks if total_blocks > 0 else 0.0
            results['bler'] = bler
            results['block_errors'] = block_errors
            results['total_blocks'] = total_blocks
        
        # Print results
        print(f"\nUDP Channel Evaluation Results:")
        print(f"BER: {ber:.6e}")
        if self.use_coding:
            print(f"BLER: {bler:.6e}")
        print(f"Total bits: {total_bits}")
        print(f"Bit errors: {bit_errors}")
        print(f"Completed iterations: {len(transmission_times)}")
        print(f"Average transmission time: {avg_transmission_time:.3f} seconds")
        print(f"Throughput: {results['throughput_bps']:.1f} bps")
        
        return results
