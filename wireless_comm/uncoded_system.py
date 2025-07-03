#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uncoded system with AWGN or Rayleigh channel using mapping_torch for modulation/demodulation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from .mapping import Constellation, Mapper, Demapper

def add_awgn(symbols, snr_db=None, ebn0_db=None, bits_per_symbol=None):
    """
    Add Additive White Gaussian Noise to symbols at specified SNR or Eb/N0
    
    Args:
        symbols (torch.Tensor): Real or complex symbols
        snr_db (float, optional): Signal-to-Noise ratio in dB
        ebn0_db (float, optional): Energy per bit to noise power spectral density ratio in dB
        bits_per_symbol (int, optional): Number of bits per symbol, required when ebn0_db is provided
        
    Returns:
        tuple: (torch.Tensor: Noisy symbols, float: Noise power)
    """
    # 检查符号是复数还是实数
    is_complex = symbols.is_complex()
    
    # 计算符号能量 (假设归一化星座)
    symbol_energy = torch.mean(torch.abs(symbols)**2).item()
    
    # 检查提供的是哪个参数并相应地计算噪声功率
    if snr_db is not None:
        # 将 SNR 从 dB 转换为线性尺度
        snr_linear = 10.0**(snr_db/10.0)
        # 基于符号能量和 SNR 计算噪声功率
        noise_power = symbol_energy / snr_linear
    elif ebn0_db is not None:
        if bits_per_symbol is None:
            raise ValueError("bits_per_symbol must be provided when using ebn0_db")
        # 将 Eb/N0 从 dB 转换为线性尺度
        ebn0_linear = 10.0**(ebn0_db/10.0)
        
        # 计算每比特能量 - 这里是关键修改点
        # BPSK 或 QPSK 都应该以相同方式处理
        bit_energy = symbol_energy / bits_per_symbol
        
        # 基于比特能量和 Eb/N0 计算噪声功率
        noise_power = bit_energy / ebn0_linear
    else:
        raise ValueError("Either snr_db or ebn0_db must be provided")
    
    # 为复数和实数符号生成相应的噪声
    if is_complex:
        # 复数信号噪声分量
        noise_var_per_dim = noise_power / 2  # 每个维度噪声功率为总功率的一半
        noise_real = torch.randn_like(symbols.real) * torch.sqrt(noise_var_per_dim)
        noise_imag = torch.randn_like(symbols.imag) * torch.sqrt(noise_var_per_dim)
        noise = torch.complex(noise_real, noise_imag)
    else:
        # 实数信号噪声分量 - BPSK 只使用一个维度，因此全部噪声功率用于该维度
        noise = torch.randn_like(symbols) * torch.sqrt(noise_power)
    
    # 将噪声添加到符号
    noisy_symbols = symbols + noise
    
    return noisy_symbols, noise_power

def add_rayleigh_channel(symbols, snr_db=None, ebn0_db=None, bits_per_symbol=None):
    """
    Add Rayleigh fading channel with AWGN to symbols at specified SNR or Eb/N0
    
    Args:
        symbols (torch.Tensor): Real or complex symbols
        snr_db (float, optional): Signal-to-Noise ratio in dB
        ebn0_db (float, optional): Energy per bit to noise power spectral density ratio in dB
        bits_per_symbol (int, optional): Number of bits per symbol, required when ebn0_db is provided
        
    Returns:
        tuple: (torch.Tensor: Received symbols after Rayleigh fading and AWGN, 
               float: Noise power, torch.Tensor: Channel coefficients)
    """
    # 检查符号是复数还是实数
    is_complex = symbols.is_complex()
    
    # 将实数符号转换为复数进行瑞利衰落处理
    if not is_complex:
        symbols_complex = torch.complex(symbols, torch.zeros_like(symbols))
    else:
        symbols_complex = symbols
    
    # 生成瑞利衰落系数 (复高斯随机变量，方差为1)
    h_real = torch.randn_like(symbols_complex.real) / np.sqrt(2)
    h_imag = torch.randn_like(symbols_complex.imag) / np.sqrt(2)
    h = torch.complex(h_real, h_imag)
    
    # 应用瑞利衰落
    faded_symbols = h * symbols_complex
    
    # 计算符号能量 (假设星座已归一化)
    symbol_energy = 1.0  # 保持单位能量的假设
    
    # 检查提供的是哪个参数并相应地计算噪声功率
    if snr_db is not None:
        # 将 SNR 从 dB 转换为线性尺度
        snr_linear = 10.0**(snr_db/10.0)
        # 基于符号能量和 SNR 计算噪声功率
        noise_power = symbol_energy / snr_linear
    elif ebn0_db is not None:
        if bits_per_symbol is None:
            raise ValueError("bits_per_symbol must be provided when using ebn0_db")
        # 将 Eb/N0 从 dB 转换为线性尺度
        ebn0_linear = 10.0**(ebn0_db/10.0)
        
        # 计算每比特能量
        bit_energy = symbol_energy / bits_per_symbol
        
        # 基于比特能量和 Eb/N0 计算噪声功率
        noise_power = bit_energy / ebn0_linear
    else:
        raise ValueError("Either snr_db or ebn0_db must be provided")
    
    # 为复数符号生成AWGN噪声
    noise_var_per_dim = noise_power / 2
    noise_real = torch.randn_like(faded_symbols.real) * torch.sqrt(noise_var_per_dim)
    noise_imag = torch.randn_like(faded_symbols.imag) * torch.sqrt(noise_var_per_dim)
    noise = torch.complex(noise_real, noise_imag)
    
    # 将噪声添加到衰落后的符号
    received_symbols = faded_symbols + noise
    
    # 如果原始符号是实数，提取实部
    if not is_complex:
        received_symbols = received_symbols.real
        h = h.real  # 对于实数符号，只返回实数信道系数
    
    return received_symbols, noise_power, h

class UncodedSystemAWGN:
    """
    Implements an uncoded communication system over AWGN or Rayleigh channel
    
    This class provides a complete pipeline for an uncoded digital communication
    system using QAM/PAM modulation over an AWGN or Rayleigh channel, with BER evaluation.
    """
    
    def __init__(self, 
                 constellation_type="qam",
                 bits_per_symbol=4, 
                 normalize=True,
                 demapping_method="app",
                 channel_model="awgn",
                 device="cpu"):
        """
        Initialize the uncoded system
        
        Args:
            constellation_type (str): Type of constellation ('qam', 'pam', or 'custom')
            bits_per_symbol (int): Number of bits per symbol
            normalize (bool): Whether to normalize constellation to unit power
            demapping_method (str): 'app' or 'maxlog' demodulation method
            channel_model (str): Channel model type ('awgn' or 'rayleigh')
            device (str): Device to run computations on ('cpu' or 'cuda')
        """
        self.device = device
        self.bits_per_symbol = bits_per_symbol
        self.demapping_method = demapping_method
        self.channel_model = channel_model.lower()
        
        # Create constellation
        self.constellation = Constellation(
            constellation_type=constellation_type,
            num_bits_per_symbol=bits_per_symbol,
            normalize=normalize
        )
        
        # Create mapper and demapper
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = Demapper(
            demapping_method=demapping_method,
            constellation=self.constellation
        )
        
        # Move objects to specified device
        self.constellation.points = self.constellation.points.to(device)
    
    def apply_channel(self, symbols, snr_db=None, ebn0_db=None, bits_per_symbol=None):
        """
        Apply the selected channel model to symbols
        
        Args:
            symbols (torch.Tensor): Real or complex symbols
            snr_db (float, optional): Signal-to-Noise ratio in dB
            ebn0_db (float, optional): Energy per bit to noise power spectral density ratio in dB
            bits_per_symbol (int, optional): Number of bits per symbol
            
        Returns:
            tuple: (torch.Tensor: Received symbols, float: Noise power, Optional[torch.Tensor]: Channel coefficients)
        """
        if self.channel_model == "awgn":
            received_symbols, noise_power = add_awgn(
                symbols, snr_db=snr_db, ebn0_db=ebn0_db, 
                bits_per_symbol=bits_per_symbol
            )
            return received_symbols, noise_power, None
        elif self.channel_model == "rayleigh":
            received_symbols, noise_power, h = add_rayleigh_channel(
                symbols, snr_db=snr_db, ebn0_db=ebn0_db,
                bits_per_symbol=bits_per_symbol
            )
            return received_symbols, noise_power, h
        else:
            raise ValueError(f"Unsupported channel model: {self.channel_model}")
    
    def __call__(self, bits, snr_db=None, ebno_db=None):
        """
        Process a batch of random bits through the uncoded system at a given SNR or Eb/N0
        
        Args:
            bits: torch.Tensor - Random bits to be transmitted
            snr_db (float or torch.Tensor, optional): Signal-to-noise ratio(s) in dB
            ebno_db (float or torch.Tensor, optional): Energy per bit to noise power spectral density ratio in dB
            
        Returns:
            tuple: (torch.Tensor: LLRs, float: Noise power, Optional[torch.Tensor]: Channel coefficients)
        """
        if snr_db is None and ebno_db is None:
            raise ValueError("Either snr_db or ebno_db must be provided")
            
        # If snr_db/ebno_db is a scalar, convert to tensor
        if snr_db is not None and not isinstance(snr_db, torch.Tensor):
            snr_db = torch.tensor([snr_db], device=self.device)
        elif ebno_db is not None and not isinstance(ebno_db, torch.Tensor):
            ebno_db = torch.tensor([ebno_db], device=self.device)
        
        # Move to device if not already
        if snr_db is not None:
            snr_db = snr_db.to(self.device)
        if ebno_db is not None:
            ebno_db = ebno_db.to(self.device)
            
        # Modulate bits to symbols
        symbols = self.mapper(bits)
        
        # Apply channel model (AWGN or Rayleigh)
        received_symbols, noise_var, h = self.apply_channel(
            symbols, 
            snr_db=snr_db, 
            ebn0_db=ebno_db, 
            bits_per_symbol=self.bits_per_symbol
        )
        
        # Compute LLRs (log-likelihood ratios) with channel equalization
        llrs = self.enhanced_demapping_with_channel_info(
            received_symbols, noise_var, h, equalization_method="mrc"
        )
        
        return llrs, noise_var, h
    
    def evaluate_ber(self, batch_size, ebno_db_range, max_errors=100, max_bits=10**6, show_progress=True):
        """
        Evaluate BER performance over a range of Eb/N0 values
        
        Args:
            batch_size (int): Number of codewords to process in each batch
            ebno_db_range (list): Range of Eb/N0 values in dB
            max_errors (int): Early stopping based on number of bit errors
            max_bits (int): Maximum number of bits to process at each Eb/N0
            show_progress (bool): Whether to display progress bar
            
        Returns:
            dict: Results containing BER measurements
        """
        try:
            from tqdm import tqdm
        except ImportError:
            # Fallback if tqdm is not available
            def tqdm(iterable, *args, **kwargs):
                return iterable
        
        bers = []
        
        # Print header
        print(f"{'Eb/N0 (dB)':<10} {'BER':<15} {'Bits':<15}")
        print("-" * 45)
        
        # Loop through each Eb/N0 point
        for ebno_db in ebno_db_range:
            bit_errors = 0
            total_bits = 0
            
            # Progress bar for this Eb/No point
            pbar = tqdm(total=max_errors, disable=not show_progress, 
                       desc=f"Eb/N0 = {ebno_db:.1f} dB")
            
            while (bit_errors < max_errors and total_bits < max_bits):
                # Generate random bits
                u = torch.randint(0, 2, (batch_size, 
                                        (self.constellation.points.shape[0].bit_length() - 1) * 
                                        (batch_size * 10 // self.bits_per_symbol)), 
                                 device=self.device, dtype=torch.float32)
                
                # Process the bits through the uncoded system
                llrs, _, _ = self(u, ebno_db=ebno_db)
                
                # Hard decision decoding (convert LLRs to bits)
                u_hat = (llrs < 0).float()
                
                # Calculate bit errors
                errors = (u != u_hat).sum().item()
                bit_errors += errors
                total_bits += u.numel()
                
                # Update progress bar
                pbar.update(errors)
                
                # Early stopping for high SNR with no errors
                if ebno_db > 10.0 and total_bits >= 1e6 and bit_errors == 0:
                    break
            
            pbar.close()
            
            # Calculate BER
            ber = bit_errors / total_bits
            bers.append(ber)
            
            # Print results for this Eb/N0
            print(f"{ebno_db:<10.1f} {ber:<15.6e} {total_bits:<15d}")
        
        # Return results
        return {
            'ebno_dbs': ebno_db_range,
            'bers': bers
        }
    
    def plot_results(self, results, title=None, filename=None):
        """
        Plot BER results
        
        Args:
            results (dict): Results from evaluate_ber
            title (str, optional): Plot title
            filename (str, optional): If provided, save figure to this file
        """
        plt.figure(figsize=(10, 7))
        
        ebno_dbs = results['ebno_dbs']
        bers = results['bers']
        
        channel_label = f"{self.channel_model.upper()} Channel"
        plt.semilogy(ebno_dbs, bers, 'bo-', label=f'BER ({channel_label})', linewidth=2)
        
        # Add theoretical uncoded BPSK curve for reference (AWGN only)
        if self.channel_model == "awgn":
            try:
                from scipy.special import erfc
                ebno_range = np.linspace(min(ebno_dbs), max(ebno_dbs), 100)
                
                # Theoretical BER for BPSK over AWGN
                energy_factor = 2.0
                ber_theory = 0.5 * erfc(np.sqrt(energy_factor * 10 ** (ebno_range / 10)))
                
                plt.semilogy(ebno_range, ber_theory, 'g-.', 
                           label='BPSK Theory (AWGN)', linewidth=1.5, alpha=0.7)
            except ImportError:
                # Skip if scipy is not available
                pass
        
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.xlabel('Eb/No (dB)', fontsize=12)
        plt.ylabel('Bit Error Rate (BER)', fontsize=12)
        
        if title:
            plt.title(title, fontsize=14)
        else:
            modulation_name = f"{2**self.bits_per_symbol}-{self.constellation.constellation_type.upper()}"
            plt.title(f'Uncoded {modulation_name} BER Performance ({channel_label})', fontsize=14)
        
        plt.legend(loc='best')
        plt.ylim(bottom=1e-6)  # Set lower limit for visibility
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def channel_equalization(self, received_symbols: torch.Tensor, h: torch.Tensor, 
                            equalization_method: str = "zf") -> torch.Tensor:
        """
        Apply channel equalization for both AWGN and Rayleigh fading channels
        
        Args:
            received_symbols (torch.Tensor): Received symbols after channel
            h (torch.Tensor): Channel coefficients
            equalization_method (str): Equalization method ('zf', 'mmse', 'mrc', or 'identity')
            
        Returns:
            torch.Tensor: Equalized symbols
        """
        if h is None:
            return received_symbols
        
        if equalization_method.lower() == "identity":
            # Identity equalization for AWGN channel (no equalization needed)
            return received_symbols
            
        elif equalization_method.lower() == "zf":
            # Zero-Forcing equalization: y_eq = y / h
            # Avoid division by very small numbers
            h_magnitude = torch.abs(h)
            h_safe = torch.where(h_magnitude < 1e-8, 
                                torch.sign(h) * 1e-8, h)
            equalized_symbols = received_symbols / h_safe
            
        elif equalization_method.lower() == "mrc":
            # Maximum Ratio Combining: y_eq = h* * y / |h|^2
            h_conj = torch.conj(h)
            h_magnitude_squared = torch.abs(h)**2
            # Avoid division by very small numbers
            h_mag_sq_safe = torch.where(h_magnitude_squared < 1e-8, 
                                       1e-8, h_magnitude_squared)
            equalized_symbols = (h_conj * received_symbols) / h_mag_sq_safe
            
        elif equalization_method.lower() == "mmse":
            # MMSE equalization: y_eq = h* * y / (|h|^2 + sigma^2)
            # For MMSE, we need noise variance estimate
            # Use a simple approach with fixed SNR assumption
            h_conj = torch.conj(h)
            h_magnitude_squared = torch.abs(h)**2
            # Estimate noise variance (this could be improved with better estimation)
            noise_var_est = 0.1  # This should ideally be estimated or passed as parameter
            denominator = h_magnitude_squared + noise_var_est
            equalized_symbols = (h_conj * received_symbols) / denominator
            
        else:
            raise ValueError(f"Unknown equalization method: {equalization_method}")
            
        return equalized_symbols
    
    def enhanced_demapping_with_channel_info(self, received_symbols: torch.Tensor, 
                                           noise_var: float, h: torch.Tensor = None,
                                           equalization_method: str = "mrc") -> torch.Tensor:
        """
        Enhanced demapping that incorporates channel information for both AWGN and fading channels
        
        Args:
            received_symbols: Received symbols
            noise_var: Noise variance
            h: Channel coefficients (for fading channels)
            equalization_method: Channel equalization method
            
        Returns:
            LLRs for each bit
        """
        if h is not None and self.channel_model == "rayleigh":
            # Rayleigh fading channel - apply channel equalization
            equalized_symbols = self.channel_equalization(
                received_symbols, h, equalization_method
            )
            
            # Adjust noise variance after equalization
            if equalization_method.lower() == "zf":
                # ZF amplifies noise by |h|^(-2)
                h_magnitude_squared = torch.abs(h)**2
                h_mag_sq_safe = torch.where(h_magnitude_squared < 1e-8, 
                                           1e-8, h_magnitude_squared)
                adjusted_noise_var = noise_var / h_mag_sq_safe.mean().item()
            elif equalization_method.lower() == "mrc":
                # MRC provides diversity gain
                h_magnitude_squared = torch.abs(h)**2
                adjusted_noise_var = noise_var / h_magnitude_squared.mean().item()
            else:
                # For MMSE and others, use original noise variance
                adjusted_noise_var = noise_var
            
            # Use the standard demapper with equalized symbols
            return self.demapper(equalized_symbols, adjusted_noise_var)
        else:
            # AWGN channel - apply identity equalization (for consistency)
            # For AWGN, h=1 (no fading), so equalization is essentially identity operation
            if self.channel_model == "awgn":
                # Create identity channel coefficients for AWGN
                h_awgn = torch.ones_like(received_symbols)
                
                # Apply identity equalization for consistency
                equalized_symbols = self.channel_equalization(
                    received_symbols, h_awgn, "identity"
                )
                return self.demapper(equalized_symbols, noise_var)
            else:
                # Fallback for other cases or when h is None
                return self.demapper(received_symbols, noise_var)