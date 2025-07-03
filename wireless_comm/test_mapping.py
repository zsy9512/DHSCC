#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for evaluating the performance of uncoded_system.py
under different SNR conditions using 16-QAM modulation

This script tests the bit error rate (BER) performance of the PyTorch
implementation of an uncoded system with QAM modulation across various SNR levels.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import argparse

# Import the UncodedSystem implementation
from uncoded_system import UncodedSystemAWGN
from mapping import Constellation


def test_uncoded_system_performance(bits_per_symbol=4, batch_size=1000, num_batches=10, snr_db_range=None,
                         demapping_methods=['maxlog'], device=None):
    """
    Test uncoded system performance at different SNR levels
    
    Args:
        bits_per_symbol (int): Number of bits per QAM symbol (default: 4 for 16QAM)
        batch_size (int): Number of samples per batch (default: 1000)
        num_batches (int): Number of batches to average over
        snr_db_range (list): Range of SNR values to test in dB
        demapping_methods (list): List of demapping methods to test
        device (str): Device to run calculations on ('cpu' or 'cuda')
        
    Returns:
        dict: Dictionary containing BER results for each method and SNR
    """
    if snr_db_range is None:
        snr_db_range = range(0, 21, 2)  # 0 to 20 dB in steps of 2 dB
        
    if demapping_methods is None:
        demapping_methods = ["app", "maxlog"]
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Fix the bits per sample to 256 as required
    bits_per_sample = 256
    
    # Calculate number of symbols needed for 256 bits
    num_symbols_per_sample = bits_per_sample // bits_per_symbol
    if bits_per_sample % bits_per_symbol != 0:
        print(f"Warning: {bits_per_sample} bits is not divisible by {bits_per_symbol} bits per symbol")
        num_symbols_per_sample = (bits_per_sample // bits_per_symbol) + 1
        
    print(f"Running UncodedSystemAWGN performance test on {device}")
    print(f"Testing SNR range: {list(snr_db_range)} dB")
    print(f"Using demapping methods: {demapping_methods}")
    print(f"Using {batch_size} samples per batch, each with {bits_per_sample} bits ({num_symbols_per_sample} symbols)")
    
    # Initialize results dictionary
    results = {method: [] for method in demapping_methods}
    
    # Initialize uncoded system objects for each demapping method
    systems = {
        method: UncodedSystemAWGN(
            constellation_type="qam",
            bits_per_symbol=bits_per_symbol,
            normalize=True,
            demapping_method=method,
            device=device
        ) for method in demapping_methods
    }
    
    # Set a different seed for each SNR to ensure independent noise realizations
    base_seed = int(time.time())
    
    # Test each SNR value
    for i, snr_db in enumerate(tqdm(snr_db_range, desc="Testing SNR levels")):
        # Set seed based on SNR value to ensure different noise patterns
        torch.manual_seed(base_seed + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(base_seed + i)
            
        # Initialize error counters for this SNR
        bit_errors = {method: 0 for method in demapping_methods}
        total_bits_processed = 0
        
        # Run multiple batches
        for _ in range(num_batches):
            # Generate random bits - shape: [batch_size, 256]
            bits = torch.randint(0, 2, (batch_size, bits_per_sample), device=device)
            total_bits_processed += bits.numel()
            
            # Test each demapping method
            for method in demapping_methods:
                system = systems[method]
                # Process bits through the uncoded system
                llrs, noise_var, _ = system(bits, snr_db)
                
                # Reshape LLRs to match original bits shape for comparison
                llrs = llrs.reshape(batch_size, bits_per_sample)
                
                # Hard decision on LLRs
                bits_hat = (llrs > 0).int()
                
                # Count bit errors
                bit_errors[method] += torch.sum((bits_hat != bits)).item()
        
        # Calculate BER for each method at this SNR
        for method in demapping_methods:
            ber = bit_errors[method] / total_bits_processed
            results[method].append(ber)
            print(f"SNR: {snr_db} dB, Method: {method}, BER: {ber:.6e}")
    
    return {"snr_db_range": list(snr_db_range), "results": results}


def plot_ber_results(results, bits_per_symbol):
    """
    Plot BER vs SNR curves
    
    Args:
        results (dict): Dictionary with BER results
        bits_per_symbol (int): Number of bits per symbol used in the test
    """
    plt.figure(figsize=(10, 6))
    
    snr_db_range = results["snr_db_range"]
    
    # Plot for each demapping method
    for method, ber_values in results["results"].items():
        plt.semilogy(snr_db_range, ber_values, 'o-', label=f"{method}")
    
    # Add theoretical BER curve for M-QAM in AWGN if available
    M = 2**bits_per_symbol  # Constellation size
    
    plt.grid(True, which="both")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title(f"{M}-QAM Uncoded System Performance in AWGN Channel")
    plt.legend()
    plt.ylim([1e-6, 1])
    
    # Save the plot
    plt.savefig(f"qam{M}_ber_performance.png", dpi=300)
    print(f"BER plot saved to qam{M}_ber_performance.png")
    plt.show()


def test_execution_speed(bits_per_symbol=4, num_symbols=10000, demapping_method="app", device=None):
    """
    Test the execution speed of uncoded system operations
    
    Args:
        bits_per_symbol (int): Number of bits per QAM symbol
        num_symbols (int): Number of symbols to process
        demapping_method (str): Demapping method to test
        device (str): Device to run calculations on ('cpu' or 'cuda')
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Testing execution speed on {device} with {num_symbols} symbols...")
    
    # Create the uncoded system
    system = UncodedSystemAWGN(
        constellation_type="qam",
        bits_per_symbol=bits_per_symbol,
        normalize=True,
        demapping_method=demapping_method,
        device=device
    )
    
    # Generate random bits
    total_bits = num_symbols * bits_per_symbol
    bits = torch.randint(0, 2, (1, total_bits), device=device)
    
    # Test system processing speed
    start_time = time.time()
    llrs, _ = system(bits, 10.0)  # 10 dB SNR
    processing_time = time.time() - start_time
    
    print(f"Uncoded system processing time for {num_symbols} symbols: {processing_time:.4f} seconds")
    
    return {"processing_time": processing_time}


def main():
    """
    Main function that runs all tests
    """
    parser = argparse.ArgumentParser(description="Test uncoded system performance")
    parser.add_argument("--bits_per_symbol", type=int, default=4,
                      help="Number of bits per symbol (default: 4 for 16QAM)")
    parser.add_argument("--min_snr", type=int, default=0,
                      help="Minimum SNR to test in dB (default: 0)")
    parser.add_argument("--max_snr", type=int, default=20,
                      help="Maximum SNR to test in dB (default: 20)")
    parser.add_argument("--snr_step", type=int, default=2,
                      help="SNR step size in dB (default: 2)")
    parser.add_argument("--batch_size", type=int, default=1000,
                      help="Number of samples per batch (default: 1000)")
    parser.add_argument("--num_batches", type=int, default=10,
                      help="Number of batches to average over (default: 10)")
    parser.add_argument("--device", type=str, default=None,
                      help="Device to run on ('cpu' or 'cuda', default: auto-detect)")
    parser.add_argument("--speed_test", action="store_true",
                      help="Run speed test")
    parser.add_argument("--methods", type=str, nargs='+', default=['app', 'maxlog'],
                      help="Demapping methods to test (default: app maxlog)")
    
    args = parser.parse_args()
    
    # Create a QAM constellation and visualize it
    constellation = Constellation("qam", args.bits_per_symbol)
    fig = constellation.show(labels=True)
    fig.savefig(f"qam{2**args.bits_per_symbol}_constellation.png", dpi=300)
    print(f"Constellation plot saved to qam{2**args.bits_per_symbol}_constellation.png")
    
    # Define SNR range
    snr_db_range = range(args.min_snr, args.max_snr + 1, args.snr_step)
    
    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        
    # Run performance test
    results = test_uncoded_system_performance(
        bits_per_symbol=args.bits_per_symbol,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        snr_db_range=snr_db_range,
        demapping_methods=args.methods,
        device=device
    )
    
    # Plot results
    plot_ber_results(results, args.bits_per_symbol)
    
    # Run speed test if requested
    if args.speed_test:
        for method in args.methods:
            test_execution_speed(
                bits_per_symbol=args.bits_per_symbol,
                demapping_method=method,
                device=device
            )


if __name__ == "__main__":
    main()