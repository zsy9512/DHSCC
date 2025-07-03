#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for evaluating the performance of code_system.py
under different Eb/N0 conditions using 5G Polar codes

This script tests the bit error rate (BER) and block error rate (BLER) performance 
of the PyTorch implementation of a 5G Polar coded system with QAM modulation
across various Eb/N0 levels.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import argparse

# Import the CodedSystemAWGN implementation
from wireless_comm.code_system import CodedSystemAWGN
from polar_decoding import PolarSCDecoder, PolarSCLDecoder
from polar_bit_pairing import get_5g_design_pattern
from polar_encoding import PolarEncoder


def test_coded_system_performance(k, n, bits_per_symbol=4, batch_size=1000, ebno_db_range=None,
                             decoder_types=['SC'], constellation_type="qam", device=None,
                             max_errors=100, max_bits=10**6):
    """
    Test coded system performance at different Eb/N0 levels
    
    Args:
        k (int): Number of information bits
        n (int): Codeword length
        bits_per_symbol (int): Number of bits per QAM symbol
        batch_size (int): Number of codewords to process in each batch
        ebno_db_range (list): Range of Eb/N0 values to test in dB
        decoder_types (list): List of decoder types to test (SC, SCL, etc.)
        constellation_type (str): Constellation type to use
        device (str): Device to run calculations on ('cpu' or 'cuda')
        max_errors (int): Early stopping based on number of bit errors
        max_bits (int): Maximum number of bits to process at each Eb/N0
        
    Returns:
        dict: Dictionary containing BER and BLER results for each decoder type and Eb/N0
    """
    if ebno_db_range is None:
        ebno_db_range = range(0, 11, 1)  # 0 to 10 dB in steps of 1 dB
        
    if decoder_types is None:
        decoder_types = ["SC", "SCL"]
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Running CodedSystemAWGN performance test on {device}")
    print(f"Code parameters: k={k}, n={n}, rate={k/n:.3f}")
    print(f"Testing Eb/N0 range: {list(ebno_db_range)} dB")
    print(f"Using decoder types: {decoder_types}")
    print(f"Using {batch_size} codewords per batch")
    print(f"Using {2**bits_per_symbol}-{constellation_type.upper()} modulation")
    
    # Initialize results dictionary
    results = {decoder_type: {'bers': [], 'blers': []} for decoder_type in decoder_types}
    
    # Initialize coded system objects for each decoder type
    systems = {
        decoder_type: CodedSystemAWGN(
            k=k,
            n=n,
            constellation_type=constellation_type,
            bits_per_symbol=bits_per_symbol,
            normalize=True,
            demapping_method="app",
            decoder_type=decoder_type,
            device=device,
            verbose=False
        ) for decoder_type in decoder_types
    }
    
    # Set a different seed for each decoder to ensure independent noise realizations
    base_seed = int(time.time())
    
    # Test each decoder type
    for decoder_idx, decoder_type in enumerate(decoder_types):
        print(f"\nTesting decoder type: {decoder_type}")
        
        # Set seed for reproducibility while ensuring different seeds per decoder
        decoder_seed = base_seed + decoder_idx
        torch.manual_seed(decoder_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(decoder_seed)
            
        # Get the system for this decoder type
        system = systems[decoder_type]
        
        # Run the BER evaluation
        result = system.evaluate_ber(
            batch_size=batch_size,
            ebno_db_range=ebno_db_range,
            max_errors=max_errors,
            max_bits=max_bits,
            show_progress=True
        )
        
        # Store results
        results[decoder_type]['bers'] = result['bers']
        results[decoder_type]['blers'] = result['blers']
    
    return {"ebno_db_range": list(ebno_db_range), "results": results}


def plot_ber_results(results, k, n, bits_per_symbol, constellation_type, filename=None):
    """
    Plot BER and BLER vs Eb/N0 curves
    
    Args:
        results (dict): Dictionary with BER results
        k (int): Number of information bits
        n (int): Codeword length
        bits_per_symbol (int): Number of bits per symbol used in the test
        constellation_type (str): Type of constellation used
        filename (str, optional): If provided, save figure to this filename
    """
    plt.figure(figsize=(12, 8))
    
    ebno_db_range = results["ebno_db_range"]
    
    # Plot for each decoder type
    for decoder_type, decoder_results in results["results"].items():
        plt.semilogy(ebno_db_range, decoder_results['bers'], 'o-', 
                   label=f"BER - {decoder_type}", linewidth=2)
        plt.semilogy(ebno_db_range, decoder_results['blers'], 's--', 
                   label=f"BLER - {decoder_type}", linewidth=2)
    
    # Add theoretical uncoded BPSK curve for reference
    try:
        from scipy.special import erfc
        ebno_range = np.linspace(min(ebno_db_range), max(ebno_db_range), 100)
        ber_theory = 0.5 * erfc(np.sqrt(10 ** (ebno_range / 10)))
        plt.semilogy(ebno_range, ber_theory, 'g-.', 
                   label='BPSK Theory (Uncoded)', linewidth=1.5, alpha=0.7)
    except ImportError:
        print("Could not import scipy.special.erfc for plotting the theoretical curve")
    
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.xlabel("Eb/N0 (dB)", fontsize=12)
    plt.ylabel("Error Rate", fontsize=12)
    plt.title(f"{2**bits_per_symbol}-{constellation_type.upper()} with Polar({n},{k}) Code Performance", fontsize=14)
    plt.legend()
    plt.ylim([1e-6, 1])
    
    # Save the plot if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        filename = f"polar_{n}_{k}_{constellation_type}{2**bits_per_symbol}_performance.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    print(f"BER plot saved to {filename}")
    plt.show()


def test_execution_speed(k, n, bits_per_symbol=4, decoder_type="SC", device=None):
    """
    Test the execution speed of coded system operations
    
    Args:
        k (int): Number of information bits
        n (int): Codeword length
        bits_per_symbol (int): Number of bits per symbol
        decoder_type (str): Type of decoder to test
        device (str): Device to run calculations on ('cpu' or 'cuda')
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Testing execution speed on {device} with Polar({n},{k}) code...")
    
    # Create the coded system
    system = CodedSystemAWGN(
        k=k,
        n=n,
        constellation_type="qam",
        bits_per_symbol=bits_per_symbol,
        normalize=True,
        demapping_method="app",
        decoder_type=decoder_type,
        device=device
    )
    
    # Number of batches to test
    batch_size = 10000
    
    # Generate random bits
    bits = torch.randint(0, 2, (batch_size, k), device=device, dtype=torch.float32)
    
    # Test system encoding speed
    start_time = time.time()
    encoded_bits = system.encoder(bits)
    encoding_time = time.time() - start_time
    
    # Generate modulated symbols
    symbols = system.mapper(encoded_bits)
    
    # Add noise to create noisy symbols at 5 dB Eb/N0
    noisy_symbols, noise_var = system.add_awgn(symbols, ebn0_db=torch.tensor([5.0],device='cuda'), bits_per_symbol=bits_per_symbol, code_rate=system.code_rate)
    
    # Generate LLRs
    llrs = system.demapper(noisy_symbols, noise_var)
    if llrs.ndim > 2 and llrs.shape[-1] == system.bits_per_symbol:
        llrs = llrs.reshape(batch_size, -1)
    
    # Test decoding speed
    start_time = time.time()
    _ = system.decoder(llrs)
    decoding_time = time.time() - start_time
    
    # Test full system processing speed
    start_time = time.time()
    _, _ = system(bits, ebno_db=5.0)
    processing_time = time.time() - start_time
    
    print(f"Encoding time for {batch_size} frames: {encoding_time:.4f} seconds")
    print(f"Decoding time for {batch_size} frames: {decoding_time:.4f} seconds")
    print(f"Full system processing time for {batch_size} frames: {processing_time:.4f} seconds")
    
    return {
        "encoding_time": encoding_time,
        "decoding_time": decoding_time,
        "processing_time": processing_time,
        "frames": batch_size
    }


def test_scl_decoder():
    """Test the SCL decoder to verify it works correctly"""
    print("\n=== Testing SCL Decoder ===")
    
    # Set a seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Set parameters
    n = 256
    k = 128
    list_size = 8
    
    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    
    # Generate frozen positions using 5G sequence
    frozen_pos = np.array(get_5g_design_pattern(n, k)).astype(np.int32)
    
    # Create encoders and decoders
    polar_encoder = PolarEncoder(frozen_pos=frozen_pos, n=n)
    sc_decoder = PolarSCDecoder(frozen_pos=frozen_pos, n=n)
    scl_decoder = PolarSCLDecoder(frozen_pos=frozen_pos, n=n, list_size=list_size)
    
    # Move models to device
    polar_encoder.to(device)
    sc_decoder.to(device)
    scl_decoder.to(device)
    
    # Generate random information bits
    info_bits = torch.randint(0, 2, (1, k), dtype=torch.float32, device=device)
    print(f"Generated {k} random information bits")
    
    # Encode the bits
    codeword = polar_encoder(info_bits)
    print(f"Encoded into codeword of length {n}")
    
    # Test at different SNR values
    ebno_dbs = np.arange(0, 4.1, 0.5)
    sc_bers = []
    scl_bers = []
    
    for ebno_db in ebno_dbs:
        # Calculate noise variance
        code_rate = k / n
        ebno_linear = 10**(ebno_db/10.0)
        noise_var = 1.0 / (2 * code_rate * ebno_linear)
        
        # Generate AWGN noise
        noise = torch.normal(0, np.sqrt(noise_var), 
                           size=codeword.shape, 
                           device=device)
        
        # Apply binary phase-shift keying (BPSK) and add noise
        received = 1.0 - 2.0 * codeword + noise
        
        # Convert to LLRs (log-likelihood ratios)
        # For BPSK, LLR = 2*r/sigma^2
        llr = 2 * received / (2 * noise_var)
        
        # Decode using SC
        sc_decoded = sc_decoder(llr)
        
        # Decode using SCL
        scl_decoded = scl_decoder(llr)
        
        # Calculate bit error rates
        sc_errors = torch.sum(sc_decoded != info_bits).item()
        scl_errors = torch.sum(scl_decoded != info_bits).item()
        
        sc_ber = sc_errors / k
        scl_ber = scl_errors / k
        
        sc_bers.append(sc_ber)
        scl_bers.append(scl_ber)
        
        print(f"Eb/N0: {ebno_db} dB - SC BER: {sc_ber:.6f}, SCL BER: {scl_ber:.6f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogy(ebno_dbs, sc_bers, 'b-o', label='SC Decoder')
    plt.semilogy(ebno_dbs, scl_bers, 'r-s', label='SCL Decoder')
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title(f'Polar({n},{k}) Performance Comparison - SCL List Size {list_size}')
    plt.legend()
    plt.savefig(f'polar_{n}_{k}_performance.png')
    plt.show()


def main():
    """
    Main function that runs all tests
    """
    parser = argparse.ArgumentParser(description="Test coded system performance")
    parser.add_argument("--k", type=int, default=128,
                      help="Number of information bits (default: 128)")
    parser.add_argument("--n", type=int, default=256,
                      help="Codeword length (default: 256)")
    parser.add_argument("--bits_per_symbol", type=int, default=4,
                      help="Number of bits per symbol (default: 4 for 16QAM)")
    parser.add_argument("--min_ebno", type=float, default=0.0,
                      help="Minimum Eb/N0 to test in dB (default: 0)")
    parser.add_argument("--max_ebno", type=float, default=10.0,
                      help="Maximum Eb/N0 to test in dB (default: 10)")
    parser.add_argument("--ebno_step", type=float, default=1.0,
                      help="Eb/N0 step size in dB (default: 1)")
    parser.add_argument("--batch_size", type=int, default=1000,
                      help="Number of codewords per batch (default: 1000)")
    parser.add_argument("--max_errors", type=int, default=100,
                      help="Maximum errors before early stopping (default: 100)")
    parser.add_argument("--max_bits", type=int, default=1000000,
                      help="Maximum bits to process per Eb/N0 point (default: 1000000)")
    parser.add_argument("--device", type=str, default=None,
                      help="Device to run on ('cpu' or 'cuda', default: auto-detect)")
    parser.add_argument("--speed_test", action="store_true",
                      help="Run speed test")
    parser.add_argument("--decoders", type=str, nargs='+', default=['SC'],
                      help="Decoder types to test (default: SC)")
    parser.add_argument("--constellation", type=str, default="qam",
                      help="Constellation type (default: qam)")
    parser.add_argument("--output", type=str, default=None,
                      help="Output filename for the BER plot")
    
    args = parser.parse_args()
    
    # Create Eb/N0 range
    ebno_db_range = np.arange(args.min_ebno, args.max_ebno + args.ebno_step, args.ebno_step)
    
    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        
    # Run performance test
    results = test_coded_system_performance(
        k=args.k,
        n=args.n,
        bits_per_symbol=args.bits_per_symbol,
        batch_size=args.batch_size,
        ebno_db_range=ebno_db_range,
        decoder_types=args.decoders,
        constellation_type=args.constellation,
        device=device,
        max_errors=args.max_errors,
        max_bits=args.max_bits
    )
    
    # Plot results
    plot_ber_results(
        results, 
        k=args.k,
        n=args.n,
        bits_per_symbol=args.bits_per_symbol,
        constellation_type=args.constellation,
        filename=args.output
    )
    
    # Run speed test if requested
    if args.speed_test:
        for decoder_type in args.decoders:
            test_execution_speed(
                k=args.k,
                n=args.n,
                bits_per_symbol=args.bits_per_symbol,
                decoder_type=decoder_type,
                device=device
            )


if __name__ == "__main__":
    main()
    test_scl_decoder()