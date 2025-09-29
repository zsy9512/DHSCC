#!/usr/bin/env python3
"""
Polar Coding + Modulation/Demodulation BER Test for DHSCC System

This test evaluates the bit error rate (BER) performance of polar     plt.semilogy(snr_range, ber_results, 'ro-', linewidth=2, markersize=8, label='Polar(128,64) + 16-QAM')oding
combined with modulation/demodulation under different SNR conditions.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sender.modulation import Modulator
from receiver.demodulation import Demodulator
from sender.polar_encoding import PolarEncoder, get_frozen_pos
from receiver.polar_decoding import PolarSCDecoder
from channel_modem.channel_modem import ChannelModem

def calculate_ber(original_bits, decoded_bits):
    """Calculate bit error rate between original and decoded bits."""
    errors = torch.sum(original_bits != decoded_bits).float()
    total_bits = original_bits.numel()
    return errors / total_bits

def test_polar_modulation_demodulation_ber():
    """
    Test BER performance of polar coding + modulation/demodulation under different SNR conditions
    """
    print("="*60)
    print("POLAR CODING + QPSK MODULATION/DEMODULATION BER vs SNR TEST")
    print("="*60)

    # Configuration
    n = 128  # Codeword length
    k = 64   # Information bits (changed from 96 to 64 for (64,128) polar code)
    n_symbols = 100  # Number of codewords to test
    bits_per_symbol = 2  # QPSK
    snr_range = np.arange(0, 21, 2)  # SNR from 0 to 20 dB

    # Create components
    class Args:
        def __init__(self):
            self.bits_per_symbol = bits_per_symbol
            self.constellation_type = 'qam'
            self.normalize_constellation = True
            self.demapping_method = 'app'

    args = Args()

    # Polar coding
    frozen_pos = get_frozen_pos(k, n)
    polar_encoder = PolarEncoder(frozen_pos, n)
    polar_decoder = PolarSCDecoder(frozen_pos, n)

    # Modulation/Demodulation
    modulator = Modulator(args)
    demodulator = Demodulator(args)
    channel = ChannelModem(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    polar_encoder = polar_encoder.to(device)
    polar_decoder = polar_decoder.to(device)
    modulator = modulator.to(device)
    demodulator = demodulator.to(device)

    print(f"Testing {n_symbols} codewords (n={n}, k={k}) with {bits_per_symbol} bits per symbol...")

    ber_results = []

    for snr in tqdm(snr_range, desc="SNR Testing"):
        total_errors = 0
        total_bits = 0

        for _ in range(n_symbols):
            # Generate random information bits
            info_bits = torch.randint(0, 2, (1, k)).float().to(device)

            # Polar encoding
            encoded = polar_encoder(info_bits)

            # Reshape for modulation (n bits -> n/bits_per_symbol symbols)
            symbols_per_codeword = n // bits_per_symbol
            encoded_reshaped = encoded.view(1, symbols_per_codeword, bits_per_symbol)

            # Modulation
            modulated = modulator.modulate(encoded_reshaped)

            # Channel transmission
            received = channel.transmit(modulated, snr)
            received = received.to(device)

            # Calculate noise variance for demodulation
            # For AWGN, noise power is 1/(2*SNR_linear) per dimension
            snr_linear = 10 ** (snr / 10)
            noise_var = 1 / (2 * snr_linear)  # Per complex dimension

            # Demodulation to get LLRs
            # First, demodulate to get LLRs for each bit
            demodulated = demodulator.demodulate(received, noise_var)

            # Polar decoding
            decoded = polar_decoder(demodulated)

            # Calculate BER
            errors = torch.sum(info_bits.int().flatten() != decoded.flatten()).float()
            total_errors += errors.item()
            total_bits += k

        ber = total_errors / total_bits
        ber_results.append(ber)
        print(f"SNR {snr} dB: BER = {ber:.6f}")

    # Print final results
    print("\nFinal BER Results:")
    for snr, ber in zip(snr_range, ber_results):
        print(f"SNR {snr} dB: BER = {ber:.6f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, ber_results, 'ro-', linewidth=2, markersize=8, label='Polar(128,64) + QPSK')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('Polar Coding + QPSK Modulation/Demodulation BER vs SNR')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim([1e-5, 1])
    plt.xlim([0, 20])

    # Save plot
    plt.savefig('polar_modulation_demodulation_ber.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Polar coding BER test completed. Results saved to polar_modulation_demodulation_ber.png")
    return snr_range, ber_results

if __name__ == "__main__":
    print("Starting Polar Coding + Modulation/Demodulation BER Test")
    snr_range, ber_results = test_polar_modulation_demodulation_ber()
    print("Test completed successfully!")