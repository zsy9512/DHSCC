#!/usr/bin/env python3
"""
Modulation/Demodulation BER Test for DHSCC System

This test evaluates the bit error rate (BER) performance of modulation/demodulation
under different SNR conditions.
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
from channel.channel_modem import ChannelModem

def calculate_ber(original_bits, decoded_bits):
    """Calculate bit error rate between original and decoded bits."""
    errors = torch.sum(original_bits != decoded_bits).float()
    total_bits = original_bits.numel()
    return errors / total_bits

def test_modulation_demodulation_ber():
    """
    Test BER performance of modulation/demodulation under different SNR conditions
    """
    print("="*60)
    print("MODULATION/DEMODULATION BER vs SNR TEST")
    print("="*60)

    # Configuration
    n_symbols = 1000  # Number of symbols to test
    bits_per_symbol = 4  # 16-QAM
    snr_range = np.arange(0, 21, 2)  # SNR from 0 to 20 dB

    # Create modulation/demodulation components
    class Args:
        def __init__(self):
            self.bits_per_symbol = bits_per_symbol
            self.constellation_type = 'qam'
            self.normalize_constellation = True
            self.demapping_method = 'app'

    args = Args()
    modulator = Modulator(args)
    demodulator = Demodulator(args)
    channel = ChannelModem(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modulator = modulator.to(device)
    demodulator = demodulator.to(device)

    print(f"Testing {n_symbols} symbols with {bits_per_symbol} bits per symbol...")

    ber_results = []

    for snr in tqdm(snr_range, desc="SNR Testing"):
        # Generate random bits
        total_bits = n_symbols * bits_per_symbol
        original_bits = torch.randint(0, 2, (1, total_bits)).float().to(device)

        # Reshape for modulation
        bits_reshaped = original_bits.view(1, n_symbols, bits_per_symbol)

        # Modulation
        modulated = modulator.modulate(bits_reshaped)

        # Channel transmission
        received = channel.transmit(modulated, snr)
        received = received.to(device)

        # Demodulation
        demodulated = demodulator.demodulate(received)

        # Calculate BER
        ber = calculate_ber(original_bits.int().flatten(), demodulated.flatten())
        ber_results.append(ber.item())

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, ber_results, 'bo-', linewidth=2, markersize=8, label='16-QAM')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('Modulation/Demodulation BER vs SNR')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim([1e-5, 1])
    plt.xlim([0, 20])

    # Save plot
    plt.savefig('modulation_demodulation_ber.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("BER test completed. Results saved to modulation_demodulation_ber.png")
    return snr_range, ber_results

if __name__ == "__main__":
    print("Starting Modulation/Demodulation BER Test")
    snr_range, ber_results = test_modulation_demodulation_ber()
    print("Test completed successfully!")