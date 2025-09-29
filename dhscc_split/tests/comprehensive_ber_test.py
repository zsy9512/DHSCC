#!/usr/bin/env python3
"""
Comprehensive BER Performance Test for DHSCC System

This test evaluates the bit error rate (BER) performance of different communication components:
1. Modulation/Demodulation BER vs SNR
2. Polar Coding + Modulation/Demodulation BER vs SNR
3. End-to-end model connectivity (sender -> quantizer -> receiver)
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
from sender.quantizer import VibrationVectorQuantizer
from sender.sender_model import SenderNet
from receiver.receiver_model import ReceiverNet
from channel_modem.channel_modem import ChannelModem

def calculate_ber(original_bits, decoded_bits):
    """Calculate bit error rate between original and decoded bits."""
    errors = torch.sum(original_bits != decoded_bits).float()
    total_bits = original_bits.numel()
    return errors / total_bits

def calculate_llr_qam(symbols, constellation, noise_var):
    """
    Calculate LLR for QAM symbols

    Args:
        symbols: Complex symbols [batch, seq_len]
        constellation: Constellation object
        noise_var: Noise variance

    Returns:
        LLR values [batch, seq_len * bits_per_symbol]
    """
    device = symbols.device
    batch_size, seq_len = symbols.shape
    bits_per_symbol = constellation.num_bits_per_symbol

    # Expand symbols for broadcasting
    symbols_exp = symbols.unsqueeze(-1)  # [batch, seq_len, 1]

    # Get constellation points and their binary representations
    points = constellation.points.to(device)  # [2^bits_per_symbol]
    num_symbols = len(points)

    # Calculate distances
    distances = torch.abs(symbols_exp - points.unsqueeze(0).unsqueeze(0)) ** 2  # [batch, seq_len, num_symbols]

    # Calculate LLR for each bit
    llr_all = []
    for bit_idx in range(bits_per_symbol):
        # Get symbols where this bit is 0 and 1
        bit_mask = torch.tensor([(i >> bit_idx) & 1 for i in range(num_symbols)], device=device)

        # Probabilities for bit = 0 and bit = 1
        prob_0 = torch.sum(torch.exp(-distances / (2 * noise_var)) * (1 - bit_mask).float().unsqueeze(0).unsqueeze(0), dim=-1)
        prob_1 = torch.sum(torch.exp(-distances / (2 * noise_var)) * bit_mask.float().unsqueeze(0).unsqueeze(0), dim=-1)

        # LLR = log(P(bit=0)/P(bit=1))
        llr = torch.log(prob_0 / prob_1)
        llr_all.append(llr)

    # Stack LLRs: [batch, seq_len, bits_per_symbol] -> [batch, seq_len * bits_per_symbol]
    llr_tensor = torch.stack(llr_all, dim=-1)
    return llr_tensor.view(batch_size, seq_len * bits_per_symbol)

def test_modulation_demodulation_ber():
    """
    Test 1: BER performance of modulation/demodulation under different SNR conditions
    """
    print("="*60)
    print("TEST 1: Modulation/Demodulation BER vs SNR")
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
    # ChannelModem handles device internally

    ber_results = []

    print(f"Testing {n_symbols} symbols with {bits_per_symbol} bits per symbol...")

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

        print(".4f")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, ber_results, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('Modulation/Demodulation BER vs SNR')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.ylim(1e-5, 1)
    plt.savefig('modulation_demodulation_ber.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"BER test completed. Results saved to modulation_demodulation_ber.png")
    return snr_range, ber_results

def test_polar_modulation_demodulation_ber():
    """
    Test 2: BER performance of polar coding + modulation/demodulation under different SNR
    """
    print("\n" + "="*60)
    print("TEST 2: Polar Coding + Modulation/Demodulation BER vs SNR")
    print("="*60)

    # Configuration
    n = 128  # Codeword length
    k = 64   # Information bits
    n_symbols = 100  # Number of codewords to test
    bits_per_symbol = 4  # 16-QAM
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

    ber_results = []

    print(f"Testing {n_symbols} codewords (n={n}, k={k}) with {bits_per_symbol} bits per symbol...")

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

            # Calculate noise variance for LLR computation
            # For AWGN, noise power is 1/(2*SNR_linear) per dimension
            snr_linear = 10 ** (snr / 10)
            noise_var = 1 / (2 * snr_linear)  # Per complex dimension

            # Calculate LLR for polar decoding
            llr = calculate_llr_qam(received, modulator.constellation, noise_var)

            # Polar decoding
            decoded = polar_decoder(llr)

            # Calculate BER
            errors = torch.sum(info_bits.int().flatten() != decoded.flatten()).float()
            total_errors += errors.item()
            total_bits += k

        ber = total_errors / total_bits
        ber_results.append(ber)

        print(".4f")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, ber_results, 'ro-', linewidth=2, markersize=8, label='Polar + Mod/Demod')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('Polar Coding + Modulation/Demodulation BER vs SNR')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.ylim(1e-5, 1)
    plt.legend()
    plt.savefig('polar_modulation_demodulation_ber.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Polar coding BER test completed. Results saved to polar_modulation_demodulation_ber.png")
    return snr_range, ber_results

def test_model_connectivity():
    """
    Test 3: End-to-end model connectivity (sender -> quantizer -> receiver)
    """
    print("\n" + "="*60)
    print("TEST 3: Model Connectivity Test (Sender -> Quantizer -> Receiver)")
    print("="*60)

    # Configuration
    batch_size = 4
    seq_len = 128
    num_classes = 10

    # Create sender arguments
    class SenderArgs:
        def __init__(self):
            self.intermediate_dim = 128
            self.quantizer_dimension = 128
            self.quantizer_n_codes_group = 8
            self.quantizer_bins = 16
            self.quantizer_n_q = 32
            self.bits_per_index = 4
            self.max_groups = 8

    # Create receiver arguments
    class ReceiverArgs:
        def __init__(self):
            self.intermediate_dim = 128
            self.num_classes = num_classes

    sender_args = SenderArgs()
    receiver_args = ReceiverArgs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Initialize models
        print("Initializing models...")
        sender_model = SenderNet(sender_args).to(device)
        quantizer = VibrationVectorQuantizer(sender_args).to(device)
        receiver_model = ReceiverNet(receiver_args, num_classes).to(device)

        sender_model.eval()
        quantizer.eval()
        receiver_model.eval()

        print("[OK] All models initialized successfully!")

        # Generate test input
        test_input = torch.randn(batch_size, 1, seq_len).to(device)
        channel_noise = torch.randn(batch_size, 1).to(device) * 0.1 + 1.0

        print(f"Test input shape: {test_input.shape}")
        print(f"Channel noise shape: {channel_noise.shape}")

        # Test forward pass
        print("\nTesting forward pass...")

        with torch.no_grad():
            # Sender: extract features
            features, importance_weights, pre_task_logits, feature_shape, original_encoded_features = sender_model(test_input, channel_noise)
            print(f"[OK] Sender output - features: {features.shape}, importance: {importance_weights.shape}")

            # Quantizer: quantize features
            quantized, indices, commitment_loss = quantizer(features)
            print(f"[OK] Quantizer output - quantized: {quantized.shape}, indices: {indices.shape}")

            # Receiver: classify
            target_shape = (batch_size, 512, 8)
            logits = receiver_model(quantized, channel_noise, target_shape)
            predictions = torch.argmax(logits, dim=1)
            print(f"[OK] Receiver output - logits: {logits.shape}, predictions: {predictions.tolist()}")

        print("\n[SUCCESS] Model connectivity test PASSED!")
        print("All components are properly connected and functional.")

        return True

    except Exception as e:
        print(f"[FAIL] Model connectivity test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all comprehensive tests."""
    print("Starting DHSCC Comprehensive BER Performance Tests")
    print("="*60)

    # Test 1: Modulation/Demodulation BER
    try:
        snr_range, mod_demod_ber = test_modulation_demodulation_ber()
        print("✓ Test 1 completed successfully")
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
        return

    # Test 2: Polar + Modulation/Demodulation BER
    try:
        snr_range, polar_ber = test_polar_modulation_demodulation_ber()
        print("✓ Test 2 completed successfully")
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
        return

    # Test 3: Model connectivity
    try:
        connectivity_success = test_model_connectivity()
        if connectivity_success:
            print("✓ Test 3 completed successfully")
        else:
            print("✗ Test 3 failed")
            return
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
        return

    # Combined plot
    print("\n" + "="*60)
    print("GENERATING COMBINED BER PLOT")
    print("="*60)

    plt.figure(figsize=(12, 8))
    plt.semilogy(snr_range, mod_demod_ber, 'bo-', linewidth=2, markersize=8, label='Modulation/Demodulation Only')
    plt.semilogy(snr_range, polar_ber, 'ro-', linewidth=2, markersize=8, label='Polar + Mod/Demod')
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title('DHSCC System BER Performance Comparison', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.ylim(1e-5, 1)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('dhscc_ber_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Combined BER plot saved to dhscc_ber_comparison.png")

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Generated files:")
    print("- modulation_demodulation_ber.png")
    print("- polar_modulation_demodulation_ber.png")
    print("- dhscc_ber_comparison.png")

if __name__ == "__main__":
    main()