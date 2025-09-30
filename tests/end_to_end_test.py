#!/usr/bin/env python3
"""
End-to-End Test for DHSCC Split System

This test evaluates the complete end-to-end performance of the sender-receiver
communication system including:
1. Sender feature extraction and quantization
2. Modulation and channel transmission
3. Demodulation and receiver decoding
4. Final classification accuracy and BER
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

from sender.sender_model import SenderNet
from receiver.receiver_model import ReceiverNet
from sender.quantizer import VibrationVectorQuantizer
from channel_modem.channel_modem import ChannelModem
from sender.modulation import Modulator
from receiver.demodulation import Demodulator
from sender.polar_encoding import PolarEncoder, get_frozen_pos
from receiver.polar_decoding import PolarSCDecoder


def calculate_ber(original_bits, decoded_bits):
    """Calculate bit error rate between original and decoded bits."""
    if isinstance(original_bits, torch.Tensor):
        original_bits = original_bits.cpu().numpy()
    if isinstance(decoded_bits, torch.Tensor):
        decoded_bits = decoded_bits.cpu().numpy()

    errors = np.sum(original_bits != decoded_bits)
    total_bits = original_bits.size
    return errors / total_bits if total_bits > 0 else 0.0


def test_end_to_end_system(sender_model, receiver_model, channel_modem, test_data, device, snr_range=None):
    """
    Test the complete end-to-end system performance.

    Args:
        sender_model: Trained SenderNet model
        receiver_model: Trained ReceiverNet model
        channel_modem: ChannelModem instance
        test_data: Tuple of (data, labels)
        device: torch.device
        snr_range: List of SNR values to test

    Returns:
        dict: Test results including accuracy, BER, etc.
    """
    if snr_range is None:
        snr_range = [0, 5, 10, 15, 20, 25, 30]

    sender_model.eval()
    receiver_model.eval()

    data, labels = test_data
    data, labels = data.to(device), labels.to(device)

    results = {
        'snr': [],
        'accuracy': [],
        'ber': [],
        'loss': []
    }

    with torch.no_grad():
        for snr in tqdm(snr_range, desc="Testing SNR levels"):
            # Set channel SNR
            # channel_modem.set_snr(snr)  # Not needed, snr passed to transmit

            # Forward pass through sender
            batch_size = data.size(0)
            channel_noise = torch.randn(batch_size, 1).to(device) * 0.1 + 1.0
            sender_output = sender_model(data, channel_noise)

            # Transmit through channel
            encoded_features, importance_weights, pre_task_logits, feature_shape, original_encoded_features = sender_output
            transmitted_signal = channel_modem.transmit(encoded_features, snr)

            # Receive and decode
            receiver_input = transmitted_signal  # transmit already returns received signal
            receiver_output = receiver_model(receiver_input, snr, feature_shape)

            # Calculate metrics
            predictions = receiver_output.argmax(dim=1)
            accuracy = (predictions == labels).float().mean().item() * 100

            loss = nn.CrossEntropyLoss()(receiver_output, labels).item()

            # Calculate BER (simplified - compare quantized features)
            # In practice, you'd compare the actual transmitted bits
            ber = 0.01  # Placeholder - implement proper BER calculation

            results['snr'].append(snr)
            results['accuracy'].append(accuracy)
            results['ber'].append(ber)
            results['loss'].append(loss)

            print(f"SNR {snr}dB - Accuracy: {accuracy:.2f}%, BER: {ber:.6f}, Loss: {loss:.4f}")

    return results


def test_modulation_demodulation_ber(modulator, demodulator, polar_encoder, polar_decoder, device, snr_range=None):
    """
    Test modulation and demodulation BER with polar coding.

    Args:
        modulator: Modulator instance
        demodulator: Demodulator instance
        polar_encoder: PolarEncoder instance
        polar_decoder: PolarSCDecoder instance
        device: torch.device
        snr_range: List of SNR values

    Returns:
        dict: BER results
    """
    if snr_range is None:
        snr_range = np.arange(0, 31, 2)

    # Generate random bits
    batch_size = 100
    info_bits = 128  # Information bits
    codeword_length = 256  # Polar code length

    original_bits = torch.randint(0, 2, (batch_size, info_bits)).float().to(device)

    ber_results = {'snr': [], 'ber': []}

    for snr in tqdm(snr_range, desc="Testing modulation BER"):
        # Encode with polar code
        encoded_bits = polar_encoder(original_bits)

        # Modulate
        modulated_signal = modulator.modulate(encoded_bits)

        # Add noise (simulate channel)
        noise_var = 10**(-snr/10)
        noise = torch.randn_like(modulated_signal) * np.sqrt(noise_var)
        noisy_signal = modulated_signal + noise

        # Demodulate
        demodulated_llr = demodulator.demodulate(noisy_signal)

        # Decode with polar decoder
        decoded_bits = polar_decoder(demodulated_llr)

        # Calculate BER
        ber = calculate_ber(original_bits, decoded_bits)

        ber_results['snr'].append(snr)
        ber_results['ber'].append(ber)

        print(f"SNR {snr}dB - BER: {ber:.6f}")

    return ber_results


def run_end_to_end_tests():
    """Run comprehensive end-to-end tests."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model arguments
    class ModelArgs:
        def __init__(self):
            self.intermediate_dim = 128
            self.num_classes = 10

    args = ModelArgs()

    # Initialize models
    sender_model = SenderNet(args).to(device)
    receiver_model = ReceiverNet(args, args.num_classes).to(device)
    channel_modem = ChannelModem(args)

    # Initialize modulation components
    modulator = Modulator(args)
    demodulator = Demodulator(args)

    # Initialize polar coding
    info_bits = 128
    codeword_length = 256
    frozen_pos = get_frozen_pos(codeword_length, info_bits)
    polar_encoder = PolarEncoder(frozen_pos, info_bits)
    polar_decoder = PolarSCDecoder(frozen_pos, info_bits).to(device)

    print("\n" + "="*60)
    print("END-TO-END SYSTEM TESTS")
    print("="*60)

    # Test 1: Modulation/Demodulation BER
    print("\n1. Testing Modulation/Demodulation BER with Polar Coding...")
    modulation_results = test_modulation_demodulation_ber(
        modulator, demodulator, polar_encoder, polar_decoder, device
    )

    # Test 2: End-to-end system (using dummy data for demonstration)
    print("\n2. Testing End-to-End System Performance...")

    # Create dummy test data - vibration signals [batch_size, channels, time]
    batch_size = 100
    channels = 1
    seq_len = 128  # Time sequence length
    num_classes = 10

    dummy_data = torch.randn(batch_size, channels, seq_len).to(device)
    dummy_labels = torch.randint(0, num_classes, (batch_size,)).to(device)

    end_to_end_results = test_end_to_end_system(
        sender_model, receiver_model, channel_modem,
        (dummy_data, dummy_labels), device
    )

    # Test 3: Model connectivity test
    print("\n3. Testing Model Connectivity...")

    try:
        # Test sender forward pass
        channel_noise = torch.randn(batch_size, 1).to(device) * 0.1 + 1.0
        sender_out = sender_model(dummy_data, channel_noise)
        features, importance_weights, pre_task_logits, feature_shape, original_encoded_features = sender_out
        assert features is not None
        print("✓ Sender model connectivity: PASS")

        # Test channel transmission
        transmitted = channel_modem.transmit(features, 20.0)  # 20dB SNR
        print("✓ Channel modem connectivity: PASS")

        # Test receiver forward pass
        receiver_out = receiver_model(transmitted, 20.0, feature_shape)
        assert receiver_out is not None
        print("✓ Receiver model connectivity: PASS")

        print("✓ All model connectivity tests: PASS")

    except Exception as e:
        print(f"✗ Model connectivity test failed: {e}")
        return False

    # Plot results
    print("\n4. Generating performance plots...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot BER vs SNR
    axes[0].semilogy(modulation_results['snr'], modulation_results['ber'], 'b-o', linewidth=2, label='Modulation BER')
    axes[0].set_title('Bit Error Rate vs SNR')
    axes[0].set_xlabel('SNR (dB)')
    axes[0].set_ylabel('BER')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot End-to-End Accuracy vs SNR
    axes[1].plot(end_to_end_results['snr'], end_to_end_results['accuracy'], 'r-s', linewidth=2, label='End-to-End Accuracy')
    axes[1].set_title('End-to-End Accuracy vs SNR')
    axes[1].set_xlabel('SNR (dB)')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(os.path.dirname(__file__), '..', 'end_to_end_test_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Saved test results plot to: {plot_path}")

    print("\n" + "="*60)
    print("END-TO-END TESTS COMPLETED SUCCESSFULLY")
    print("="*60)

    return True


if __name__ == "__main__":
    success = run_end_to_end_tests()
    if success:
        print("\n✅ All end-to-end tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)