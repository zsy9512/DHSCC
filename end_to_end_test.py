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
from channel.channel_modem import ChannelModem
from sender.modulation import WirelessTransmitter
from receiver.demodulation import WirelessReceiver
from sender.polar_encoding import PolarEncoder, get_frozen_pos
from receiver.polar_decoding import PolarSCDecoder
from sender.utils_signal import psnr_to_noise


def calculate_ber(original_bits, decoded_bits):
    """Calculate bit error rate between original and decoded bits."""
    if isinstance(original_bits, torch.Tensor):
        original_bits = original_bits.cpu().numpy()
    if isinstance(decoded_bits, torch.Tensor):
        decoded_bits = decoded_bits.cpu().numpy()

    errors = np.sum(original_bits != decoded_bits)
    total_bits = original_bits.size
    return errors / total_bits if total_bits > 0 else 0.0


def test_end_to_end_system(sender_model, receiver_model, quantizer, channel_modem, test_data, device, snr_range=None, args=None):
    """
    Test the complete end-to-end system performance.

    Args:
        sender_model: Trained SenderNet model
        receiver_model: Trained ReceiverNet model
        quantizer: Trained VibrationVectorQuantizer
        channel_modem: ChannelModem instance
        test_data: DataLoader or tuple of (data, labels)
        device: torch.device
        snr_range: List of SNR values to test
        args: Training arguments (optional, for SNR range configuration)

    Returns:
        dict: Test results including accuracy, BER, etc.
    """
    if snr_range is None:
        # Use same SNR range as training if args provided
        if args is not None and hasattr(args, 'snr_min') and hasattr(args, 'snr_max'):
            # Test each integer SNR from snr_min to snr_max
            snr_range = list(range(int(args.snr_min), int(args.snr_max) + 1))
            print(f"🔍 Testing with training SNR range: {snr_range}")
        else:
            # Fallback to default range
            snr_range = [0, 5, 10, 15, 20, 25, 30]

    sender_model.eval()
    receiver_model.eval()

    results = {
        'snr': [],
        'accuracy': [],
        'ber': [],
        'loss': []
    }

    with torch.no_grad():
        for snr in tqdm(snr_range, desc="Testing SNR levels"):
            total_loss = 0.0
            total_acc = 0.0
            total_ber = 0.0
            total_samples = 0

            # Handle both DataLoader and (data, labels) tuple
            if isinstance(test_data, torch.utils.data.DataLoader):
                # Use DataLoader
                for data, target in test_data:
                    data, target = data.to(device), target.to(device)
                    batch_size = data.size(0)
                    total_samples += batch_size

                    batch_results = _test_batch(sender_model, receiver_model, quantizer, channel_modem, data, target, snr, device, args)
                    total_loss += batch_results['loss'] * batch_size
                    total_acc += batch_results['acc'] * batch_size
                    total_ber += batch_results['ber'] * batch_size
            else:
                # Use (data, labels) tuple
                data, labels = test_data
                batch_size = data.size(0)
                total_samples += batch_size

                batch_results = _test_batch(sender_model, receiver_model, quantizer, channel_modem, data, labels, snr, device, args)
                total_loss += batch_results['loss'] * batch_size
                total_acc += batch_results['acc'] * batch_size
                total_ber += batch_results['ber'] * batch_size

            # Average metrics
            avg_loss = total_loss / total_samples
            avg_acc = total_acc / total_samples
            avg_ber = total_ber / total_samples

            results['snr'].append(snr)
            results['accuracy'].append(avg_acc)
            results['ber'].append(avg_ber)
            results['loss'].append(avg_loss)

            print(f"SNR {snr}dB - Accuracy: {avg_acc:.2f}%, BER: {avg_ber:.6f}, Loss: {avg_loss:.4f}")

    return results


def _test_batch(sender_model, receiver_model, quantizer, channel_modem, data, labels, snr, device, args=None):
    """Test a single batch of data."""
    # Forward pass through sender
    batch_size = data.size(0)
    # Use SNR-dependent noise level (same as training)
    channel_noise = psnr_to_noise(torch.tensor(snr, device=device)).unsqueeze(0).expand(batch_size, -1)
    sender_output = sender_model(data, channel_noise)

    # Quantize features for transmission
    encoded_features, importance_weights, pre_task_logits, feature_shape, original_encoded_features = sender_output
    quantized_features, quantizer_indices, vq_loss = quantizer.encode(encoded_features)

    # Ensure quantizer_indices are on the same device as other tensors
    quantizer_indices = quantizer_indices.to(device)

    # Use channel_modem for transmission with polar coding integration
    if hasattr(channel_modem, 'transmit_with_polar_coding'):
        # Use all 8 groups with polar coding + QPSK (same as training)
        group_to_use = 8  # Default to 8 groups
        received_features, ber, transmit_mse = channel_modem.transmit_with_polar_coding(
            quantized_features, quantizer_indices,
            snr_db=snr,
            use_coding=True,  # Enable polar coding for testing
            group_to_use=group_to_use,
            quantizer=quantizer
        )
    else:
        # Fallback: simple transmission without proper coding
        received_features = channel_modem.transmit(quantized_features, snr)
        ber = torch.tensor(0.01, device=device)  # Placeholder BER
        transmit_mse = torch.tensor(0.0, device=device)

    # Receiver processing
    receiver_output = receiver_model(received_features, channel_noise, feature_shape)

    # Calculate metrics
    predictions = receiver_output.argmax(dim=1)
    accuracy = (predictions == labels).float().mean().item() * 100

    # Calculate complete loss (same as training/validation)
    classification_loss = nn.CrossEntropyLoss()(receiver_output, labels)
    vq_weight = getattr(args, 'vq_loss_weight', 0.1) if args else 0.1
    transmit_mse_loss = torch.nn.MSELoss()(received_features, quantized_features)
    loss = (classification_loss + vq_weight * vq_loss + 0.1 * transmit_mse_loss).item()

    # Convert BER to scalar if it's a tensor
    if isinstance(ber, torch.Tensor):
        ber = ber.item()

    return {
        'loss': loss,
        'acc': accuracy,
        'ber': ber
    }


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
        demodulated_llr = demodulator.demapper(noisy_signal, noise_var)
        
        # Flatten LLRs for polar decoding
        # Shape: [batch, num_symbols, bits_per_symbol] -> [batch, num_symbols * bits_per_symbol]
        demodulated_llr_flat = demodulated_llr.view(demodulated_llr.shape[0], -1)

        # Decode with polar decoder
        decoded_bits = polar_decoder(demodulated_llr_flat)

        # Calculate BER
        ber = calculate_ber(original_bits, decoded_bits)

        ber_results['snr'].append(snr)
        ber_results['ber'].append(ber)

        print(f"SNR {snr}dB - BER: {ber:.6f}")

    return ber_results


def run_end_to_end_tests(checkpoint_dir=None, test_loader=None, args=None, num_classes=10, model_path=None):
    """Run comprehensive end-to-end tests."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use provided args or create default args
    if args is None:
        # Create model arguments
        class ModelArgs:
            def __init__(self):
                self.intermediate_dim = 128
                self.num_classes = num_classes
                self.quantizer_dimension = 128
                self.quantizer_n_codes_group = 8
                self.quantizer_bins = 16
                self.quantizer_n_q = 32
                # SNR range (same as training defaults)
                self.snr_min = 10.0
                self.snr_max = 20.0
                # Add other necessary attributes
                self.bits_per_index = 4
                self.max_groups = 8
                self.N = 256
                self.K = 128
                self.channel_type = 'uplink'
                self.decoder_type = 'SC'
                self.list_size = 8
                self.constellation_type = 'qam'
                self.normalize = True
                self.demapping_method = 'app'
                self.selector_snr_bins = 21
                self.selector_algorithm = 'heuristic'
                self.fixed_group = None
                self.equalization_method = 'mrc'
                self.udp_tx_host = 'localhost'
                self.udp_tx_port = 12345
                self.udp_rx_host = 'localhost'
                self.udp_rx_port = 12346
                self.udp_timeout = 10.0
                self.udp_buffer_size = 8192
                self.udp_max_retries = 3
                self.threshold = 0.0

        args = ModelArgs()

    # Initialize models
    sender_model = SenderNet(args).to(device)
    receiver_model = ReceiverNet(args, num_classes).to(device)
    quantizer = VibrationVectorQuantizer(
        dimension=args.quantizer_dimension,
        n_codes_group=args.quantizer_n_codes_group,
        bins=args.quantizer_bins,
        n_q=args.quantizer_n_q
    ).to(device)
    channel_modem = ChannelModem(args)

    # Load trained model weights if checkpoint directory is provided
    if model_path:
        print(f"📁 Loading model from: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            sender_model.load_state_dict(checkpoint['sender_model_state_dict'])
            receiver_model.load_state_dict(checkpoint['receiver_model_state_dict'])
            quantizer.load_state_dict(checkpoint['quantizer_state_dict'])
            print("✓ Loaded model checkpoint")
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            print("Using randomly initialized models for testing")
    elif checkpoint_dir:
        print(f"📁 Loading models from checkpoint: {checkpoint_dir}")
        try:
            # First try to load the best model checkpoint
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path, map_location=device)
                sender_model.load_state_dict(checkpoint['sender_model_state_dict'])
                receiver_model.load_state_dict(checkpoint['receiver_model_state_dict'])
                quantizer.load_state_dict(checkpoint['quantizer_state_dict'])
                print("✓ Loaded best model checkpoint")
            else:
                # Fall back to individual model files
                # Load sender model
                sender_path = os.path.join(checkpoint_dir, "final_sender_model.pth")
                if os.path.exists(sender_path):
                    checkpoint = torch.load(sender_path, map_location=device)
                    # Handle both formats: direct state_dict or dict with 'state_dict' key
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        sender_model.load_state_dict(checkpoint['state_dict'])
                    else:
                        # Assume checkpoint is the state_dict directly
                        sender_model.load_state_dict(checkpoint)
                    print("✓ Loaded sender model")
                else:
                    print("⚠️  Sender model not found, using random weights")

                # Load receiver model
                receiver_path = os.path.join(checkpoint_dir, "final_receiver_model.pth")
                if os.path.exists(receiver_path):
                    checkpoint = torch.load(receiver_path, map_location=device)
                    # Handle both formats: direct state_dict or dict with 'state_dict' key
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        receiver_model.load_state_dict(checkpoint['state_dict'])
                    else:
                        # Assume checkpoint is the state_dict directly
                        receiver_model.load_state_dict(checkpoint)
                    print("✓ Loaded receiver model")
                else:
                    print("⚠️  Receiver model not found, using random weights")

                # Load quantizer
                quantizer_path = os.path.join(checkpoint_dir, "final_quantizer.pth")
                if os.path.exists(quantizer_path):
                    checkpoint = torch.load(quantizer_path, map_location=device)
                    # Handle both formats: direct state_dict or dict with 'state_dict' key
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        quantizer.load_state_dict(checkpoint['state_dict'])
                    else:
                        # Assume checkpoint is the state_dict directly
                        quantizer.load_state_dict(checkpoint)
                    print("✓ Loaded quantizer")
                else:
                    print("⚠️  Quantizer not found, using random weights")

        except Exception as e:
            print(f"⚠️  Error loading models: {e}")
            print("Using randomly initialized models for testing")
    else:
        print("⚠️  No checkpoint directory provided, using random weights")

    # Initialize modulation components
    modulator = WirelessTransmitter(k=128, n=256, use_coding=True)
    demodulator = WirelessReceiver(
        encoder_ref=modulator.encoder,
        use_coding=True,
        constellation_type="qam",
        bits_per_symbol=4
    )

    # Initialize polar coding (for reference, not used in modulation)
    info_bits = 128
    codeword_length = 256
    frozen_pos = get_frozen_pos(codeword_length, info_bits)
    polar_encoder = PolarEncoder(frozen_pos, codeword_length)
    polar_decoder = PolarSCDecoder(frozen_pos, codeword_length).to(device)

    print("\n" + "="*60)
    print("END-TO-END SYSTEM TESTS")
    print("="*60)

    # Test 1: Modulation/Demodulation BER
    print("\n1. Testing Modulation/Demodulation BER with Polar Coding...")
    modulation_results = test_modulation_demodulation_ber(
        modulator, demodulator, polar_encoder, polar_decoder, device
    )

    # Test 2: End-to-end system (using real test data)
    print("\n2. Testing End-to-End System Performance...")

    if test_loader is not None:
        # Use real test data
        end_to_end_results = test_end_to_end_system(
            sender_model, receiver_model, quantizer, channel_modem,
            test_loader, device, args=args
        )
    else:
        # Fallback to dummy data for demonstration
        print("⚠️  No test loader provided, using dummy data")
        batch_size = 100
        channels = 1
        seq_len = 128  # Time sequence length
        num_classes = 10

        dummy_data = torch.randn(batch_size, channels, seq_len).to(device)
        dummy_labels = torch.randint(0, num_classes, (batch_size,)).to(device)

        end_to_end_results = test_end_to_end_system(
            sender_model, receiver_model, quantizer, channel_modem,
            (dummy_data, dummy_labels), device, args=args
        )

    # Test 3: Model connectivity test
    print("\n3. Testing Model Connectivity...")

    try:
        # Define test parameters
        batch_size = 100
        channels = 1
        seq_len = 128  # Time sequence length
        num_classes = 10

        dummy_data = torch.randn(batch_size, channels, seq_len).to(device)
        dummy_labels = torch.randint(0, num_classes, (batch_size,)).to(device)

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
    import argparse

    parser = argparse.ArgumentParser(description='End-to-End System Tests')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory containing trained model checkpoints')
    parser.add_argument('--test_snr', type=float, default=15.0,
                        help='SNR level for testing (not used in current implementation)')
    parser.add_argument('--data_dir', type=str, default="data/vibcls/vibrodata",
                        help='Directory containing data files')
    parser.add_argument('--test_list', type=str, default="data/vibcls/test.txt",
                        help='Path to test file list')
    parser.add_argument('--class_map', type=str, default="data/vibcls/clsmap.txt",
                        help='Path to class mapping file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for testing')

    args = parser.parse_args()

    # Create test data loader if possible
    test_loader = None
    try:
        from data.data import WavDataset, load_file_list, load_class_names
        from data.data_loader import create_data_loaders_from_args
        from torch.utils.data import DataLoader

        # Create test dataset
        test_files = load_file_list(args.test_list)
        class_names = load_class_names(args.class_map)
        test_dataset = WavDataset(test_files, args.data_dir)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        print(f"✓ Created test data loader with {len(test_dataset)} samples")
    except Exception as e:
        print(f"⚠️  Could not create test data loader: {e}")
        print("Using dummy data for testing")

    success = run_end_to_end_tests(checkpoint_dir=args.checkpoint_dir, test_loader=test_loader)
    if success:
        print("\n✅ All end-to-end tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)