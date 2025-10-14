#!/usr/bin/env python3
"""
Channel Process for Separated End-to-End Deployment

This module implements the channel process that:
1. Receives binary bits from sender
2. Performs polar encoding
3. Modulates and transmits through AWGN channel
4. Demodulates and performs polar decoding
5. Reconstructs features using quantizer
6. Sends reconstructed features to receiver
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import argparse
from multiprocessing import Queue
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from channel.polar_encoding import PolarEncoder
from channel.polar_decoding import PolarSCDecoder
from channel.system_integration import WirelessCommunicationSystem
from sender.quantizer import VibrationVectorQuantizer


def load_quantizer_for_channel(args, device):
    """Load trained quantizer for feature reconstruction."""
    # print("🔄 Loading quantizer for channel...")

    # Create quantizer
    quantizer = VibrationVectorQuantizer(
        dimension=args.intermediate_dim,
        n_codes_group=args.quantizer_n_codes_group,
        bins=args.quantizer_bins,
        n_q=args.quantizer_n_q
    ).to(device)

    # Load quantizer weights
    quantizer_path = getattr(args, 'quantizer_path', None)
    if not quantizer_path:
        # Try to find the final_quantizer.pth
        checkpoint_dir = getattr(args, 'checkpoint_dir', './checkpoints')
        quantizer_path = os.path.join(checkpoint_dir, 'direct_polar_training_*', 'final_quantizer.pth')
        import glob
        matching_files = glob.glob(quantizer_path)
        if matching_files:
            quantizer_path = max(matching_files, key=os.path.getctime)
            # print(f"📂 Found quantizer: {quantizer_path}")

    if quantizer_path and os.path.exists(quantizer_path):
        quantizer_checkpoint = torch.load(quantizer_path, map_location=device)
        # Handle different checkpoint formats
        if 'state_dict' in quantizer_checkpoint:
            quantizer.load_state_dict(quantizer_checkpoint['state_dict'])
        else:
            quantizer.load_state_dict(quantizer_checkpoint)
        # print(f"✅ Loaded quantizer from {quantizer_path}")
    else:
        # print("⚠️  Quantizer weights not found, using random reconstruction")
        pass

    quantizer.eval()
    return quantizer


def binary_to_indices(binary, original_shape, bits_per_index=4):
    """Convert binary back to quantizer indices."""
    device = binary.device
    batch_size, total_bits = binary.shape
    bits_per_index = bits_per_index
    indices_per_sample = total_bits // bits_per_index

    indices = []
    for b in range(batch_size):
        sample_bits = binary[b]
        sample_indices = []
        for i in range(0, total_bits, bits_per_index):
            bit_chunk = sample_bits[i:i+bits_per_index]
            # Convert bits to integer
            bit_chunk_int = [int(bit.item()) if hasattr(bit, 'item') else int(bit) for bit in bit_chunk]
            idx = sum(bit_val * (2**(bits_per_index-1-j)) for j, bit_val in enumerate(bit_chunk_int))
            sample_indices.append(idx)
        indices.append(sample_indices)

    indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)

    # Reshape to expected format [batch_size, n_groups, group_size]
    # original_shape should be (batch_size, n_groups, group_size)
    if len(original_shape) == 3:
        n_groups = original_shape[1]
        group_size = original_shape[2]
        indices_tensor = indices_tensor.view(batch_size, n_groups, group_size)
    else:
        # Fallback: assume it's [batch_size, total_indices]
        indices_tensor = indices_tensor.view(batch_size, -1)

    return indices_tensor


def channel_process(args, from_sender_queue, to_receiver_queue):
    """Main channel process function with polar coding and feature reconstruction."""
    # print("\n" + "="*50)
    # print("CHANNEL PROCESS STARTED")
    # print("="*50)

    # Setup device (try CUDA for channel operations to match direct test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"📡 Channel using device: {device}")

    # Initialize channel modem for transmission
    # print("🔧 Initializing channel modem...")
    from channel.channel_modem import ChannelModem
    # Ensure ChannelModem uses the same device as our process
    import channel.channel_modem
    channel.channel_modem.device = device  # Override the module-level device
    channel_modem = ChannelModem(args)

    # Initialize modulation and demodulation components
    # print("🔧 Initializing modulation/demodulation...")
    from channel.transmitter import WirelessTransmitter
    from channel.receiver import WirelessReceiver
    from channel.channel import WirelessChannel

    # Create transmitter for modulation only (no coding since we handle it separately)
    transmitter = WirelessTransmitter(
        use_coding=False,  # We handle polar coding separately
        constellation_type="qam",  # QAM modulation (QPSK is 4-QAM with 2 bits/symbol)
        bits_per_symbol=2,  # 2 bits per symbol for QPSK
        device=device
    )

    # Create receiver for demodulation only (no decoding since we handle it separately)
    receiver = WirelessReceiver(
        use_coding=False,  # We handle polar decoding separately
        constellation_type="qam",
        bits_per_symbol=2,
        demapping_method="app",
        device=device
    )

    # Create channel for AWGN simulation
    wireless_channel = WirelessChannel(
        channel_model=args.channel_model,
        device=device
    )

    # Load quantizer for feature reconstruction
    quantizer = load_quantizer_for_channel(args, device)

    # print("📡 Channel begin processing data")

    with torch.no_grad():
        while True:
            # Receive data from sender
            data_packet = from_sender_queue.get()

            if data_packet.get('type') == 'end':
                # print("📡 Received end signal from sender")
                # Forward end signal to receiver
                to_receiver_queue.put({'type': 'end'})
                break

            batch_idx = data_packet['batch_idx']
            batch_size = data_packet['batch_size']
            snr_db = data_packet.get('snr_db', args.channel_snr)

            # Extract data from packet
            quantized_features = torch.tensor(data_packet['quantized_features'], dtype=torch.float, device=device)
            codeword = torch.tensor(data_packet['codeword'], dtype=torch.float, device=device)
            target = torch.tensor(data_packet['target'], dtype=torch.long, device=device)
            feature_shape = data_packet.get('feature_shape', None)  # Extract feature_shape
            selected_indices_shape = data_packet.get('selected_indices_shape', None)  # Extract shape for reconstruction
            # Ensure quantizer is on the same device
            quantizer = quantizer.to(device)
            # Also ensure channel_modem components are on correct device
            if hasattr(channel_modem, 'wireless_interface') and channel_modem.wireless_interface is not None:
                # Move wireless_interface to correct device if possible
                pass  # WirelessInterface handles device internally
            codeword = codeword.reshape(1, -1).to(device)  # Ensure correct shape and device
            # Use channel_modem for transmission (codeword already encoded)
            llrs, ber, transmit_mse = channel_modem.transmit_encoded_codeword(
                quantized_features, codeword,
                snr_db=snr_db,
                quantizer=quantizer,
                selected_indices_shape=selected_indices_shape
            )
            
            # Prepare data packet for receiver
            receiver_packet = {
                'batch_idx': batch_idx,
                'batch_size': batch_size,
                'snr_db': snr_db,
                'llrs': llrs.cpu().numpy(),
                'target': target.cpu().numpy(),
                'feature_shape': feature_shape,  # Pass feature_shape to receiver
                'selected_indices_shape': selected_indices_shape  # Pass shape for binary_to_indices
            }

            # Send to receiver
            to_receiver_queue.put(receiver_packet)

    # print("📡 Channel process completed")


def create_channel_args():
    """Create argument parser for channel process."""
    parser = argparse.ArgumentParser(description='Channel Process')

    # Channel parameters
    parser.add_argument('--channel_model', type=str, default='awgn',
                        choices=['awgn', 'rayleigh', 'udp'],
                        help='Channel type for simulation')
    parser.add_argument('--channel_snr', type=float, default=10.0,
                        help='Signal-to-noise ratio for channel simulation')

    # Polar coding parameters
    parser.add_argument('--N', type=int, default=256, help='Polar code length')
    parser.add_argument('--K', type=int, default=128, help='Information bits')
    parser.add_argument('--decoder_type', type=str, default='SC',
                        choices=['SC', 'SCL', 'BP'],
                        help='Polar decoder type')

    # Quantizer parameters
    parser.add_argument('--intermediate_dim', type=int, default=128)
    parser.add_argument('--quantizer_n_codes_group', type=int, default=8)
    parser.add_argument('--quantizer_bins', type=int, default=16)
    parser.add_argument('--quantizer_n_q', type=int, default=32)  # Match training default

    # Model paths
    parser.add_argument('--quantizer_path', type=str, help='Path to quantizer weights')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')

    return parser


if __name__ == "__main__":
    # This is for standalone testing
    parser = create_channel_args()
    args = parser.parse_args()

    # Create dummy queues for testing
    from_sender = Queue()
    to_receiver = Queue()

    channel_process(args, from_sender, to_receiver)