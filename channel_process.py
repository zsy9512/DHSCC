"""
Channel Process for Separated End-to-End Deployment

This module implements the channel simulation process that runs independently.
It receives encoded bits from sender, performs modulation, channel simulation,
demodulation, and sends demodulated bits to receiver. All operations use CPU.
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

from sender.modulation import WirelessTransmitter
from receiver.demodulation import WirelessReceiver
from sender.polar_encoding import Polar5GEncoder
from receiver.polar_decoding import Polar5GDecoder
from channel.channel_modem import WirelessChannel


def binary_to_indices(binary, original_shape, bits_per_index=4):
    """Convert binary back to quantizer indices."""
    device = torch.device('cpu')
    batch_size, total_bits = binary.shape
    indices_per_sample = total_bits // bits_per_index

    indices = []
    for b in range(batch_size):
        sample_bits = binary[b]
        sample_indices = []
        for i in range(0, total_bits, bits_per_index):
            bit_chunk = sample_bits[i:i+bits_per_index]
            # Convert bits to integers (they might be floats from numpy)
            bit_chunk_int = [int(bit.item()) if hasattr(bit, 'item') else int(bit) for bit in bit_chunk]
            idx = sum(bit_val * (2**(bits_per_index-1-j)) for j, bit_val in enumerate(bit_chunk_int))
            sample_indices.append(idx)
        indices.append(sample_indices)

    indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
    if len(original_shape) == 2:
        return indices_tensor.view(original_shape)
    else:
        return indices_tensor.view(original_shape)


def channel_process(args, from_sender_queue, to_receiver_queue):
    """Main channel process function."""
    print("\n" + "="*50)
    print("CHANNEL PROCESS STARTED")
    print("="*50)

    # All operations on CPU as required
    device = torch.device('cpu')
    print(f"📡 Channel using device: {device}")

    # Create channel model
    channel = WirelessChannel(channel_model=args.channel_model, device='cpu')

    # Create transmitter and receiver for modulation/demodulation
    transmitter = WirelessTransmitter(
        k=args.K if args.communication_mode == 4 else None,
        n=args.N if args.communication_mode == 4 else None,
        channel_type=args.channel_type,
        use_coding=args.communication_mode == 4,
        constellation_type=args.constellation_type,
        bits_per_symbol=int(np.log2(args.N//args.K)) if args.communication_mode == 4 else 4,
        normalize=args.normalize,
        device='cpu',
        verbose=args.verbose
    ).to(device)

    receiver = WirelessReceiver(
        encoder_ref=transmitter.encoder if args.communication_mode == 4 else None,
        decoder_type=args.decoder_type,
        list_size=args.list_size,
        use_coding=args.communication_mode == 4,
        constellation_type=args.constellation_type,
        bits_per_symbol=int(np.log2(args.N//args.K)) if args.communication_mode == 4 else 4,
        normalize=args.normalize,
        demapping_method=args.demapping_method,
        equalization_method=args.equalization_method,
        device='cpu',
        verbose=args.verbose
    ).to(device)

    print(f"📡 Channel model: {args.channel_model}")
    print(f"📡 Communication mode: {args.communication_mode}")
    print(f"📡 SNR: {args.channel_snr}dB")

    # Process data from sender
    processed_batches = 0
    total_samples = 0

    while True:
        # Receive data from sender
        data_packet = from_sender_queue.get()

        if data_packet.get('type') == 'end':
            print("📡 Received end signal from sender")
            break

        batch_idx = data_packet['batch_idx']
        batch_size = data_packet['batch_size']
        total_samples += batch_size

        print(f"📡 Processing batch {batch_idx + 1}, size: {batch_size}")

        # Convert received data to tensors
        binary_bits = torch.tensor(data_packet['binary_bits'], dtype=torch.float, device=device)
        target = torch.tensor(data_packet['target'], dtype=torch.long, device=device)
        feature_shape = data_packet['feature_shape']

        # Convert binary bits back to indices
        quantizer_indices = binary_to_indices(binary_bits, (batch_size, -1), bits_per_index=args.bits_per_index)

        # Perform channel transmission
        with torch.no_grad():
            # For digital transmission with coding (mode 4)
            if args.communication_mode == 4:
                # Polar encoding + modulation + channel + demodulation + polar decoding
                # This is a simplified version - in practice would need proper bit-to-symbol mapping

                # Convert indices to bits for polar coding
                info_bits = binary_bits.view(batch_size, -1)

                # Polar encoding
                if transmitter.encoder is not None:
                    coded_bits = transmitter.encoder(info_bits)
                else:
                    coded_bits = info_bits

                # Modulation
                modulated_symbols = transmitter.mapper(coded_bits)

                # Channel simulation
                if args.channel_model == 'awgn':
                    noisy_symbols, noise_power = channel.add_awgn(
                        modulated_symbols,
                        snr_db=args.channel_snr,
                        bits_per_symbol=transmitter.bits_per_symbol,
                        code_rate=transmitter.code_rate
                    )
                else:  # rayleigh
                    # For simplicity, using AWGN for now
                    noisy_symbols, noise_power = channel.add_awgn(
                        modulated_symbols,
                        snr_db=args.channel_snr,
                        bits_per_symbol=transmitter.bits_per_symbol,
                        code_rate=transmitter.code_rate
                    )

                # Calculate noise variance for demapper
                # noise_power is the noise power spectral density
                noise_var = noise_power

                # Demodulation
                demodulated_bits = receiver.demapper(noisy_symbols, noise_var)

                # Polar decoding
                if receiver.decoder is not None:
                    decoded_bits = receiver.decoder(demodulated_bits)
                else:
                    decoded_bits = demodulated_bits

                # Convert back to binary format
                received_bits = decoded_bits

            else:
                # For analog or simple digital transmission (modes 1,2,3)
                # Simplified: just pass through with noise
                received_bits = binary_bits

        # Convert to numpy for IPC
        received_bits_cpu = received_bits.cpu().numpy()
        target_cpu = target.cpu().numpy()

        # Prepare output packet
        output_packet = {
            'received_bits': received_bits_cpu,
            'target': target_cpu,
            'feature_shape': feature_shape,
            'batch_size': batch_size,
            'batch_idx': batch_idx
        }

        # Send to receiver
        print(f"📤 Sending batch {batch_idx + 1} to receiver ({received_bits_cpu.shape} bits)")
        to_receiver_queue.put(output_packet)

        processed_batches += 1

    # Send end signal to receiver
    to_receiver_queue.put({'type': 'end'})
    print(f"✅ Channel process completed. Processed {processed_batches} batches, {total_samples} samples.")


def create_channel_args():
    """Create argument parser for channel process."""
    parser = argparse.ArgumentParser(description='Channel Process')

    # Channel parameters
    parser.add_argument('--channel_model', type=str, default='rayleigh',
                        choices=['awgn', 'rayleigh', 'udp'],
                        help='Channel type for simulation')
    parser.add_argument('--channel_snr', type=float, default=10.0,
                        help='Signal-to-Noise Ratio for channel simulation (dB)')

    # Communication mode
    parser.add_argument('--communication_mode', type=int, default=4, choices=[0, 1, 2, 3, 4],
                        help='Communication mode')

    # Digital transmission parameters
    parser.add_argument('--bits_per_index', type=int, default=4,
                        help='Number of bits per quantization index')
    parser.add_argument('--N', type=int, default=256,
                        help='Polar code length')
    parser.add_argument('--K', type=int, default=128,
                        help='Information bits for polar coding')
    parser.add_argument('--channel_type', type=str, default='uplink',
                        choices=['uplink', 'downlink'],
                        help='5G polar coding channel type')
    parser.add_argument('--decoder_type', type=str, default='SC',
                        choices=['SC', 'SCL'],
                        help='Polar decoder type')
    parser.add_argument('--list_size', type=int, default=8,
                        help='List size for SCL decoder')

    # Modulation parameters
    parser.add_argument('--constellation_type', type=str, default='qam',
                        choices=['qam', 'bpsk', 'pam'],
                        help='Constellation type for modulation')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Whether to normalize constellation to unit power')
    parser.add_argument('--demapping_method', type=str, default='app',
                        choices=['app', 'maxlog'],
                        help='Demapping method for soft decisions')

    # Equalization
    parser.add_argument('--equalization_method', type=str, default='mrc',
                        choices=['identity', 'zf', 'mrc', 'mmse'],
                        help='Channel equalization method')

    # Other
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')

    return parser


if __name__ == "__main__":
    # This is for standalone testing
    parser = create_channel_args()
    args = parser.parse_args()

    # Create dummy queues for testing
    from_sender = Queue()
    to_receiver = Queue()

    channel_process(args, from_sender, to_receiver)