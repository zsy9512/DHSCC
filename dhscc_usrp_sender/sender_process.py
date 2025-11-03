"""
Sender Process for Separated End-to-End Deployment

This module implements the sender side of t    if quantizer_path and os.path.exists(quantizer_path):
        quantizer_checkpoint = torch.load(quantizer_path, map_location=device)
        # Handle different checkpoint formats
        if 'state_dict' in quantizer_checkpoint:
            quantizer.load_state_dict(quantizer_checkpoint['state_dict'])
        else:
            quantizer.load_state_dict(quantizer_checkpoint)
        # print(f"✅ Loaded quantizer from {quantizer_path}")
    else:
        # print("⚠️  Quantizer not found, using randomly initialized quantizer")
        passed communication system.
It loads the trained sender model and quantizer, processes input data,
and sends encoded bits through IPC to the channel process.
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

from sender.services import SenderService

def sender_process(args, to_channel_queue=None, from_channel_queue=None, test_mode=False, usrp_tx=None):
    """Main sender process function with polar encoding."""
    
    # Initialize sender service
    sender_service = SenderService(args)
    data_loader = sender_service.get_data_loader()
    
    # Process data
    codewords = []
    processed_batches = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            
            # Process batch using service
            codeword = sender_service.process_batch(data, target)
            
            if test_mode:
                codewords.append(codeword)
            else:
                # Normal mode: send to queue (if queue provided)
                if to_channel_queue:
                    # Prepare data packet
                    data_packet = {
                        'codeword': codeword,
                        'batch_idx': batch_idx,
                    }
                    to_channel_queue.put(data_packet)
                
                # Send via USRP if provided
                if usrp_tx is not None:
                    sender_service.send_via_usrp(codeword, usrp_tx)
            
            processed_batches += 1
            if hasattr(args, 'max_batches') and args.max_batches and processed_batches >= args.max_batches:
                break
    
    if test_mode:
        return codewords
    elif to_channel_queue:
        to_channel_queue.put({'type': 'end'})


def create_sender_args():
    """Create argument parser for sender process."""
    parser = argparse.ArgumentParser(description='Sender Process')

    # Model parameters
    parser.add_argument('--intermediate_dim', type=int, default=128)

    # Communication parameters
    parser.add_argument('--bits_per_index', type=int, default=4)
    parser.add_argument('--group_to_use', type=int, default=8, help='Number of groups to transmit')
    parser.add_argument('--N', type=int, default=256, help='Polar code length')
    parser.add_argument('--K', type=int, default=128, help='Information bits')

    # Model paths
    parser.add_argument('--sender_model_path', type=str, help='Path to sender model weights')
    parser.add_argument('--quantizer_path', type=str, help='Path to quantizer weights')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')

    # Other parameters
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--max_batches', type=int, help='Maximum batches to process (for testing)')

    return parser


if __name__ == "__main__":
    # This is for standalone testing
    parser = create_sender_args()
    args = parser.parse_args()

    # Create dummy queues for testing
    to_channel = Queue()
    from_channel = Queue()

    sender_process(args, to_channel, from_channel)