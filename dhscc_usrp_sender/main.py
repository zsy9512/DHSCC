"""
Main Entry Point for DHSCC USRP Sender with Multi-threading

This module initializes USRP hardware and coordinates multi-threaded sending process.
"""

import sys
import os
import argparse
from sender.usrp_tx import TXProgram

def create_args():
    """Create argument parser with all necessary parameters."""
    parser = argparse.ArgumentParser(description='DHSCC USRP Sender')
    
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
    
    # USRP parameters
    parser.add_argument('--tx_freq', type=float, default=915e6)
    parser.add_argument('--rate', type=float, default=1e6)
    parser.add_argument('--tx_gain', type=float, default=50)
    parser.add_argument('--repeat_count', type=int, default=10)
    
    # UDP communication parameters
    parser.add_argument('--enable_udp', action='store_true', help='Enable UDP communication mode')
    parser.add_argument('--udp_host', type=str, default='127.0.0.1', help='UDP receiver host')
    parser.add_argument('--udp_port', type=int, default=50001, help='UDP receiver port')
    parser.add_argument('--channel_snr_db', type=float, default=20.0, help='Channel SNR in dB')
    
    # New parameter for category selection
    parser.add_argument('--category', type=int, choices=range(10), help='Specify category (0-9) to load specific test data. If not specified, uses default test.txt')
    
    return parser

def main():
    """
    Main function: Initialize USRP, start data generation and TX threads
    """
    try:
        # Set working directory
        os.chdir("e:\\PhD_work\\code\\uhdcode\\dhscc_usrp_sender")
        
        # Create argument parser and set defaults
        parser = create_args()
        
        # Parse arguments (use command line arguments if provided, otherwise use defaults)
        args = parser.parse_args()
        
        # If no category specified, print default behavior
        if args.category is None:
            print("No category specified, using default test.txt data")
        else:
            print(f"Using category {args.category} test data (test_{args.category}.txt)")
        
        # Create TX program with multi-threading
        tx_program = TXProgram(
            center_freq=args.tx_freq,
            samp_rate=args.rate,
            tx_gain=args.tx_gain,
            args=args
        )
        
        # Start the program (this will run indefinitely)
        tx_program.start()
        
    except Exception as e:
        print(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()