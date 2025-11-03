"""
Sender Services Module

This module provides service classes for sender functionality,
extracting business logic from the main process file for better organization.
"""

import torch
import torch.nn as nn
from .sender_model import SenderNet
from .quantizer import VibrationVectorQuantizer, indices_to_binary
import sys
import os
import socket
import pickle
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import create_data_loaders_from_args
from .polar_encoding import Polar5GEncoder
from .dqpsk_modem import modulate_bits
import os
import uhd

class SenderService:
    """
    Service class for sender operations.
    
    Encapsulates model loading, data processing, and encoding logic.
    """
    
    def __init__(self, args):
        """
        Initialize sender service with arguments.
        
        Args:
            args: Parsed command line arguments
        """
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # UDP communication parameters
        self.enable_udp = getattr(args, 'enable_udp', False)
        self.udp_host = getattr(args, 'udp_host', '127.0.0.1')
        self.udp_port = getattr(args, 'udp_port', 50001)
        self.channel_snr_db = getattr(args, 'channel_snr_db', 20.0)  # SNR in dB
        
        # Load models
        self.sender_model, self.quantizer = self._load_sender_models()
        
        # Create data loader
        self.train_loader, self.test_loader, self.num_classes = create_data_loaders_from_args(args)
        self.data_loader = self.test_loader  # Always use test loader for simplified setup
        
        # Initialize polar codec
        self.polar_codec = PolarCodec(
            k=args.K, 
            n=args.N, 
            channel_type="uplink", 
            device=self.device
        )
    
    def _load_sender_models(self):
        """Load trained sender model and quantizer."""
        # Create models
        sender_model = SenderNet(self.args).to(self.device)
        quantizer = VibrationVectorQuantizer(self.args).to(self.device)
        
        # Load sender model
        checkpoint_dir = r'E:\PhD_work\code\uhdcode\dhscc_usrp_sender\direct_polar_training_20251009_210057'
        sender_model_path = os.path.join(checkpoint_dir, 'final_sender_model.pth')
        
        if os.path.exists(sender_model_path):
            sender_checkpoint = torch.load(sender_model_path, map_location=self.device)
            if 'state_dict' in sender_checkpoint:
                sender_model.load_state_dict(sender_checkpoint['state_dict'])
            else:
                sender_model.load_state_dict(sender_checkpoint)
        else:
            raise FileNotFoundError(f"Sender model not found at {sender_model_path}")
        
        # Load quantizer
        quantizer_path = os.path.join(checkpoint_dir, 'final_quantizer.pth')
        
        if os.path.exists(quantizer_path):
            quantizer_checkpoint = torch.load(quantizer_path, map_location=self.device)
            if 'state_dict' in quantizer_checkpoint:
                quantizer.load_state_dict(quantizer_checkpoint['state_dict'])
            else:
                quantizer.load_state_dict(quantizer_checkpoint)
        else:
            raise FileNotFoundError(f"Quantizer not found at {quantizer_path}")
        
        sender_model.eval()
        quantizer.eval()
        
        return sender_model, quantizer
    
    def process_batch(self, data, target, frame_counter=0):
        """
        Process a single batch of data.
        
        Args:
            data: Input data tensor
            target: Target labels
            frame_counter: Frame counter for frame index encoding
            
        Returns:
            tx_signal: DQPSK modulated and RRC filtered signal (numpy array)
        """
        data, target = data.to(self.device), target.to(self.device)
        # Generate channel noise
        channel_noise =1.0
        
        # Forward pass through sender
        sender_output = self.sender_model(data, channel_noise)
        features, importance_weights, pre_task_logits, feature_shape, original_encoded_features = sender_output
        original_encoded_features = original_encoded_features.detach()  # 确保无梯度
        
        # Quantize features
        quantized_features, quantizer_indices, vq_loss = self.quantizer.encode(features)
        
        # Adaptive group selection
        group_to_use = getattr(self.args, 'group_to_use', 8)
        if len(quantizer_indices.shape) == 3:
            selected_indices = quantizer_indices[:, :group_to_use, :]
        else:
            selected_indices = quantizer_indices[:, :group_to_use] if quantizer_indices.shape[1] > group_to_use else quantizer_indices
        #print(f"Selected indices shape: {selected_indices}")
        # Convert indices to binary
        binary_bits = indices_to_binary(selected_indices, bits_per_index=self.args.bits_per_index)
        #print(f"Binary bits{binary_bits}")

        # Perform polar encoding
        codeword = self.polar_codec.encode(binary_bits)
        # Write codeword to txt file in overwrite mode
        #np.save('codeword.npy', codeword.cpu().numpy())
        bits = codeword.reshape(1, -1).to(self.device)
        
        # Convert codeword to bits (assuming binary values)
        bits = (bits.detach().cpu().numpy() > 0).astype(int).flatten()
        #print(f"send  {bits[:20]}, {bits[20:40]}")
        tx_signal = modulate_bits(bits, frame_counter=frame_counter)
        # UDP communication (if enabled)
        if self.enable_udp:
            # Generate frame index
            idx = frame_counter
            
            # Serialize and send selected_indices via UDP
            data = pickle.dumps((idx, tx_signal))
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.sendto(data, (self.udp_host, self.udp_port))
                sock.close()
                #print(f"UDP发送帧 {idx}, indices shape: {selected_indices.shape}")
            except Exception as e:
                print(f"UDP发送失败: {e}")

        # DQPSK modulation and RRC filtering
        #print(f'[调试] bits : {bits}')
        
        return tx_signal
    
    def send_via_usrp(self, codeword, usrp_tx):
        """
        Send codeword via USRP hardware.
        
        Args:
            codeword: Polar encoded codeword (numpy array)
            usrp_tx: Tuple of (usrp, tx_streamer) from init_usrp
        """
        try:
            # Unpack USRP objects
            usrp, tx_streamer = usrp_tx
            
            # Import DQPSK modem
            from .dqpsk_modem import modulate_bits
            
            # Convert codeword to bits (assuming binary values)
            bits = (codeword > 0.5).astype(int).flatten()
            print(f"Codeword转换为 {len(bits)} 比特")
            
            # DQPSK modulation
            tx_signal = modulate_bits(bits)
            print(f"DQPSK调制完成，信号长度: {len(tx_signal)}")
            
            # Create TX metadata
            tx_md = uhd.types.TXMetadata()
            tx_md.start_of_burst = True
            tx_md.end_of_burst = True
            
            # Send via USRP
            print("开始USRP发送...")
            num_sent = tx_streamer.send(tx_signal, tx_md, timeout=1.0)
            print(f"USRP发送完成，发送了 {num_sent} 个采样点")
            
        except Exception as e:
            print(f"USRP发送失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_data_loader(self):
        """Get the data loader for iteration."""
        return self.data_loader

    def simple_qpsk_modulate(self, bits):
        """
        Simple QPSK modulation without complex frame structure.
        
        Args:
            bits: Binary bits array
            
        Returns:
            symbols: Complex QPSK symbols array
        """
        # QPSK constellation (Gray coding)
        constellation = np.array([
            1+1j,   # 00
            -1+1j,  # 01
            1-1j,   # 10
            -1-1j   # 11
        ], dtype=np.complex64) / np.sqrt(2)
        
        symbols = []
        for i in range(0, len(bits), 2):
            bit0 = bits[i]
            bit1 = bits[i+1] if i+1 < len(bits) else 0
            idx = (bit0 << 1) | bit1
            symbols.append(constellation[idx])
        
        return np.array(symbols, dtype=np.complex64)


class PolarCodec:
    """Unified 5G Polar coding codec manager."""
    
    def __init__(self, k=128, n=256, channel_type="uplink", device="cpu"):
        self.k = k
        self.n = n
        self.channel_type = channel_type
        self.device = device
        
        # Initialize 5G polar encoder
        self.encoder = Polar5GEncoder(
            k=k,
            n=n,
            channel_type=channel_type,
            verbose=False,
            dtype=torch.float32
        ).to(device)
    
    def encode(self, binary_bits):
        """Encode binary bits to polar codeword."""
        return self.encoder(binary_bits.float())
    
    def update_device(self, device):
        """Update device for all components."""
        self.device = device
        self.encoder = self.encoder.to(device)