
"""
Receiver Process for Separated End-to-End Deployment
Implements the receiver side: loads model, receives demodulated bits, performs classification.
"""


import os
import sys
import argparse
import torch
import torch.nn as nn
from multiprocessing import Queue


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from receiver.receiver_model import ReceiverNet
from data.data_loader import create_data_loaders_from_args
from sender.quantizer import VibrationVectorQuantizer
from channel.polar_decoding import Polar5GDecoder
from channel.polar_encoding import Polar5GEncoder


class PolarCodec:
    """Unified 5G Polar coding codec manager."""
    def __init__(self, k=128, n=256, channel_type="uplink", device="cpu"):
        self.k = k
        self.n = n
        self.channel_type = channel_type
        self.device = device
        self.encoder = Polar5GEncoder(k=k, n=n, channel_type=channel_type, verbose=False, dtype=torch.float32).to(device)
        self.decoder = Polar5GDecoder(self.encoder).to(device)

    def encode(self, binary_bits):
        return self.encoder(binary_bits.float())

    def decode(self, llrs):
        return self.decoder(llrs)

    def update_device(self, device):
        self.device = device
        self.encoder = self.encoder.to(device)


def load_quantizer_for_receiver(args, device):
    """Load trained quantizer for feature reconstruction at receiver."""
    quantizer = VibrationVectorQuantizer(
        dimension=args.intermediate_dim,
        n_codes_group=args.quantizer_n_codes_group,
        bins=args.quantizer_bins,
        n_q=args.quantizer_n_q
    ).to(device)

    quantizer_path = getattr(args, 'quantizer_path', None)
    if not quantizer_path:
        checkpoint_dir = getattr(args, 'checkpoint_dir', './checkpoints')
        quantizer_path = os.path.join(checkpoint_dir, 'direct_polar_training_*', 'final_quantizer.pth')
        import glob
        matching_files = glob.glob(quantizer_path)
        if matching_files:
            quantizer_path = matching_files[0]

    if quantizer_path and os.path.exists(quantizer_path):
        quantizer_checkpoint = torch.load(quantizer_path, map_location=device)
        quantizer.load_state_dict(quantizer_checkpoint)
    else:
        print(f"[Warning] Could not find quantizer weights at {quantizer_path}")

    return quantizer


def load_receiver_models(args, device, num_classes):
    """Load trained receiver model."""
    receiver_model = ReceiverNet(args, num_classes).to(device)
    receiver_model_path = getattr(args, 'receiver_model_path', None)
    if not receiver_model_path:
        checkpoint_dir = getattr(args, 'checkpoint_dir', './checkpoints')
        if os.path.exists(checkpoint_dir):
            subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
            training_dirs = [d for d in subdirs if 'training' in d or 'direct_polar' in d]
            if training_dirs:
                training_dirs.sort(key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)), reverse=True)
                latest_dir = training_dirs[0]
                receiver_model_path = os.path.join(checkpoint_dir, latest_dir, 'final_receiver_model.pth')

    if receiver_model_path and os.path.exists(receiver_model_path):
        receiver_checkpoint = torch.load(receiver_model_path, map_location=device)
        if 'state_dict' in receiver_checkpoint:
            receiver_model.load_state_dict(receiver_checkpoint['state_dict'])
        else:
            receiver_model.load_state_dict(receiver_checkpoint)
    receiver_model.eval()
    return receiver_model


def receiver_process(args, from_channel_queue):
    """Main receiver process function - directly classify reconstructed features."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, num_classes = create_data_loaders_from_args(args)
    receiver_model = load_receiver_models(args, device, num_classes)
    quantizer = load_quantizer_for_receiver(args, device)
    polar_codec = PolarCodec(k=args.K, n=args.N, channel_type="uplink", device=device)
    receiver_process_main_loop(args, from_channel_queue, receiver_model, quantizer, polar_codec, device)


def binary_to_indices(binary, original_shape, bits_per_index=4):
    """Convert binary back to quantizer indices."""
    device = binary.device
    batch_size, total_bits = binary.shape
    indices = []
    for b in range(batch_size):
        sample_bits = binary[b]
        sample_indices = []
        for i in range(0, total_bits, bits_per_index):
            bit_chunk = sample_bits[i:i+bits_per_index]
            bit_values = bit_chunk.int().tolist()
            idx = sum(bit_val * (2**(bits_per_index-1-j)) for j, bit_val in enumerate(bit_values))
            sample_indices.append(idx)
        indices.append(sample_indices)
    indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
    return indices_tensor.view(original_shape)


def receiver_process_main_loop(args, from_channel_queue, receiver_model, quantizer, polar_codec, device):
    """Main processing loop for receiver."""
    results = {}
    with torch.no_grad():
        while True:
            data_packet = from_channel_queue.get()
            if data_packet.get('type') == 'end':
                break
            batch_idx = data_packet['batch_idx']
            batch_size = data_packet['batch_size']
            snr_db = data_packet.get('snr_db', 15)
            llrs = torch.tensor(data_packet['llrs'], dtype=torch.float, device=device)
            target = torch.tensor(data_packet['target'], dtype=torch.long, device=device)
            feature_shape = data_packet.get('feature_shape', None)
            selected_indices_shape = data_packet.get('selected_indices_shape', None)

            llrs = llrs.view(batch_size, -1)
            
            decoded_bits = polar_codec.decode(llrs)
            received_indices = binary_to_indices(decoded_bits, selected_indices_shape, bits_per_index=4)
            reconstructed_features = quantizer.decode(received_indices)
            if feature_shape is not None:
                if isinstance(feature_shape, (list, tuple)):
                    feature_shape = torch.Size(feature_shape)
            else:
                feature_shape = (batch_size, 8, 4)
            channel_noise = torch.randn(batch_size, 1, device=device) * 0.1 + 1.0
            receiver_output = receiver_model(reconstructed_features, channel_noise, feature_shape)
            loss = nn.CrossEntropyLoss()(receiver_output, target)
            predictions = receiver_output.argmax(dim=1)
            batch_correct = (predictions == target).sum().item()
            if snr_db not in results:
                results[snr_db] = {'correct': 0, 'total': 0, 'loss': 0.0}
            results[snr_db]['correct'] += batch_correct
            results[snr_db]['total'] += batch_size
            results[snr_db]['loss'] += loss.item()
    import sys
    original_stdout = sys.__stdout__
    sys.stdout = original_stdout
    print("\n分离式部署测试结果:")
    print("="*30)
    for snr in sorted(results.keys()):
        acc = results[snr]['correct'] / results[snr]['total'] * 100
        print(f"SNR {snr}dB: Accuracy = {acc:.2f}%")
    sys.stdout = open(os.devnull, 'w')

def create_receiver_args():
    """Create argument parser for receiver process."""
    parser = argparse.ArgumentParser(description='Receiver Process')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default="data/vibcls/vibrodata")
    parser.add_argument('--train_list', type=str, default="data/vibcls/train.txt")
    parser.add_argument('--test_list', type=str, default="data/vibcls/test.txt")
    parser.add_argument('--class_map', type=str, default="data/vibcls/clsmap.txt")

    # Quantizer parameters
    parser.add_argument('--intermediate_dim', type=int, default=128)
    parser.add_argument('--quantizer_n_codes_group', type=int, default=8)
    parser.add_argument('--quantizer_bins', type=int, default=16)
    parser.add_argument('--quantizer_n_q', type=int, default=32)  # Match training default
    parser.add_argument('--N', type=int, default=256, help='Polar code length')
    parser.add_argument('--K', type=int, default=128, help='Information bits')
    parser.add_argument('--snr_min', type=int, default=10, help='Minimum SNR for testing')
    parser.add_argument('--snr_max', type=int, default=21, help='Maximum SNR for testing')

    # Other parameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--max_batches', type=int, help='Maximum batches to process (for testing)')

    # Model paths
    parser.add_argument('--receiver_model_path', type=str, help='Path to receiver model weights')
    parser.add_argument('--quantizer_path', type=str, help='Path to quantizer weights')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')

    return parser


if __name__ == "__main__":
    parser = create_receiver_args()
    args = parser.parse_args()
    from_channel = Queue()
    receiver_process(args, from_channel)