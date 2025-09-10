#!/usr/bin/env python3
"""
VL-VFE (Variable-Length Vibration Feature Extraction) with VIB (Variational Information Bottleneck)
for vibration signal classification.

This model implements adaptive feature selection for vibration classification
under variable channel conditions with SNR-dependent compression.

Usage:
    # Training
    python VL_VFE_VIB.py --epochs 300 --batch_size 128 --lr 1e-3
    
    # Testing with specific SNR
    python VL_VFE_VIB.py --test 1 --weights path/to/model.pth --snr 2.0
    
    # SNR range analysis
    python VL_VFE_VIB.py --test 1 --weights path/to/model.pth --snr -1 --snr_min 0 --snr_max 5 --snr_step 111

Requirements:
    pip install torch torchaudio numpy matplotlib
"""

# Check for required dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import StepLR
    import torchaudio
except ImportError as e:
    print(f"Error: Missing PyTorch dependencies: {e}")
    print("Please install PyTorch with: pip install torch torchaudio")
    exit(1)

try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error: Missing dependencies: {e}")
    print("Please install with: pip install numpy matplotlib")
    exit(1)

import argparse
import copy
import math
import os
import sys
import os.path

# Add the project root to path to import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)


from utils.data import load_file_list, load_class_names, WavDataset
from wireless_comm.interface import WirelessInterface
from models.quantizer import VibrationVectorQuantizer
from models.receiver_network import ReceiverNetwork
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser(
    description='VL-VFE VIB: Variable-Length Vibration Feature Extraction with Variational Information Bottleneck',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Model parameters
parser.add_argument('--intermediate_dim', type=int, default=256, 
                    help='Intermediate dimension size for feature compression')

# Training parameters
parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--gamma', type=float, default=0.5, help='Learning rate decay factor')
parser.add_argument('--beta', type=float, default=1e-2, help='KL divergence weight coefficient')
parser.add_argument('--threshold', type=float, default=0, help='Threshold for feature pruning')
parser.add_argument('--decay_step', type=int, default=60, help='Learning rate decay step size')

# Execution mode
parser.add_argument('--test', type=int, default=0, choices=[0, 1],
                    help='Execution mode: 0=training, 1=testing')
parser.add_argument('--weights', type=str, default=None,
                    help='Path to model weights file for testing/resuming training')

# SNR parameters
parser.add_argument('--snr', type=float, default=10.0, 
                    help='SNR for single test in dB (use negative value for range testing)')
parser.add_argument('--snr_min', type=float, default=0, 
                    help='Minimum SNR for training/testing range in dB')
parser.add_argument('--snr_max', type=float, default=5, 
                    help='Maximum SNR for training/testing range in dB')
parser.add_argument('--snr_step', type=int, default=11, 
                    help='Number of SNR steps for comprehensive testing')

# Data parameters
parser.add_argument('--sample_rate', type=int, default=2800, 
                    help='Sample rate for audio files')

args = parser.parse_args()

# Validate arguments
if args.test == 1 and args.weights is None:
    print("Warning: Testing mode enabled but no weights file provided. Using random weights.")

if args.snr_min >= args.snr_max:
    print("Error: snr_min must be less than snr_max")
    exit(1)
# Dataset preparation
def prepare_datasets():
    """Prepare train and test datasets for vibration classification"""
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..')
    dataset_root = os.path.join(project_root, 'dataset', 'vibcls')
    
    # Check if dataset directory exists
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset directory not found at {dataset_root}")
        print("Please ensure the dataset is properly placed in the project structure")
        exit(1)
    
    # Check for required dataset files
    required_files = ['clsmap.txt', 'train.txt', 'test.txt']
    for file in required_files:
        file_path = os.path.join(dataset_root, file)
        if not os.path.exists(file_path):
            print(f"Error: Required dataset file not found: {file_path}")
            exit(1)
    
    # Check for data directory
    data_dir = os.path.join(dataset_root, 'vibrodata')
    if not os.path.exists(data_dir):
        print(f"Error: Vibration data directory not found at {data_dir}")
        exit(1)
    
    print(f"Dataset root: {dataset_root}")
    print(f"Data directory: {data_dir}")
    
    # Load class mapping
    try:
        class_names = load_class_names(os.path.join(dataset_root, 'clsmap.txt'))
    except Exception as e:
        print(f"Error loading class mapping: {e}")
        exit(1)
        
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    for idx, name in class_names.items():
        print(f"Class {idx}: {name}")
    
    # Load file lists
    try:
        train_list = load_file_list(os.path.join(dataset_root, 'train.txt'))
        test_list = load_file_list(os.path.join(dataset_root, 'test.txt'))
    except Exception as e:
        print(f"Error loading file lists: {e}")
        exit(1)
    
    print(f"Training samples: {len(train_list)}")
    print(f"Test samples: {len(test_list)}")
    
    if len(train_list) == 0 or len(test_list) == 0:
        print("Error: Empty dataset found")
        exit(1)
    
    # Create datasets
    try:
        train_dataset = WavDataset(
            train_list,
            data_dir,
            augment=True,
            sample_rate=args.sample_rate
        )
        
        test_dataset = WavDataset(
            test_list,
            data_dir,
            augment=False,
            sample_rate=args.sample_rate
        )
    except Exception as e:
        print(f"Error creating datasets: {e}")
        exit(1)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1000, 
        shuffle=False, 
        num_workers=2
    )
    
    return train_loader, test_loader, num_classes




def seed_torch(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class gamma_layer(nn.Module):

    def __init__(self, input_channel, output_channel):
        super(gamma_layer, self).__init__()
        self.H = nn.Parameter(torch.ones(output_channel, input_channel))
        self.b = nn.Parameter(torch.ones(output_channel))
        self.H.data.normal_(0, 0.1)
        self.b.data.normal_(0, 0.001)

    def forward(self, x):
        H = torch.abs(self.H)
        x = F.linear(x,H)
        return torch.tanh(x)

class gamma_function(nn.Module):

    def __init__(self):
        super(gamma_function, self).__init__()
        self.f1 = gamma_layer(1,16)
        self.f2 = gamma_layer(16,16)
        self.f3 = gamma_layer(16,16)
        self.f4 = gamma_layer(16,args.intermediate_dim)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        return x

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x): 
        return x*self.weight

class VibNet(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()

        self.args = args
        self.hidden_channel = args.intermediate_dim
        self.num_classes = num_classes

        # 1D convolutional layers for vibration signal processing
        # Input: 1x512 vibration signal
        self.prep = nn.Sequential(
                    nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3, bias=False),
                    nn.BatchNorm1d(64),
                    nn.ReLU()
                    )
        
        self.layer1 = nn.Sequential(
                    nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2)  # 512 -> 256
                    )
        
        self.layer1_res = nn.Sequential(
                    nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm1d(128),
                    nn.ReLU()
                    )
        
        self.layer2 = nn.Sequential(
                    nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2)  # 256 -> 128
                    )
        
        self.layer3 = nn.Sequential(
                    nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=4, stride=4)  # 128 -> 32
                    )
        
        self.layer3_res = nn.Sequential(
                    nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm1d(512),
                    nn.ReLU()
                    )

        # Classifier layers
        self.classifier1 = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),  # Global average pooling
                    Flatten()
                    )
        self.classifier2 = nn.Sequential(
                    nn.Linear(512, self.num_classes, bias=False),
                    Mul(0.125)
                    )
        
        # Encoder layers for VL-VFE
        self.encoder1 = nn.Sequential(
                        nn.Conv1d(512, 4, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm1d(4),
                        nn.ReLU()
                        )
        
        # Adaptive pooling to get fixed size output
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16)  # 32 -> 16
        
        self.encoder2 = nn.Sequential(
                        nn.Linear(64, 64),  # 4 * 16 = 64
                        nn.Sigmoid()
                        )

        # Encoder 3 parameters
        self.encoder3_weight = nn.Parameter(torch.Tensor(self.hidden_channel, 64))
        self.encoder3_bias = nn.Parameter(torch.Tensor(self.hidden_channel))
        self.encoder3_weight.data.normal_(0, 0.5)
        self.encoder3_bias.data.normal_(0, 0.1)

        # Decoder layers
        self.decoder1 = nn.Linear(self.hidden_channel, 64)
        self.decoder1_2 = nn.Sequential(
                        nn.Linear(64, 64),
                        nn.ReLU()
                        )
        self.decoder1_2_2 = nn.Sequential(
                        nn.Linear(1, 16),
                        nn.ReLU(),
                        nn.Linear(16, 16),
                        nn.ReLU(),
                        nn.Linear(16, 16),
                        nn.ReLU()
                        )
        self.decoder1_3 = nn.Sequential(
                        nn.Linear(64+16, 64),
                        nn.ReLU()
                        )
        
        # Upsample to match the encoder size
        self.decoder2_upsample = nn.Upsample(size=32, mode='linear', align_corners=False)
        self.decoder2 = nn.Sequential(
                        nn.Conv1d(4, 512, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm1d(512),
                        nn.ReLU()
                        )

        self.Tanh = nn.Tanh()
        self.gamma_mu = gamma_function().to(device)
        self.upper_tri_matrix = torch.triu(torch.ones((args.intermediate_dim, args.intermediate_dim))).to(device)
        # 初始化无线传输接口
        self.receiver= ReceiverNetwork(args)
        self.wireless_interface = WirelessInterface(comm_mode=3,channel_model='awgn')
        self.quantizer = VibrationVectorQuantizer(
            dimension=128,
            n_codes_group=8,
            bins=16,
            n_q=32
        )
    def forward(self, x, epoch, snr=2.0):
        # Input processing
        x = self.prep(x)
        x = self.layer1(x)
        res = self.layer1_res(x)
        x = res + x
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Encoder
        encoded_features = self.encoder1(x)  # [batch, 4, 32]
        former_shape = encoded_features.shape
        encoded_features = self.adaptive_pool(encoded_features)  # [batch, 4, 16]
        x = torch.reshape(encoded_features, (encoded_features.size()[0], 4*16))  # [batch, 64]

        # Dynamic Channel Conditions
        if self.training:
            # Random SNR sampling during training
            random_snr = torch.rand(1) * (args.snr_max - args.snr_min+1e-6) + args.snr_min
            channel_noise = 10 ** (-random_snr / 20.0)
            snr=random_snr
        else:
            # Convert SNR to channel noise for testing
            channel_noise = torch.FloatTensor([10 ** (-snr / 20.0)])
        snr=snr.cpu().item() if isinstance(snr, torch.Tensor) else snr
        channel_noise = channel_noise.to(device)
        x = self.encoder2(x)
        x_norm2 = torch.norm(x, dim=1)
        x = 64 * (x.permute(1,0)/(x_norm2+1e-6)).permute(1,0)
        
        weight3 = F.tanh(self.encoder3_weight)
        bias3 = F.tanh(self.encoder3_bias)
        weight3 = torch.clamp(torch.abs(weight3), min=1e-3) * torch.sign(weight3.detach())
        bias3 = torch.clamp(torch.abs(bias3), min=1e-3) * torch.sign(bias3.detach())
        l2_norm_squared = torch.sum(weight3.pow(2), dim=1) + bias3.pow(2)
        l2_norm = l2_norm_squared.pow(0.5)
        weight3 = (weight3.permute(1,0) / (l2_norm+1e-6)).permute(1,0)
        bias3 = bias3 / (l2_norm+1e-6)
        x = F.linear(x, weight3, bias3)

        mu = self.gamma_mu(channel_noise)
        mu = F.linear(mu, self.upper_tri_matrix)
        mu = torch.clamp(mu, min=1e-4)
        encoded_feature = torch.tanh(x * mu)
        encoded_feature = torch.clamp(torch.abs(encoded_feature), min=1e-2) * torch.sign(encoded_feature.detach())
        
        # KL divergence
        

        # Gaussian channel noise
        #x = encoded_feature + torch.randn_like(encoded_feature) * channel_noise

        # if self.training:
        #     if epoch > 60:
        #         x = x * self.get_mask(mu, threshold=args.threshold)
        # else:
        #     x = x * self.get_mask(mu, threshold=args.threshold)
       
        quantized_features, all_indices, commit_loss  = self.quantizer(encoded_feature)
        KL = self.KL_log_uniform(channel_noise, torch.abs(quantized_features.detach()))
        transmission_result = self.wireless_interface.transmit_mode3(
                indices=all_indices,
                snr_db=snr,
                bits_per_index=4,
                group_to_use=8
            )
        received_indices = transmission_result['received_indices']
        # 反量化
        received_features_decoded = self.quantizer.decode(received_indices)
        
        # For gradient flow: use quantized_features as base and add detached difference
        # This ensures gradients flow through the quantizer but receiver only sees transmitted+zero data
        received_features = quantized_features + (received_features_decoded - quantized_features).detach()
        # Decoder
        # x = F.relu(self.decoder1(received_features))
        # x = self.decoder1_2(x)
        # noise_feature = self.decoder1_2_2(channel_noise)
        # noise_feature = noise_feature.expand(x.size()[0], 16)
        # x = torch.cat((x, noise_feature), dim=1)
        # x = self.decoder1_3(x)
        
        # # Reshape and decode
        # x = torch.reshape(x, (-1, 4, 16))
        # x = self.decoder2_upsample(x)  # Upsample to [batch, 4, 32]
        # decoded_feature = self.decoder2(x)
        
        # # Residual connection
        # x = self.layer3_res(decoded_feature)
        # x = x + decoded_feature
        
        # # Classification
        # x = self.classifier1(x)
        # output = self.classifier2(x)
        output=self.receiver(received_features, former_shape)
        return output, (KL * 0.1 / channel_noise),torch.mean(commit_loss)

    def KL_log_uniform(self, channel_noise, encoded_feature):
        alpha = (channel_noise/encoded_feature)
        k1 = 0.635766
        k2 = 1.8732
        k3 = 1.48695
        batch_size = alpha.size(0)
        KL_term = k1 * F.sigmoid(k2 + k3 * 2 * torch.log(alpha)) - 0.5 * F.softplus(-2 * torch.log(alpha)) - k1
        return - torch.sum(KL_term) / batch_size

    def get_mask(self, mu, threshold=args.threshold):
        alpha = mu.detach()
        hard_mask = (alpha > threshold).float()
        return hard_mask

    def get_mask_inference(self, snr, threshold=args.threshold):
        channel_noise = torch.FloatTensor([10 ** (-snr / 20.0)]).to(device)
        mu = self.gamma_mu(channel_noise)
        alpha = F.linear(mu, self.upper_tri_matrix)
        hard_mask = (alpha > threshold).float()
        return hard_mask, alpha

def train(model, train_loader, test_loader):
    
    test_acc = 0
    pruned_dim = 0
    saved_model = {}
    
    # Get current directory for saving checkpoints
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(current_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):

        print('\nepoch:{}'.format(epoch))
        
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            
            # Ensure input is the right shape for 1D convolution
            if len(x.shape) == 2:  # [batch, 512]
                x = x.unsqueeze(1)  # [batch, 1, 512]
            elif len(x.shape) == 3 and x.shape[1] > 1:  # [batch, channels, 512]
                x = x.mean(dim=1, keepdim=True)  # Average channels -> [batch, 1, 512]
            
            model.train()
            output, KL,commit_loss = model(x, epoch)
            criterion = nn.CrossEntropyLoss()
            criterion = criterion.to(device)
            loss1 = criterion(output, y)
            
            if epoch <= 20:
                loss = loss1+0.1*commit_loss
            else:
                anneal_ratio = min(1, (epoch - 20)/20)
                loss = loss1 + args.beta * KL * anneal_ratio+0.1*commit_loss

            if torch.isnan(loss):
                raise Exception("NaN value")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        scheduler.step()

        acc, pruned_number, _ = test(model, test_loader, epoch, snr=10.0)

        print('Test Accuracy: ', acc, 'Pruned dim:', pruned_number, 'Activated dim:', args.intermediate_dim - pruned_number)

        if epoch > 180:
            if (acc > test_acc and pruned_number == pruned_dim) or pruned_number > pruned_dim:
                test_acc = acc
                pruned_dim = pruned_number
                saved_model = copy.deepcopy(model.state_dict())
                print('Best ckpt:', test_acc, 'pruned_number:', pruned_dim, 'beta:', args.beta)
                checkpoint_path = os.path.join(checkpoint_dir, 'VIB_best_ckpt_dim:{}_beta:{}_model.pth'.format(args.intermediate_dim, args.beta))
                torch.save({'model': saved_model}, checkpoint_path)
                
    print('Best Accuracy:', test_acc, 'Intermediate Dim:', args.intermediate_dim, 'Beta:', args.beta)
    final_model_path = os.path.join(checkpoint_dir, 'VIB_model_dim:{}_beta:{}_accuracy:{:.2f}_model.pth'.format(args.intermediate_dim - pruned_dim, args.beta, test_acc))
    torch.save({'model': saved_model}, final_model_path)

def test(model, test_loader, epoch, snr=2.0, collect_mu=False):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        mu_values = None
        
        for i, (signals, labels) in enumerate(test_loader): 
            signals = signals.to(device)
            labels = labels.to(device)
            
            # Ensure input is the right shape for 1D convolution
            if len(signals.shape) == 2:  # [batch, 512]
                signals = signals.unsqueeze(1)  # [batch, 1, 512]
            elif len(signals.shape) == 3 and signals.shape[1] > 1:  # [batch, channels, 512]
                signals = signals.mean(dim=1, keepdim=True)  # Average channels -> [batch, 1, 512]
            
            outputs, _,_= model(signals, epoch, snr)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        hard_mask, mu = model.get_mask_inference(snr)
        index = torch.nonzero(torch.lt(hard_mask, 0.5)).squeeze(1)
        pruned_number = index.size()[0]
        
        if collect_mu:
            mu_values = mu.cpu().numpy()
            
        return 100 * correct / total, pruned_number, mu_values if collect_mu else None

def collect_mu_statistics(mu_values):
    """Collect statistics about mu values"""
    return {
        "mean": np.mean(mu_values),
        "std": np.std(mu_values),
        "min": np.min(mu_values),
        "max": np.max(mu_values),
        "median": np.median(mu_values),
        "values": mu_values
    }

def noise_to_snr(noise_level):
    """Convert noise level to SNR in dB"""
    return 20 * math.log10(1.0 / noise_level)

def snr_to_noise(snr_db):
    """Convert SNR in dB to noise level"""
    return 10 ** (-snr_db / 20.0)

def test_snr_range(model, test_loader, snr_min=0, snr_max=5, steps=111):
    """Test model across a range of SNR levels"""
    snr_values = np.linspace(snr_min, snr_max, steps)
    accuracies = []
    pruned_dims = []
    mu_stats = []
    
    for snr in snr_values:
        #print(f"Testing SNR: {snr:.2f} dB")
        accuracy = 0
        pruned_number = 0
        all_mu_values = []
        
        t = 5  # Run multiple tests and average for stability
        for i in range(t):
            if i == 0:  # Collect mu values only on the first run to save memory
                acc, pruned, mu_values = test(model, test_loader, 0, snr, collect_mu=True)
                all_mu_values = mu_values
            else:
                acc, pruned, _ = test(model, test_loader, 0, snr, collect_mu=False)
            accuracy += acc
            pruned_number = pruned  # Take the last pruned number
        
        accuracies.append(accuracy / t)
        pruned_dims.append(pruned_number)
        
        # Collect statistics about mu values
        stats = collect_mu_statistics(all_mu_values)
        mu_stats.append(stats)
        
        print(f'SNR, {snr:.2f} dB, Accuracy, {accuracy/t:.2f}')
    
    return snr_values, accuracies, pruned_dims, mu_stats

def plot_results(snr_values, accuracies, pruned_dims, mu_stats=None, output_dir='.'):
    """Plot results: Accuracy vs SNR and Pruned dimensions vs SNR"""
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Accuracy vs SNR
    plt.subplot(1, 2, 1)
    plt.plot(snr_values, accuracies, 'o-', color='blue')
    plt.grid(True)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy (%)')
    plt.title('Vibration Classification Accuracy vs SNR')
    
    # Plot 2: Active dimensions vs SNR
    active_dims = [args.intermediate_dim - p for p in pruned_dims]
    plt.subplot(1, 2, 2)
    plt.plot(snr_values, active_dims, 'o-', color='green')
    plt.grid(True)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Active Dimensions')
    plt.title('Active Dimensions vs SNR')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/VL_VFE_VIB_SNR_results.png', dpi=300, bbox_inches='tight')
    print(f"Results saved to {output_dir}/VL_VFE_VIB_SNR_results.png")
    plt.close()
    
    # If mu statistics are provided, plot additional mu analysis
    if mu_stats is not None:
        plot_mu_distribution(snr_values, mu_stats, output_dir, pruned_dims)

def plot_mu_distribution(snr_values, mu_stats, output_dir='.', pruned_dims=None):
    """Create detailed plots of mu values at different SNR levels"""
    plt.figure(figsize=(14, 12))
    
    # Plot 1: Boxplot of mu values
    plt.subplot(2, 2, 1)
    data = [stats['values'] for stats in mu_stats]
    plt.boxplot(data, tick_labels=[f"{s:.1f}" for s in snr_values])
    plt.xlabel('SNR (dB)')
    plt.ylabel('mu values')
    plt.title('Distribution of mu values at different SNR levels')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Mean and std of mu values
    plt.subplot(2, 2, 2)
    means = [stats['mean'] for stats in mu_stats]
    stds = [stats['std'] for stats in mu_stats]
    plt.errorbar(snr_values, means, yerr=stds, fmt='o-', capsize=5, markersize=8, color='blue')
    plt.xlabel('SNR (dB)')
    plt.ylabel('mu value')
    plt.title('Mean and Std Dev of mu values')
    plt.grid(True)
    
    # Plot 3: Histogram of mu values at selected SNRs
    plt.subplot(2, 2, 3)
    num_bins = 50
    
    # Select a few representative SNRs to avoid overcrowding
    indices = np.linspace(0, len(snr_values)-1, min(4, len(snr_values)), dtype=int)
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, idx in enumerate(indices):
        plt.hist(mu_stats[idx]['values'], num_bins, alpha=0.5, 
                 label=f"SNR={snr_values[idx]:.1f}dB", color=colors[i])
    
    plt.xlabel('mu values')
    plt.ylabel('Frequency')
    plt.title('Histogram of mu values at selected SNR levels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Active dimensions vs mu threshold
    plt.subplot(2, 2, 4)
    sample_idx = len(snr_values) // 2 if len(snr_values) > 1 else 0  # Use middle SNR for this analysis
    thresholds = np.linspace(0, np.max(mu_stats[sample_idx]['values']), 100)
    active_dims = []
    
    for thresh in thresholds:
        active = np.sum(mu_stats[sample_idx]['values'] > thresh)
        active_dims.append(active)
    
    plt.plot(thresholds, active_dims)
    
    # Only show average active dims line if pruned_dims is provided
    if pruned_dims is not None and len(pruned_dims) > 0:
        avg_pruned = np.mean([p for p in pruned_dims])
        plt.axhline(y=args.intermediate_dim - avg_pruned, 
                    color='r', linestyle='--', 
                    label=f'Avg Active Dims: {args.intermediate_dim - avg_pruned:.1f}')
    
    plt.axvline(x=args.threshold, color='g', linestyle='--', 
                label=f'Current Threshold: {args.threshold}')
    plt.xlabel('mu threshold')
    plt.ylabel('Active Dimensions')
    plt.title(f'Active Dimensions vs mu threshold at SNR={snr_values[sample_idx]:.1f}dB')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/VL_VFE_VIB_mu_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Mu analysis saved to {output_dir}/VL_VFE_VIB_mu_analysis.png")
    plt.close()

if __name__=='__main__':
    seed_torch(0)
    
    # Prepare datasets
    train_loader, test_loader, num_classes = prepare_datasets()
    
    # Initialize model
    model = VibNet(args, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
    
    if args.test == 0:
        train(model, train_loader, test_loader)
    else:
        # Create output directory for plots (relative to current directory)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model weights
        if args.weights:
            model.load_state_dict(torch.load(args.weights)['model'])
        
        # Test a single SNR level if specified with a positive value
        if args.snr > 0:
            accuracy = 0
            t = 20
            mu_values = None
            pruned_number = 0
            
            for i in range(t):
                if i == 0:  # Collect mu values only on the first run
                    acc, pruned, mu = test(model, test_loader, 0, args.snr, collect_mu=True)
                    pruned_number = pruned
                    mu_values = mu
                else:
                    acc, pruned, _ = test(model, test_loader, 0, args.snr, collect_mu=False)
                    pruned_number = pruned  # Take the last value
                accuracy += acc
            
            print(f'SNR: {args.snr:.2f} dB')
            print(f'Test Accuracy: {accuracy/t:.2f}%, Pruned dim: {pruned_number}, Activated dim: {args.intermediate_dim - pruned_number}')
            
            # Plot mu distribution for single SNR level
            if mu_values is not None:
                mu_stats = [collect_mu_statistics(mu_values)]
                snr_values = [args.snr]
                plot_mu_distribution(snr_values, mu_stats, output_dir)
        else:
            # Run comprehensive testing across SNR levels
            print("Running comprehensive SNR level analysis...")
            
            # For continuous range analysis
            snr_values, accuracies, pruned_dims, mu_stats = test_snr_range(
                model, test_loader, args.snr_min, args.snr_max, args.snr_step)
            plot_results(snr_values, accuracies, pruned_dims, mu_stats, output_dir)
            
            # Save the results to a file for future reference
            results = {
                'snr_values': snr_values.tolist(),
                'accuracies': accuracies,
                'active_dims': [args.intermediate_dim - p for p in pruned_dims],
            }
            
            np.save(f'{output_dir}/VL_VFE_VIB_snr_results.npy', results)
            print(f"Numerical results saved to {output_dir}/VL_VFE_VIB_snr_results.npy")
