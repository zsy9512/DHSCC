"""
Main VibrationVLVFENet Module

This module integrates all components into a complete end-to-end system:
- Sender Network (feature extraction, encoding, importance estimation)
- Quantizer (vector quantization)
- Wireless Channel (communication simulation)
- Receiver Network (decoding, reconstruction, classification)
- Adaptive Group Selector (for digital communication modes)
- Training control and phase management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from models.sender_network import SenderNetwork
from models.quantizer import VibrationVectorQuantizer
from models.wireless_channel import WirelessChannelModule
from models.receiver_network import ReceiverNetwork
from models.adaptive_group_selector import AdaptiveGroupSelector
from utils.signal_processing import psnr_to_noise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DHSCCNet(nn.Module):
    """
    Variable-Length Variable Feature Encoder Network for vibration signal classification.
    
    This model implements an end-to-end system for vibration signal classification
    with dynamic feature selection based on channel conditions. It integrates:
    
    1. Sender Network: Feature extraction, encoding, and importance estimation
    2. Quantizer: Vector quantization for digital modes
    3. Wireless Channel: Communication simulation with multiple modes
    4. Receiver Network: Decoding, reconstruction, and classification
    5. Adaptive Group Selector: Dynamic group selection for digital modes
    
    Args:
        args: Configuration parameters
        num_classes: Number of output classes
    """
    def __init__(self, args, num_classes=10):
        super().__init__()
        
        # Store configuration
        self.args = args
        self.num_classes = num_classes
        self.comm_mode = getattr(args, 'communication_mode', 1)  # 使用正确的参数名
        self.hidden_channel = args.intermediate_dim
        # Fix parameter name: use snr_min/max instead of test_snr_min/max
        self.snr_max = getattr(args, 'snr_max', 5.0)
        self.snr_min = getattr(args, 'snr_min', 0.0)
        # Fix missing parameter: test mode is set based on args.mode in validate_arguments
        self.test = getattr(args, 'test', 0)
        
        # Initialize core network modules
        self.sender_network = SenderNetwork(args)
        
        self.quantizer = VibrationVectorQuantizer(
            dimension=128,
            n_codes_group=8,
            bins=16,
            n_q=4*8
        )
        
        self.wireless_channel = WirelessChannelModule(args)
        
        self.receiver_network = ReceiverNetwork(
            args, 
            sender_feature_extractor=self.sender_network.feature_extractor
        )
        
        # Initialize adaptive group selector for digital communication modes
        if self.comm_mode in [3, 4]:
            self.adaptive_group_selector = AdaptiveGroupSelector(
                snr_min=getattr(args, 'snr_min', 0.0),
                snr_max=getattr(args, 'snr_max', 5.0),
                snr_bins=getattr(args, 'selector_snr_bins', 11)
            )
            print(f"Initialized adaptive group selector for communication mode {self.comm_mode}")
        else:
            self.adaptive_group_selector = None
            print(f"No group selector needed for communication mode {self.comm_mode}")
        
        # Training control parameters - match dhscc_train.py parameter names
        self.encoder_pretrain_epochs = getattr(args, 'pretrain_epochs', 100)
        self.encoder_select_epochs = getattr(args, 'selector_epochs', 200)
        self.encoder_weights_frozen = False
        
        # Performance tracking for adaptive selector
        self.batch_snrs = []
        self.batch_groups = []
        
        print(f"Initialized DHSCCNet with communication mode {self.comm_mode}")
        print(f"Training phases: Pretrain(0-{self.encoder_pretrain_epochs}), Select({self.encoder_pretrain_epochs}-{self.encoder_select_epochs}), Final({self.encoder_select_epochs}+)")
        
    def forward(self, x, epoch, snr=0.1):
        """
        End-to-end forward pass through the complete system.
        
        Args:
            x: Input vibration signal [batch_size, channels, time]
            epoch: Current training epoch for stage control
            snr: Signal-to-noise ratio (used in test mode)
            
        Returns:
            tuple: (classification_output, commitment_loss, ber, transmit_mse, 
                   zeros_count, group_to_use, debug_info)
        """
        batch_size = x.size(0)
        
        # Stage control: freeze sender weights during selector training only
        if self.is_selector_phase(epoch) and not self.encoder_weights_frozen:
            self._freeze_sender_weights()
        elif self.is_final_phase(epoch) and self.encoder_weights_frozen:
            # Ensure weights are unfrozen in final phase
            self._unfreeze_sender_weights()
        
        # Determine SNR/noise level
        if self.test:
            snr_value = torch.tensor([snr], device=device).float()
        else:
            snr_value = torch.rand(
                size=(1,), device=device
            ) * (self.snr_max - self.snr_min + torch.finfo(torch.float32).eps) + self.snr_min
        
        # Convert SNR to noise value
        channel_noise = psnr_to_noise(snr_value)
        
        # === SENDER PROCESSING ===
        encoded_features, importance_weights, pre_task_logits, feature_shape, original_encoded_features = self.sender_network(
            x, channel_noise
        )
        
        # Initialize default values
        ber = torch.tensor(0.0).to(device)
        commit_loss = torch.tensor(0.0).to(device)
        zeros_count = 0
        group_to_use = 8
        
        # === QUANTIZATION (for modes 2, 3, 4) ===
        if self.comm_mode in [2, 3, 4]:
            quantized_features, all_indices, commit_loss = self.quantizer(encoded_features)
        else:
            quantized_features, all_indices = None, None
        
        # === GROUP SELECTION (for modes 3, 4) ===
        use_coding = (self.comm_mode == 4)  # Default coding setting
        
        if self.comm_mode in [3, 4] and self.adaptive_group_selector is not None:
            # Apply staged training logic similar to network.py
            if epoch < self.encoder_pretrain_epochs:
                # Pretrain phase: use default group_to_use, disable coding
                group_to_use = 8  # Default value
                use_coding = False  # Disable coding in pretrain phase
                
                # Override with fixed group if specified
                if hasattr(self.args, 'fixed_group') and self.args.fixed_group is not None:
                    group_to_use = self.args.fixed_group
            else:
                # Get adaptive group selection - 在batch级别确定group，而不是sample级别
                # 这样确保同一个batch中所有样本使用相同的group
                # if not hasattr(self, '_current_batch_group') or not self.training:
                #     # 每个新batch或测试时重新计算group          
                self._current_batch_group = self._get_group_to_use(snr_value, epoch)
                group_to_use = self._current_batch_group
                use_coding = (self.comm_mode == 4)  # Restore original coding setting
            
            # Override with fixed group if specified
            if hasattr(self.args, 'fixed_group') and self.args.fixed_group is not None:
                group_to_use = self.args.fixed_group
            
            # Track performance data during training
            if self.training:
                self.batch_snrs.append(snr_value.item())
                self.batch_groups.append(int(group_to_use))  # 确保存储为整数
        
        # === WIRELESS TRANSMISSION ===
        received_features, ber, transmit_mse, sent_features = self.wireless_channel(
            features=encoded_features,
            channel_noise=channel_noise,
            psnr_value=snr_value,  # Use correct parameter name
            # Mode-specific parameters
            attn_weights=importance_weights,
            quantized_features=quantized_features,
            indices=all_indices,
            group_to_use=group_to_use,
            use_coding=use_coding,  # Use the stage-aware coding setting
            quantizer_decode_fn=self.quantizer.decode if self.quantizer else None
        )
        
        # === RECEIVER PROCESSING ===
        classification_output = self.receiver_network(
            received_features, channel_noise, feature_shape
        )
        
        # Prepare debug information
        debug_info = {
            'sent_feature': sent_features.detach().cpu(),
            'encoded_feature': encoded_features.detach().cpu(),
            'original_feature': original_encoded_features.detach().cpu(),
            'snr': snr_value.item(),
            'attn_weights': importance_weights.detach().cpu()
        }
        
        return (classification_output, torch.mean(commit_loss), 
                ber, transmit_mse*channel_noise, zeros_count, 
                group_to_use, debug_info)
    
    def reset_batch_group(self):
        """重置batch级别的group选择，应在每个新batch开始时调用"""
        if hasattr(self, '_current_batch_group'):
            delattr(self, '_current_batch_group')
    
    def set_inference_mode(self, inference_mode: bool):
        """设置推理模式，同时设置自适应选择器的推理模式"""
        if self.adaptive_group_selector is not None:
            self.adaptive_group_selector.set_inference_mode(inference_mode)
        
        # 在推理模式下重置batch group以确保重新计算
        if inference_mode and hasattr(self, '_current_batch_group'):
            delattr(self, '_current_batch_group')
    
    def _get_group_to_use(self, snr_value, epoch):
        """
        Get the number of groups to use based on training phase and SNR.
        
        Args:
            snr_value: Current SNR value
            epoch: Current training epoch
            
        Returns:
            int: Number of groups to use for transmission
        """
        if epoch < self.encoder_pretrain_epochs:
            # Pretrain phase: use default
            return 8
        elif epoch < self.encoder_select_epochs:
            # Selector training phase: use adaptive selection with exploration
            group = self.adaptive_group_selector.get_group_to_use(snr_value.item(), is_training=True)
            return int(group)  # 确保返回整数
        else:
            # Final training phase: use learned strategy without exploration
            group = self.adaptive_group_selector.get_group_to_use(snr_value.item(), is_training=False)
            return int(group)  # 确保返回整数
    
    def update_performance(self, batch_accuracy, epoch):
        """
        Update adaptive selector performance data.
        
        Args:
            batch_accuracy: Current batch accuracy
            epoch: Current epoch
        """
        # 只在选择器训练/更新阶段更新性能数据
        if (self.is_selector_update_phase(epoch) and 
            self.adaptive_group_selector is not None and
            len(self.batch_snrs) > 0 and len(self.batch_groups) > 0):
            
            # 使用最近一个批次的数据更新性能
            snr = self.batch_snrs[-1]
            group = int(self.batch_groups[-1])  # 确保group_to_use为整数
            
            # 调用自适应选择器更新方法
            self.adaptive_group_selector.update_performance(snr, group, batch_accuracy)
    
    # === TRAINING CONTROL METHODS ===
    
    def set_training_phase(self, phase):
        """
        Set the training phase for the model.
        
        Args:
            phase (str): Training phase - 'pretrain', 'selector', or 'final'
        """
        self.training_phase = phase
        
        if phase == 'pretrain':
            # Pretrain phase: train base model with wireless channel but disable coding
            self._unfreeze_all_weights()
            self.encoder_weights_frozen = False
            print("Set to pretrain phase: base model training (coding disabled in pretrain)")
            
        elif phase == 'selector':
            # Selector phase: train adaptive group selector with frozen sender/quantizer
            self._freeze_sender_weights()
            print("Set to selector phase: adaptive group selector training (sender/quantizer frozen)")
            
        elif phase == 'final':
            # Final phase: joint training of full model
            self._unfreeze_sender_weights()
            print("Set to final phase: joint training of full model (all unfrozen)")
            
        else:
            raise ValueError(f"Unknown training phase: {phase}")
    
    def _freeze_base_model(self):
        """Freeze all parameters except adaptive group selector."""
        for name, param in self.named_parameters():
            if 'adaptive_group_selector' not in name:
                param.requires_grad = False
        print("Frozen base model parameters (excluding adaptive group selector)")
    
    def _unfreeze_selector_weights(self):
        """Unfreeze only adaptive group selector parameters."""
        # AdaptiveGroupSelector is not a nn.Module, so no parameters to unfreeze
        # It manages its own internal state and learning process
        if self.adaptive_group_selector is not None:
            print("Adaptive group selector is ready for training (internal state management)")
        else:
            print("No adaptive group selector to unfreeze")
    
    def _unfreeze_all_weights(self):
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True
        print("Unfrozen all model parameters")
    
    def get_training_phase(self):
        """Get current training phase."""
        return getattr(self, 'training_phase', 'final')
    
    def is_pretrain_phase(self, epoch=None):
        """Check if in pretrain phase."""
        if hasattr(self, 'training_phase'):
            return self.training_phase == 'pretrain'
        return epoch is not None and epoch < self.encoder_pretrain_epochs
    
    def is_selector_phase(self, epoch=None):
        """Check if in selector training phase."""
        if hasattr(self, 'training_phase'):
            return self.training_phase == 'selector'
        return (epoch is not None and 
                self.encoder_pretrain_epochs <= epoch < self.encoder_select_epochs)
    
    def is_final_phase(self, epoch=None):
        """Check if in final training phase."""
        if hasattr(self, 'training_phase'):
            return self.training_phase == 'final'
        return epoch is not None and epoch >= self.encoder_select_epochs
    
    def _freeze_sender_weights(self):
        """Freeze sender and quantizer weights during selector training."""
        if self.encoder_weights_frozen:
            return
            
        print(f"\\n=== Freezing Sender Components for Adaptive Selector Training ===")
        
        for param in self.sender_network.parameters():
            param.requires_grad = False
        for param in self.quantizer.parameters():
            param.requires_grad = False
            
        self.encoder_weights_frozen = True
        
        frozen_params = sum(1 for param in self.parameters() if not param.requires_grad)
        total_params = sum(1 for param in self.parameters())
        trainable_params = total_params - frozen_params
        
        print(f"Frozen {frozen_params}/{total_params} parameters")
        print(f"Remaining trainable parameters: {trainable_params}")
        print("Receiver components remain trainable")
        print("=" * 80)
    
    def _unfreeze_sender_weights(self):
        """Unfreeze sender and quantizer weights for final training."""
        print("\\nUnfreezing sender weights for final training...")
        
        for param in self.sender_network.parameters():
            param.requires_grad = True
        for param in self.quantizer.parameters():
            param.requires_grad = True
            
        self.encoder_weights_frozen = False
        
        trainable_params = sum(1 for param in self.parameters() if param.requires_grad)
        total_params = sum(1 for param in self.parameters())
        print(f"Unfrozen weights: {trainable_params}/{total_params} parameters now trainable")
    
    def start_final_training(self, epoch):
        """Initialize final training phase."""
        if epoch >= self.encoder_select_epochs and not hasattr(self, 'final_training_started'):
            print("\\n" + "=" * 80)
            print("STARTING FINAL TRAINING PHASE")
            print("=" * 80)
            
            # Finalize adaptive group selector if available
            if self.adaptive_group_selector is not None:
                # Set inference mode (no more exploration)
                if hasattr(self.adaptive_group_selector, 'set_inference_mode'):
                    self.adaptive_group_selector.set_inference_mode(True)
                    print("✓ Adaptive selector set to inference mode")
                    
                # Finalize training
                if hasattr(self.adaptive_group_selector, 'finalize_training'):
                    self.adaptive_group_selector.finalize_training()
                    print("✓ Adaptive selector training finalized")
            
            # Unfreeze encoder weights for fine-tuning
            self._unfreeze_sender_weights()
            
            self.final_training_started = True
            print("✓ Final training phase initialized")
            print("=" * 80)
    
    # === MODEL STATE MANAGEMENT ===
    
    def save_modular_state(self, epoch, save_dir):
        """
        Save model state in modular format for independent deployment.
        
        Args:
            epoch: Current training epoch
            save_dir: Directory to save model states
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save individual modules
        self.sender_network.save_state(os.path.join(save_dir, f"sender_epoch_{epoch}.pth"))
        self.quantizer.save_state(os.path.join(save_dir, f"quantizer_epoch_{epoch}.pth"))
        self.wireless_channel.save_state(os.path.join(save_dir, f"wireless_epoch_{epoch}.pth"))
        self.receiver_network.save_state(os.path.join(save_dir, f"receiver_epoch_{epoch}.pth"))
        
        # Save adaptive selector if available
        if self.adaptive_group_selector is not None:
            selector_state = self.adaptive_group_selector.save_state()
            torch.save(selector_state, os.path.join(save_dir, f"adaptive_selector_epoch_{epoch}.pth"))
        
        # Save complete model
        torch.save(self.state_dict(), os.path.join(save_dir, f"complete_epoch_{epoch}.pth"))
        
        print(f"Modular model states saved to {save_dir}")
    
    def load_modular_state(self, epoch, save_dir):
        """
        Load model state from modular format.
        
        Args:
            epoch: Epoch number to load
            save_dir: Directory containing model states
        """
        # Load individual modules
        self.sender_network.load_state(os.path.join(save_dir, f"sender_epoch_{epoch}.pth"))
        self.quantizer.load_state(os.path.join(save_dir, f"quantizer_epoch_{epoch}.pth"))
        self.wireless_channel.load_state(os.path.join(save_dir, f"wireless_epoch_{epoch}.pth"))
        self.receiver_network.load_state(os.path.join(save_dir, f"receiver_epoch_{epoch}.pth"))
        
        # Load adaptive selector if available
        if self.adaptive_group_selector is not None:
            selector_path = os.path.join(save_dir, f"adaptive_selector_epoch_{epoch}.pth")
            if os.path.exists(selector_path):
                selector_state = torch.load(selector_path, map_location=device)
                self.adaptive_group_selector.load_state(selector_state)
        
        print(f"Modular model states loaded from {save_dir}")
    
    # === ADAPTIVE SELECTOR MANAGEMENT ===
    
    def get_group_strategy_summary(self):
        """Get current group selection strategy summary."""
        if self.adaptive_group_selector is not None:
            return self.adaptive_group_selector.get_strategy_summary()
        return []
    
    def get_adaptive_selector_statistics(self):
        """Get adaptive selector statistics."""
        if self.adaptive_group_selector is not None:
            stats = self.adaptive_group_selector.get_statistics()
            stats['algorithm_type'] = 'heuristic'
            return stats
        return {'algorithm_type': 'none', 'message': 'No adaptive selector available'}
    
    def reset_adaptive_selector_statistics(self):
        """Reset adaptive selector statistics."""
        if self.adaptive_group_selector is not None:
            self.adaptive_group_selector.reset_statistics()
    
    def save_adaptive_selector_state(self):
        """Save adaptive selector state."""
        if self.adaptive_group_selector is not None:
            state = self.adaptive_group_selector.save_state()
            state['algorithm_type'] = 'heuristic'
            return state
        return None
    
    def load_adaptive_selector_state(self, state):
        """Load adaptive selector state."""
        if self.adaptive_group_selector is not None and state is not None:
            self.adaptive_group_selector.load_state(state)
    
    def get_default_group_to_use(self, snr_value):
        """
        Get default group_to_use for pretraining phase.
        
        Args:
            snr_value: Current SNR value
            
        Returns:
            int: Default group_to_use value
        """
        return 8
    
    def is_selector_training_phase(self, epoch):
        """
        Check if current epoch is in selector training phase.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            bool: True if in selector training phase
        """
        return (epoch >= self.encoder_pretrain_epochs and 
                epoch < self.encoder_select_epochs)
    
    def is_selector_phase(self, epoch):
        """
        Alias for is_selector_training_phase for backward compatibility.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            bool: True if in selector training phase
        """
        return self.is_selector_training_phase(epoch)
    
    def is_selector_update_phase(self, epoch):
        """
        Check if current epoch is in selector update phase.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            bool: True if in selector update phase
        """
        return (epoch >= self.encoder_pretrain_epochs and 
                epoch < self.encoder_select_epochs)
    
    def is_final_phase(self, epoch):
        """
        Check if current epoch is in final training phase.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            bool: True if in final training phase
        """
        return epoch >= self.encoder_select_epochs
        
    def is_final_training_phase(self, epoch):
        """
        Alias for is_final_phase for consistency with original code.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            bool: True if in final training phase
        """
        return self.is_final_phase(epoch)
