#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration examples for different wireless communication modes and channel models.

This file demonstrates how to configure the WirelessInterface for different scenarios,
including various channel models and transmission modes.
"""

class WirelessCommConfig:
    """Base configuration class for wireless communication."""
    
    def __init__(self):
        # Communication mode (1-4)
        self.comm_mode = 4
        
        # Coding parameters
        self.N = 256  # Code length (must be power of 2)
        self.k = 128  # Information bits
        self.channel_type = "uplink"  # or "downlink"
        self.decoder_type = "SC"  # or "SCL"
        self.list_size = 8  # For SCL decoder
        
        # Modulation parameters
        self.bits_per_symbol = 4  # 16-QAM
        self.constellation_type = "qam"  # "qam", "bpsk", "pam"
        self.normalize = True
        self.demapping_method = "app"  # "app", "maxlog"
        
        # Channel parameters
        self.channel_model = "awgn"  # "awgn", "rayleigh", "udp"
        self.equalization_method = "mrc"  # "mrc", "zf", "mmse"
        
        # UDP parameters (for hardware transmission)
        self.udp_tx_host = "localhost"
        self.udp_tx_port = 12345
        self.udp_rx_host = "localhost"
        self.udp_rx_port = 12346
        self.udp_timeout = 10.0
        self.udp_buffer_size = 8192
        
        # System parameters
        self.device = "cuda"
        self.verbose = False


class AWGNChannelConfig(WirelessCommConfig):
    """Configuration for AWGN channel."""
    
    def __init__(self, comm_mode=4):
        super().__init__()
        self.comm_mode = comm_mode
        self.channel_model = "awgn"
        self.equalization_method = "mrc"  # Simple for AWGN
        
        # Optimized parameters for AWGN
        if comm_mode == 3:  # Uncoded
            self.bits_per_symbol = 2  # QPSK for better performance
            self.constellation_type = "qam"
        elif comm_mode == 4:  # Coded
            self.bits_per_symbol = 4  # 16-QAM with coding
            self.constellation_type = "qam"
            self.decoder_type = "SCL"  # Better performance
            self.list_size = 16


class RayleighChannelConfig(WirelessCommConfig):
    """Configuration for Rayleigh fading channel."""
    
    def __init__(self, comm_mode=4):
        super().__init__()
        self.comm_mode = comm_mode
        self.channel_model = "rayleigh"
        self.equalization_method = "mmse"  # Better for fading
        
        # Robust parameters for fading channel
        if comm_mode == 3:  # Uncoded
            self.bits_per_symbol = 2  # QPSK for robustness
            self.constellation_type = "qam"
        elif comm_mode == 4:  # Coded
            self.bits_per_symbol = 2  # Conservative modulation
            self.constellation_type = "qam"
            self.decoder_type = "SCL"
            self.list_size = 32  # Larger list for better performance
            
            # More robust coding
            self.N = 512
            self.k = 256


class UDPChannelConfig(WirelessCommConfig):
    """Configuration for UDP channel (real hardware)."""
    
    def __init__(self, comm_mode=4, tx_host="192.168.1.100", rx_host="192.168.1.101"):
        super().__init__()
        self.comm_mode = comm_mode
        self.channel_model = "udp"
        
        # UDP network configuration
        self.udp_tx_host = tx_host
        self.udp_tx_port = 12345
        self.udp_rx_host = rx_host
        self.udp_rx_port = 12346
        self.udp_timeout = 30.0  # Longer timeout for real hardware
        self.udp_buffer_size = 16384  # Larger buffer
        
        # Hardware-optimized parameters
        self.bits_per_symbol = 2  # Conservative for real RF
        self.constellation_type = "qam"
        self.decoder_type = "SCL"
        self.list_size = 8
        
        # Robust coding for real channel
        if comm_mode == 4:
            self.N = 1024
            self.k = 512


class HighThroughputConfig(WirelessCommConfig):
    """Configuration optimized for high throughput."""
    
    def __init__(self, comm_mode=4):
        super().__init__()
        self.comm_mode = comm_mode
        self.channel_model = "awgn"  # Assume good channel
        
        # High throughput parameters
        self.bits_per_symbol = 6  # 64-QAM
        self.constellation_type = "qam"
        self.decoder_type = "SC"  # Faster than SCL
        
        # Shorter codes for lower latency
        self.N = 256
        self.k = 192  # High rate code


class LowLatencyConfig(WirelessCommConfig):
    """Configuration optimized for low latency."""
    
    def __init__(self, comm_mode=4):
        super().__init__()
        self.comm_mode = comm_mode
        self.channel_model = "awgn"
        
        # Low latency parameters
        self.bits_per_symbol = 2  # QPSK - simple demodulation
        self.constellation_type = "qam"
        self.decoder_type = "SC"  # Faster decoding
        
        # Short codes for lower latency
        self.N = 128
        self.k = 64


class RobustConfig(WirelessCommConfig):
    """Configuration optimized for robustness in poor channel conditions."""
    
    def __init__(self, comm_mode=4):
        super().__init__()
        self.comm_mode = comm_mode
        self.channel_model = "rayleigh"
        
        # Robust parameters
        self.bits_per_symbol = 1  # BPSK - most robust
        self.constellation_type = "bpsk"
        self.decoder_type = "SCL"
        self.list_size = 64  # Large list for better performance
        self.equalization_method = "mmse"
        
        # Low rate code for robustness
        self.N = 1024
        self.k = 256  # Rate 1/4


class AnalogTransmissionConfig(WirelessCommConfig):
    """Configuration for analog transmission modes (1 and 2)."""
    
    def __init__(self, comm_mode=1):
        super().__init__()
        self.comm_mode = comm_mode  # Mode 1 or 2
        
        # Analog transmission doesn't need coding parameters
        self.channel_model = "awgn"  # Simple channel for analog
        
        # Threshold for importance masking (Mode 1)
        self.threshold = 0.5
        
        # Noise parameters can be configured separately
        self.noise_variance = 0.1


def get_config_by_scenario(scenario: str, **kwargs):
    """
    Factory function to get configuration by scenario name.
    
    Args:
        scenario: Scenario name
        **kwargs: Additional parameters
        
    Returns:
        Configuration object
    """
    configs = {
        'awgn': AWGNChannelConfig,
        'rayleigh': RayleighChannelConfig,
        'udp': UDPChannelConfig,
        'high_throughput': HighThroughputConfig,
        'low_latency': LowLatencyConfig,
        'robust': RobustConfig,
        'analog': AnalogTransmissionConfig,
        'default': WirelessCommConfig
    }
    
    if scenario not in configs:
        print(f"Unknown scenario '{scenario}', using default config")
        scenario = 'default'
    
    return configs[scenario](**kwargs)


def print_config_comparison():
    """Print a comparison of different configurations."""
    scenarios = ['awgn', 'rayleigh', 'udp', 'high_throughput', 'low_latency', 'robust']
    
    print("Configuration Comparison:")
    print("=" * 80)
    print(f"{'Scenario':<15} {'Mode':<6} {'Channel':<10} {'Modulation':<12} {'Code Rate':<10} {'Decoder':<8}")
    print("-" * 80)
    
    for scenario in scenarios:
        config = get_config_by_scenario(scenario)
        
        # Calculate code rate
        if hasattr(config, 'k') and hasattr(config, 'N'):
            code_rate = f"{config.k/config.N:.3f}"
        else:
            code_rate = "N/A"
        
        # Get modulation description
        if config.bits_per_symbol == 1:
            modulation = "BPSK"
        else:
            modulation = f"{2**config.bits_per_symbol}-{config.constellation_type.upper()}"
        
        print(f"{scenario:<15} {config.comm_mode:<6} {config.channel_model:<10} "
              f"{modulation:<12} {code_rate:<10} {config.decoder_type:<8}")
    
    print("=" * 80)


if __name__ == "__main__":
    print_config_comparison()
    
    # Example usage
    print("\nExample: Creating AWGN channel configuration")
    awgn_config = get_config_by_scenario('awgn', comm_mode=4)
    print(f"Channel model: {awgn_config.channel_model}")
    print(f"Modulation: {2**awgn_config.bits_per_symbol}-{awgn_config.constellation_type.upper()}")
    print(f"Code rate: {awgn_config.k}/{awgn_config.N} = {awgn_config.k/awgn_config.N:.3f}")
