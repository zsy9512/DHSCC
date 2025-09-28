# DHSCC Separated Architecture

This directory contains the refactored DHSCC system separated into independent sender and receiver components as per the requirements.

## Architecture Overview

The system is divided into three main components:

### 1. Sender (`/sender/`)
- **Independent training** with quantizer integration
- **Direct SNR input** during training
- **Models**: `SenderModel` (integrates `SenderNetwork` + `VibrationVectorQuantizer`)
- **Training**: Three-phase strategy (pretrain, selector/quantizer optimization, final)
- **Output**: Model files and transmission-ready features

### 2. Receiver (`/receiver/`)
- **Independent training** and testing
- **Adaptive processing** based on channel conditions
- **Models**: `ReceiverModel` (integrates `ReceiverNetwork`)
- **Training**: Simulates channel effects for robust training
- **Output**: Classification results from received features

### 3. Channel (`/channel/`)
- **Independent module** handling wireless transmission
- **Modulation/Demodulation** integrated
- **Channel coding/decoding** interfaces for sender/receiver
- **Multiple modes**: Analog (1,2) and Digital (3,4) transmission

## Key Features

- ✅ **Independent Training**: Each component trains separately with its own model files
- ✅ **Quantizer Integration**: Quantizer is part of sender training as required
- ✅ **Direct SNR Input**: SNR provided directly during training (not random generation)
- ✅ **Channel Separation**: Channel+modulation as independent third component
- ✅ **No Code Reuse**: Each component has its own copied utilities and models

## Usage

### Training Sender
```bash
cd sender
python sender_main.py --mode train --communication_mode 4 --snr_min 5.0 --snr_max 15.0
```

### Training Receiver
```bash
cd receiver
python receiver_main.py --mode train --communication_mode 4 --snr_min 5.0 --snr_max 15.0
```

### Testing Sender
```bash
cd sender
python sender_main.py --mode test --model_path sender_checkpoints/training_*/models/sender_best_model.pth
```

### Testing Receiver
```bash
cd receiver
python receiver_main.py --mode test --model_path receiver_checkpoints/training_*/models/receiver_best_model.pth
```

## File Structure

```
separated_dhscc/
├── sender/
│   ├── models/
│   │   ├── sender_model.py      # Integrated sender with quantizer
│   │   ├── sender_network.py    # Feature extraction & encoding
│   │   ├── quantizer.py         # Vector quantization
│   │   └── [supporting VQ files]
│   ├── utils/                   # Copied utilities
│   ├── sender_main.py           # Entry point
│   ├── sender_train.py          # Training logic
│   └── sender_test.py           # Testing logic
├── receiver/
│   ├── models/
│   │   ├── receiver_model.py    # Integrated receiver
│   │   └── receiver_network.py  # Decoding & classification
│   ├── utils/                   # Copied utilities
│   ├── receiver_main.py         # Entry point
│   ├── receiver_train.py        # Training logic
│   └── receiver_test.py         # Testing logic
├── channel/
│   ├── channel_module.py        # Main channel interface
│   ├── wireless_channel.py      # Original channel logic
│   └── wireless_comm/           # Communication protocols
└── shared/
    └── data/                    # Common data utilities
```

## Training Strategy

### Sender Training Phases
1. **Pretrain** (0-80 epochs): Focus on classification and basic feature learning
2. **Selector** (80-140 epochs): Optimize quantizer (equivalent to selector for sender)
3. **Final** (140-200 epochs): Balanced training of all components

### Receiver Training
- **Continuous training** with simulated channel effects
- **Multiple error types**: Gaussian, impulse, fading, quantization, bit errors
- **Robustness optimization** across SNR range

## Model Files

Each component saves independent model files:
- **Sender**: `sender_best_model.pth` (includes quantizer state)
- **Receiver**: `receiver_best_model.pth`
- **Channel**: Configuration-based (no trainable parameters)

## Requirements Compliance

- ✅ **Requirement 1**: Independent training and testing with separate model files
- ✅ **Requirement 2**: Quantizer integrated into sender training
- ✅ **Requirement 3**: Channel+modulation as independent third component
- ✅ **Requirement 4**: Direct SNR input during training
- ✅ **Requirement 5**: New folder structure with copied code (no reuse)

## Data Flow

1. **Training**: Each component trains independently with simulated data
2. **Testing**: 
   - Sender produces transmission-ready features
   - Channel processes features with communication effects
   - Receiver processes received features for classification

This architecture allows for true distributed deployment where sender and receiver can be trained and deployed completely independently.