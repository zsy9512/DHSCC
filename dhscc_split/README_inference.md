# DHSCC Split Inference Scripts

This directory contains separate inference scripts for deploying the DHSCC (Deep Hierarchical Split Communication Classification) system on two machines connected via wireless channel.

## Overview

The DHSCC system is split into two components:
- **Sender**: Processes vibration signals, extracts features, quantizes them, and prepares for wireless transmission
- **Receiver**: Receives wireless transmissions, decodes features, and performs vibration signal classification

## Directory Structure

```
dhscc_split/
├── sender/
│   └── sender_inference.py      # Sender inference script
├── receiver/
│   └── receiver_inference.py    # Receiver inference script
└── README_inference.md          # This documentation
```

## Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Trained model checkpoint (automatically detected from `../../checkpoints/end_to_end_training_*/final/best_end_to_end_model.pth`)
- Vibration dataset in `../../dataset/vibcls/`

## Usage

### 1. Sender Machine Setup

Run the sender inference script to process vibration data and prepare for transmission:

```bash
cd sender
python sender_inference.py [options]
```

#### Options:
- `--data_dir`: Directory containing vibration data (default: `../dataset/vibcls/vibrodata`)
- `--test_list`: Path to test file list (default: `../dataset/vibcls/test.txt`)
- `--class_map`: Path to class mapping file (default: `../dataset/vibcls/clsmap.txt`)
- `--checkpoint`: Path to trained model checkpoint (default: auto-detected)
- `--batch_size`: Number of samples per batch (default: 5)
- `--max_samples`: Maximum samples to process (default: all)
- `--channel_snr`: Channel SNR for simulation (default: 10.0 dB)
- `--output_dir`: Directory to save transmission data (default: `./sender_output`)

#### Example:
```bash
cd sender
# Process 5 samples at a time, save to custom output directory
python sender_inference.py --batch_size 5 --max_samples 50 --output_dir ./transmission_data
```

### 2. Wireless Transmission

After running the sender script, the processed data will be saved in the output directory (e.g., `./transmission_data/`) as `.pt` files. These files contain:
- Quantized features
- Importance weights
- Feature shape information
- Original labels (for evaluation)

Transfer these files to the receiver machine via your wireless communication channel.

### 3. Receiver Machine Setup

Run the receiver inference script to process received data and perform classification:

```bash
cd receiver
python receiver_inference.py [options]
```

#### Options:
- `--checkpoint`: Path to trained model checkpoint (default: auto-detected)
- `--input_dir`: Directory containing transmission data from sender (default: `./sender_output`)
- `--batch_pattern`: Pattern to match transmission batch files (default: `batch_*.pt`)
- `--channel_snr`: Channel SNR for simulation (default: 10.0 dB)
- `--channel_model`: Channel model type (default: `rayleigh`)
- `--output_file`: File to save results (default: `./receiver_results.txt`)
- `--save_predictions`: Save detailed predictions to file

#### Example:
```bash
cd receiver
# Process received data and save detailed results
python receiver_inference.py --input_dir ./received_data --save_predictions
```

## Output

### Sender Output
- Transmission batch files (`.pt`) in the output directory
- Each batch contains quantized features ready for wireless transmission

### Receiver Output
- Classification results saved to text file
- Overall accuracy and per-batch statistics
- Optional detailed predictions with confidence scores

## Wireless Channel Integration

The scripts currently simulate wireless transmission using the `ChannelModem` class. For real wireless deployment:

1. **Sender**: Replace the file saving with actual wireless transmission
2. **Receiver**: Replace file loading with wireless reception
3. **ChannelModem**: Implement real wireless communication protocols

## Performance Notes

- Batch size of 5 samples balances processing efficiency with memory usage
- Channel SNR affects classification accuracy - higher SNR improves performance
- The system uses Rayleigh fading channel model by default
- Processing time depends on hardware and batch size

## Troubleshooting

### Common Issues:

1. **Model checkpoint not found**: Ensure training has completed and checkpoint exists
2. **Dataset files not found**: Verify paths to `../../dataset/vibcls/` directory
3. **CUDA out of memory**: Reduce batch size or use CPU with `--batch_size 1`
4. **Wireless interface warnings**: Expected in simulation mode, implement real wireless for deployment

### Error Messages:

- `"VibrationVectorQuantizer initialization error"`: Check quantizer parameters
- `"WavDataset file not found"`: Verify dataset paths and file permissions
- `"Channel transmission failed"`: Check channel parameters and SNR settings

## Architecture Details

The split architecture allows:
- Independent scaling of sender/receiver hardware
- Bandwidth-efficient wireless transmission
- Robust performance under varying channel conditions
- Modular deployment on edge devices

For more details on the DHSCC system, see the main project documentation.