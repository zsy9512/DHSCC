#!/usr/bin/env python3
"""
Test script for 5G Polar decoding using codeword.npy data with DHSCC standard decoder.

This script loads the polar codewords from codeword.npy, converts them to LLRs,
and performs 5G polar decoding using the standard DHSCC polar decoder to verify functionality.
"""

import torch
import numpy as np
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dhscc_modules'))

from dhscc_modules.dhscc_polar_decoder import PolarSCDecoder, Polar5GDecoder, generate_5g_ranking

def main():
    print("Testing Standard DHSCC Polar Decoding with codeword.npy")

    # Load the codeword data
    codeword_path = 'codeword.npy'
    if not os.path.exists(codeword_path):
        print(f"Error: {codeword_path} not found")
        return

    codeword = np.load(codeword_path)
    print(f"Loaded codeword shape: {codeword.shape}")
    print(f"Codeword dtype: {codeword.dtype}")
    print(f"First codeword first 20 bits: {codeword[0][:20]}")
    print(f"Unique values: {np.unique(codeword)}")

    # Convert binary codeword to LLRs
    # For perfect channel, LLR = +large for 1, -large for 0
    llr_scale = 10.0  # Large value for high SNR
    llr_codeword = (2 * codeword - 1) * llr_scale  # 0 -> -10, 1 -> +10

    print(f"LLR codeword shape: {llr_codeword.shape}")
    print(f"LLR range: [{llr_codeword.min():.1f}, {llr_codeword.max():.1f}]")

    # 使用5G极化码编码器和解码器
    print("\nInitializing 5G Polar5GEncoder and Polar5GDecoder...")
    k = 128  # 信息比特数
    n = 256  # 码字长度

    # 首先创建5G极化码编码器
    from dhscc_modules.dhscc_polar_decoder import Polar5GEncoder, Polar5GDecoder
    encoder_5g = Polar5GEncoder(k, n, channel_type="uplink", verbose=False, dtype=torch.float32)
    
    # 然后创建对应的解码器
    decoder_5g = Polar5GDecoder(encoder_5g, dec_type="SC", dtype=torch.float32)

    # 转为torch tensor
    llr_tensor = torch.from_numpy(llr_codeword).float()
    print(f"Input LLR tensor shape: {llr_tensor.shape}")

    # 执行解码
    print("\nPerforming 5G polar decoding...")
    try:
        with torch.no_grad():
            decoded_bits = decoder_5g(llr_tensor)

        print(f"Decoding successful!")
        print(f"Decoded bits shape: {decoded_bits.shape}")
        print(f"Decoded bits dtype: {decoded_bits.dtype}")

        # 展示每个码字的解码结果
        print("\nDecoded results for each codeword:")
        for i in range(decoded_bits.shape[0]):
            bits = decoded_bits[i].cpu().numpy()
            print(f"Codeword {i}: first 20 bits: {bits[:20]}")
            print(f"Codeword {i}: last 20 bits: {bits[-20:]}")
            print(f"Codeword {i}: bit distribution: {np.bincount(bits.astype(int))}")

        # 验证解码结果是否为二值
        unique_vals = torch.unique(decoded_bits)
        print(f"\nUnique values in decoded bits: {unique_vals}")
        if len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals:
            print("✓ Decoded bits are properly binary")
        else:
            print("⚠ Warning: Decoded bits contain non-binary values")

    except Exception as e:
        print(f"Decoding failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()