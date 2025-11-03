"""
DQPSK Modem Module

This module implements DQPSK (Differential Quadrature Phase Shift Keying) modulation
and demodulation functions, ported from dqpsk_system.py.
"""

import numpy as np
from scipy import signal
import crcmod

# QPSK constellation (Gray coded) - normalized
CONSTELLATION = np.array([
    1 + 1j,  # 00
    -1 + 1j, # 01
    1 - 1j,  # 10
    -1 - 1j  # 11
], dtype=np.complex64) / np.sqrt(2)

# Differential encoding reference symbol
DIFF_REF_SYMBOL = 1 + 0j

# Frame structure constants (copied from dqpsk_system.py)
PSS_LEN = 32
SSS_LEN = 32
RS_LEN = 64
PREAMBLE_LEN = PSS_LEN + SSS_LEN + RS_LEN
DATA_BITS = 1280  # 640 symbols (2 bits/symbol)
DATA_SYMBOLS = DATA_BITS // 2

def design_rrc_filter(sps=2, roll_off=0.35, num_symbols=10):
    """
    Design Root Raised Cosine (RRC) filter for pulse shaping.
    
    Args:
        sps (int): Samples per symbol
        roll_off (float): Roll-off factor
        num_symbols (int): Filter length in symbols
        
    Returns:
        np.ndarray: RRC filter coefficients
    """
    taps = num_symbols * sps
    t = np.arange(-num_symbols/2, num_symbols/2, 1/sps)
    rrc = []
    
    for ti in t:
        if abs(ti) < 1e-10:
            val = 1 + roll_off * (4/np.pi - 1)
        elif abs(abs(ti) - 1/(4*roll_off)) < 1e-10:
            val = (roll_off/np.sqrt(2)) * ((1+2/np.pi)*np.sin(np.pi/(4*roll_off)) + 
                                         (1-2/np.pi)*np.cos(np.pi/(4*roll_off)))
        else:
            pi_t = np.pi * ti
            sin_t = np.sin(pi_t * (1 - roll_off))
            cos_t = np.cos(pi_t * (1 + roll_off))
            denom = pi_t * (1 - (4 * roll_off * ti)**2)
            val = (sin_t + 4 * roll_off * ti * cos_t) / denom
        rrc.append(val)
    
    rrc = np.array(rrc)
    return rrc / np.sqrt(np.sum(rrc**2))

def bits_to_symbols(bits):
    """
    Convert bits to QPSK symbols using Gray coding.
    
    Args:
        bits (np.ndarray): Input bits (length must be even)
        
    Returns:
        np.ndarray: QPSK symbols
    """
    if len(bits) % 2 != 0:
        raise ValueError("Number of bits must be even for QPSK modulation")
    
    symbols = []
    for i in range(0, len(bits), 2):
        bit1, bit2 = bits[i], bits[i+1]
        if (bit1, bit2) == (0, 0):
            idx = 0
        elif (bit1, bit2) == (0, 1):
            idx = 1
        elif (bit1, bit2) == (1, 0):
            idx = 2
        else:
            idx = 3
        symbols.append(CONSTELLATION[idx])
    
    return np.array(symbols, dtype=np.complex64)

def symbols_to_bits(symbols):
    """
    Convert QPSK symbols to bits using Gray coding.
    
    Args:
        symbols (np.ndarray): Input QPSK symbols
        
    Returns:
        np.ndarray: Output bits
    """
    bits = []
    for symbol in symbols:
        # Find nearest constellation point
        distances = np.abs(symbol - CONSTELLATION)
        idx = np.argmin(distances)
        
        # Convert to bits (Gray coding)
        if idx == 0:
            bits.extend([0, 0])
        elif idx == 1:
            bits.extend([0, 1])
        elif idx == 2:
            bits.extend([1, 0])
        else:
            bits.extend([1, 1])
    
    return np.array(bits)

def differential_encode(symbols):
    """
    Apply differential encoding to QPSK symbols.
    
    Args:
        symbols (np.ndarray): Input QPSK symbols
        
    Returns:
        np.ndarray: Differentially encoded symbols
    """
    if len(symbols) == 0:
        return np.array([])
    
    encoded = [symbols[0] * DIFF_REF_SYMBOL]
    for i in range(1, len(symbols)):
        encoded.append(symbols[i] * encoded[i-1])
    
    return np.array(encoded, dtype=np.complex64)

def differential_decode(symbols):
    """
    Apply differential decoding to QPSK symbols.
    
    Args:
        symbols (np.ndarray): Input differentially encoded symbols
        
    Returns:
        np.ndarray: Differentially decoded symbols
    """
    if len(symbols) == 0:
        return np.array([])
    
    decoded = []
    # First symbol uses reference symbol
    decoded.append(symbols[0] * np.conj(DIFF_REF_SYMBOL))
    
    # Subsequent symbols use differential decoding
    for i in range(1, len(symbols)):
        decoded.append(symbols[i] * np.conj(symbols[i-1]))
    
    return np.array(decoded, dtype=np.complex64)

# Copied functions from dqpsk_system.py for frame generation
def _generate_pss():
    n = np.arange(PSS_LEN - 1)
    u = 25
    zc_sequence = np.exp(-1j * np.pi * u * n * (n + 1) / (PSS_LEN - 1))
    return np.concatenate([zc_sequence, [0]])

def _generate_sss():
    def _m_sequence(degree, taps):
        length = 2**degree - 1
        reg = np.ones(degree, dtype=int)
        mseq = []
        for _ in range(length * 2):
            mseq.append(reg[-1])
            feedback = 0
            for tap in taps:
                feedback ^= reg[tap-1]
            reg = np.roll(reg, 1)
            reg[0] = feedback
        return np.array(mseq) * 2 - 1  # Map to -1/+1
    
    mseq1 = _m_sequence(8, [8, 4, 3, 2])
    mseq2 = _m_sequence(8, [8, 6, 5, 1])
    sss = np.concatenate([
        mseq1[:16] + 1j * mseq1[16:32],
        mseq2[:16] + 1j * mseq2[16:32]
    ])
    return sss / np.sqrt(np.mean(np.abs(sss)**2))

def _generate_rs():
    base_seq = np.array([1, 1, -1, 1, -1, -1, 1, -1, 
                         1, -1, -1, 1, -1, 1, 1, -1], dtype=np.complex64)
    return np.tile(base_seq, 4)

def encode_frame_index_hamming_parity(frame_index: int) -> np.ndarray:
    """编码帧序号：3比特输入 -> 8比特输出 (7位汉明码 + 1位偶校验)"""
    if not (0 <= frame_index <= 7):
        raise ValueError("Frame index must be between 0 and 7")
    
    # 3比特帧序号，左移一位补0作为4比特输入
    data_bits = np.array([0, (frame_index >> 2) & 1, (frame_index >> 1) & 1, frame_index & 1], dtype=int)
    
    # 汉明(7,4)编码：生成校验位
    p1 = data_bits[0] ^ data_bits[1] ^ data_bits[3]  # 位置1,3,5,7
    p2 = data_bits[0] ^ data_bits[2] ^ data_bits[3]  # 位置2,3,6,7
    p4 = data_bits[1] ^ data_bits[2] ^ data_bits[3]  # 位置5,6,7
    
    # 7位汉明码：p1 p2 d1 p4 d2 d3 d4
    hamming_bits = np.array([p1, p2, data_bits[0], p4, data_bits[1], data_bits[2], data_bits[3]], dtype=int)
    
    # 偶校验位（对7位汉明码）
    parity = np.sum(hamming_bits) % 2
    
    # 返回8比特：7位汉明码 + 1位偶校验
    return np.concatenate([hamming_bits, [parity]])

def generate_frame(data_bits, frame_counter=0):
    """生成包含前导和数据的DQPSK帧，返回符号"""
    if len(data_bits) != DATA_BITS:
        raise ValueError(f"Data bits must be exactly {DATA_BITS}")
    
    # 转换为符号
    data_symbols = bits_to_symbols(data_bits)
    # 差分编码
    encoded_data_symbols = differential_encode(data_symbols)
    
    # 生成帧序号并编码为比特
    frame_index_bits = encode_frame_index_hamming_parity(frame_counter)
    # 转换为符号
    frame_index_symbols = bits_to_symbols(frame_index_bits)
    # 差分编码
    encoded_frame_index_symbols = differential_encode(frame_index_symbols)
    
    # 构建完整帧：前导 + 帧序号 + 数据
    frame = np.concatenate([_generate_pss(), _generate_sss(), _generate_rs(), encoded_frame_index_symbols, encoded_data_symbols])
    
    return frame

def modulate_bits(bits, sps=2, roll_off=0.35, frame_counter=0):
    """
    Complete DQPSK modulation pipeline: bits -> frame generation -> pulse shaping.
    
    Args:
        bits (np.ndarray): Input bits (must be DATA_BITS length)
        sps (int): Samples per symbol
        roll_off (float): RRC roll-off factor
        frame_counter (int): Frame counter for frame index encoding
        
    Returns:
        np.ndarray: Modulated signal samples
    """
    # Generate frame (symbols)
    frame_symbols = generate_frame(bits, frame_counter)
    #print(f"Generated frame with {len(frame_symbols)} symbols")
    # Pulse shaping
    rrc_filter = design_rrc_filter(sps=sps, roll_off=roll_off)
    upsampled = np.zeros(len(frame_symbols) * sps, dtype=np.complex64)
    upsampled[::sps] = frame_symbols
    
    tx_signal = np.convolve(upsampled, rrc_filter, mode='full')
    
    # rx_signal= np.convolve(tx_signal, rrc_filter, mode='full')
    # delay = len(rrc_filter)
    # # 下采样并补偿
    #downsampled_corrected = rx_signal[delay::sps]

    # print(f"Pulse-shaped signal length: {len(tx_signal)} samples, rx_signal length: {len(rx_signal)} samples")
    # print(f"downsampled_corrected length: {len(downsampled_corrected)} samples")
    return tx_signal.astype(np.complex64)

def demodulate_signal(rx_signal, sps=2, roll_off=0.35):
    """
    Complete DQPSK demodulation pipeline: matched filtering -> diff decoding -> symbol detection -> bits.
    
    Args:
        rx_signal (np.ndarray): Received signal samples
        sps (int): Samples per symbol
        roll_off (float): RRC roll-off factor
        
    Returns:
        np.ndarray: Demodulated bits
    """
    # Matched filtering
    rrc_filter = design_rrc_filter(sps=sps, roll_off=roll_off)
    filtered = np.convolve(rx_signal, rrc_filter, mode='full')
    
    # Compensate for RRC filter delay
    rrc_delay = (len(rrc_filter) - 1) // 2
    filtered = filtered[rrc_delay:]
    
    # Downsampling with proper timing
    symbols = filtered[::sps]
    
    # Differential decoding
    decoded_symbols = differential_decode(symbols)
    
    # Symbol to bits conversion
    bits = symbols_to_bits(decoded_symbols)
    
    return bits