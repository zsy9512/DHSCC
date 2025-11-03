# DQPSK Modem for receiver
import numpy as np
from scipy.optimize import minimize_scalar

class DQPSKModem:
    """DQPSK解调器类，包含同步和解调方法"""

    def __init__(self, samp_rate=1e6, sps=2, roll_off=0.35):
        self.samp_rate = samp_rate
        self.sps = sps
        self.roll_off = roll_off
        self.symbol_rate = samp_rate / sps
        self.Ts = 1.0 / self.symbol_rate

        # Frame structure
        self.pss_len = 32
        self.sss_len = 32
        self.rs_len = 64
        self.preamble_len = self.pss_len + self.sss_len + self.rs_len

        # Synchronization sequences
        self.pss = self._generate_pss()
        self.sss = self._generate_sss()
        self.rs = self._generate_rs()
        self.frame_header = np.concatenate([self.pss, self.sss, self.rs])
        self.rrc_filter = self._design_rrc_filter()
        
        # Differential encoding reference symbol
        self.diff_ref_symbol = 1 + 0j

    def _generate_pss(self):
        """生成PSS序列"""
        n = np.arange(self.pss_len - 1)
        u = 25
        zc_sequence = np.exp(-1j * np.pi * u * n * (n + 1) / (self.pss_len - 1))
        return np.concatenate([zc_sequence, [0]])

    def _generate_sss(self):
        """生成SSS序列"""
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

    def _generate_rs(self):
        """生成RS序列"""
        base_seq = np.array([1, 1, -1, 1, -1, -1, 1, -1, 
                           1, -1, -1, 1, -1, 1, 1, -1], dtype=np.complex64)
        return np.tile(base_seq, 4)

    def _design_rrc_filter(self, num_symbols=10):
        """设计RRC滤波器"""
        taps = num_symbols * self.sps
        t = np.arange(-num_symbols/2, num_symbols/2, 1/self.sps)
        rrc = []
        for ti in t:
            if abs(ti) < 1e-10:
                val = 1 + self.roll_off * (4/np.pi - 1)
            elif abs(abs(ti) - 1/(4*self.roll_off)) < 1e-10:
                val = (self.roll_off/np.sqrt(2)) * ((1+2/np.pi)*np.sin(np.pi/(4*self.roll_off)) + 
                                                   (1-2/np.pi)*np.cos(np.pi/(4*self.roll_off)))
            else:
                pi_t = np.pi * ti
                sin_t = np.sin(pi_t * (1 - self.roll_off))
                cos_t = np.cos(pi_t * (1 + self.roll_off))
                denom = pi_t * (1 - (4 * self.roll_off * ti)**2)
                val = (sin_t + 4 * self.roll_off * ti * cos_t) / denom
            rrc.append(val)
        rrc = np.array(rrc)
        return rrc / np.sqrt(np.sum(rrc**2))

    def matched_filter(self, rx_signal):
        """匹配滤波"""
        rx_filtered = np.convolve(rx_signal, self.rrc_filter, mode='full')
        rx_symbols = rx_filtered[::self.sps]
        return rx_symbols

    def _enhanced_pss_sync(self, rx_symbols):
        """增强型PSS符号定时同步，带插值优化"""
        corr = np.correlate(rx_symbols, self.frame_header, mode='valid')
        corr_power = np.abs(corr)**2
        max_idx = np.argmax(corr_power)
        if max_idx > 0 and max_idx < len(corr_power) - 1:
            y1, y2, y3 = corr_power[max_idx-1], corr_power[max_idx], corr_power[max_idx+1]
            if y1 - 2*y2 + y3 != 0:
                delta = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3)
                timing_offset = max_idx + delta
            else:
                timing_offset = max_idx
        else:
            timing_offset = max_idx
        timing_offset = int(np.round(timing_offset))
        return timing_offset

    def _enhanced_sss_sync(self, rx_symbols, timing_offset):
        """SSS粗频同步"""
        sss_start = timing_offset + self.pss_len
        sss_end = sss_start + self.sss_len
        if sss_end > len(rx_symbols):
            return 0.0
        rx_sss = rx_symbols[sss_start:sss_end]
        # 方法1: 相位差法
        tx_phase = np.angle(self.sss)
        tx_phase_diff = np.diff(tx_phase)
        rx_phase = np.angle(rx_sss)
        rx_phase_diff = np.diff(rx_phase)
        freq_phase_diff = rx_phase_diff - tx_phase_diff
        freq_phase_diff = np.angle(np.exp(1j * freq_phase_diff))
        freq_est1 = np.mean(freq_phase_diff) / (2 * np.pi * self.Ts)
        # 方法2: 基于相关的频率估计
        freq_search = np.linspace(-1000/2, 1000/2, 500)
        corr_values = []
        for f_test in freq_search:
            n = np.arange(len(rx_sss))
            rx_corrected = rx_sss * np.exp(-1j * 2 * np.pi * f_test * n * self.Ts)
            corr = np.abs(np.sum(rx_corrected * np.conj(self.sss)))
            corr_values.append(corr)
        max_corr_idx = np.argmax(corr_values)
        freq_est2 = freq_search[max_corr_idx]
        coarse_freq = 0.3 * freq_est1 + 0.7 * freq_est2
        return coarse_freq

    def _enhanced_rs_sync(self, rx_symbols, timing_offset, coarse_freq):
        """RS细频同步"""
        n = np.arange(len(rx_symbols))
        rx_symbols_corrected = rx_symbols * np.exp(-1j * 2 * np.pi * coarse_freq * n * self.Ts)
        rs_start = timing_offset + self.pss_len + self.sss_len
        rs_end = rs_start + self.rs_len
        if rs_end > len(rx_symbols_corrected):
            return 0.0
        rx_rs = rx_symbols_corrected[rs_start:rs_end]
        group_size = 16
        num_groups = self.rs_len // group_size
        phase_diffs = []
        for i in range(num_groups - 1):
            group1 = rx_rs[i*group_size:(i+1)*group_size]
            group2 = rx_rs[(i+1)*group_size:(i+2)*group_size]
            cross_corr = np.sum(group1 * np.conj(group2))
            phase_diff = np.angle(cross_corr)
            phase_diffs.append(phase_diff)
        avg_phase_diff = np.mean(phase_diffs)
        fine_freq1 = avg_phase_diff / (2 * np.pi * group_size * self.Ts)
        def freq_cost_function(f_test):
            n_rs = np.arange(len(rx_rs))
            rx_test = rx_rs * np.exp(-1j * 2 * np.pi * f_test * n_rs * self.Ts)
            corr = np.abs(np.sum(rx_test * np.conj(self.rs)))
            return -corr
        search_range = 1000
        result = minimize_scalar(freq_cost_function, 
                               bounds=(fine_freq1 - search_range, fine_freq1 + search_range),
                               method='bounded')
        fine_freq2 = result.x
        fine_freq = 0.6 * fine_freq1 + 0.4 * fine_freq2
        return fine_freq

    def _init_costas_loop(self, loop_bw=0.01):
        """初始化Costas环用于相位同步"""
        class CostasLoop:
            def __init__(self, loop_bw):
                self.loop_bw = loop_bw
                self.phase = 0.0
                self.freq = 0.0
                self.alpha = 4 * loop_bw
                self.beta = 4 * loop_bw**2 / 2
                
            def process(self, symbols):
                out = []
                for sym in symbols:
                    # 相位旋转
                    rotated = sym * np.exp(-1j * self.phase)
                    out.append(rotated)
                    
                    # 相位误差检测（QPSK）
                    if rotated.real > 0:
                        pd = rotated.real * rotated.imag  # 简化的鉴相器
                    else:
                        pd = -rotated.real * rotated.imag
                        
                    # 环路滤波
                    self.freq += self.beta * pd
                    self.phase += self.alpha * pd + self.freq
                    
                    # 相位折叠到[-π, π]
                    self.phase = (self.phase + np.pi) % (2 * np.pi) - np.pi
                return np.array(out)
        
        return CostasLoop(loop_bw)

    def frequency_correction(self, rx_symbols, timing_offset):
        """频率校正"""
        # PSS同步
        coarse_freq = self._enhanced_sss_sync(rx_symbols, timing_offset)
        fine_freq = self._enhanced_rs_sync(rx_symbols, timing_offset, coarse_freq)
        total_freq = coarse_freq + fine_freq
        
        n = np.arange(len(rx_symbols))
        rx_symbols_corr = rx_symbols * np.exp(-1j * 2 * np.pi * total_freq * n * self.Ts)
        return rx_symbols_corr

    def phase_correction(self, data_symbols):
        """相位校正"""
        costas = self._init_costas_loop(loop_bw=0.001)
        synced = costas.process(data_symbols)
        return synced

    def differential_decode(self, symbols):
        """差分解码"""
        if len(symbols) == 0:
            return np.array([])
        
        decoded = []
        # 第一个符号使用参考符号解码
        decoded.append(symbols[0] * np.conj(self.diff_ref_symbol))
        
        # 后续符号进行差分解码
        for i in range(1, len(symbols)):
            decoded.append(symbols[i] * np.conj(symbols[i-1]))
        
        return np.array(decoded, dtype=np.complex64)

    def calculate_snr(self, symbols):
        """计算SNR"""
        if len(symbols) == 0:
            return None
        
        # 计算信号功率
        signal_power = np.mean(np.abs(symbols)**2)
        
        # 估计噪声功率：使用最小噪声下限 0.001 防止负值或过大SNR误差
        estimated_noise = 0.001

        # 计算 SNR（线性 -> dB）
        noise_power = max(estimated_noise, 1e-12)
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def decode_frame_index_hamming_parity(self, bits):
        """解码帧序号：8比特输入 -> (帧序号, 校验是否通过)"""
        if len(bits) != 8:
            return -1, False
        
        hamming_bits = bits[:7]
        parity = bits[7]
        
        # 校验偶校验位
        computed_parity = np.sum(hamming_bits) % 2
        if computed_parity != parity:
            return -1, False  # 校验失败
        
        # 汉明(7,4)解码：计算综合校验位
        s1 = hamming_bits[0] ^ hamming_bits[2] ^ hamming_bits[4] ^ hamming_bits[6]
        s2 = hamming_bits[1] ^ hamming_bits[2] ^ hamming_bits[5] ^ hamming_bits[6]
        s4 = hamming_bits[3] ^ hamming_bits[4] ^ hamming_bits[5] ^ hamming_bits[6]
        syndrome = (s4 << 2) | (s2 << 1) | s1
        
        # 纠错（如果有1位错误）
        if syndrome > 0 and syndrome <= 7:
            hamming_bits[syndrome - 1] ^= 1  # 纠错
        
        # 提取数据位：d1 d2 d3 d4 -> 帧序号
        # 汉明码结构：p1 p2 d1 p4 d2 d3 d4
        # 数据位：d1(位置2), d2(位置4), d3(位置5), d4(位置6)
        d1, d2, d3, d4 = hamming_bits[2], hamming_bits[4], hamming_bits[5], hamming_bits[6]
        frame_index = (d2 << 2) | (d3 << 1) | d4  # d1是补0位，忽略
        
        # 校验是否在0-7范围内
        valid = 0 <= frame_index <= 7
        return frame_index if valid else -1, valid

    def decode_frame_index(self, demod_symbols):
        """从解调符号中解码帧序号"""
        try:
            # 根据帧结构：前导128符号后是4个帧序号符号
            frame_index_symbols = demod_symbols[self.preamble_len:self.preamble_len+4]
            # 符号到比特转换（Gray编码）
            ideal_points = np.array([1+1j, -1+1j, 1-1j, -1-1j], dtype=np.complex64) / np.sqrt(2)
            bits = []
            for sym in frame_index_symbols:
                distances = np.abs(sym - ideal_points)
                idx = np.argmin(distances)
                # 与发送端_bits_to_symbols匹配的比特映射
                if idx == 0:
                    bits.extend([0, 0])  # 1+1j -> 00
                elif idx == 1:
                    bits.extend([0, 1])  # -1+1j -> 01
                elif idx == 2:
                    bits.extend([1, 0])  # 1-1j -> 10
                else:
                    bits.extend([1, 1])  # -1-1j -> 11
            # 汉明解码
            frame_index, valid = self.decode_frame_index_hamming_parity(np.array(bits))
            return frame_index if valid else None
        except Exception as e:
            print(f"帧序号解码错误: {e}")
            return None

    def _symbols_to_bits(self, symbols):
        """符号到比特转换"""
        bits = []
        for symbol in symbols:
            # 定义理想星座点
            ideal_points = np.array([1+1j, -1+1j, 1-1j, -1-1j], dtype=np.complex64) / np.sqrt(2)
            
            # 找到最近的星座点
            distances = np.abs(symbol - ideal_points)
            idx = np.argmin(distances)
            
            # 转换为比特（Gray编码）
            if idx == 0:
                bits.extend([0, 0])  # 1+1j -> 00
            elif idx == 1:
                bits.extend([0, 1])  # -1+1j -> 01
            elif idx == 2:
                bits.extend([1, 0])  # 1-1j -> 10
            else:
                bits.extend([1, 1])  # -1-1j -> 11
        return np.array(bits)