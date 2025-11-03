#!/usr/bin/env python3
"""
DHSCC处理程序 - 信号处理和GUI显示
从接收程序接收数据，进行DQPSK解调、帧同步、极化码解码、去量化、神经网络分类
"""

import numpy as np
import threading
import queue
import time
import argparse
import sys
import os
import multiprocessing
from multiprocessing.managers import BaseManager
import torch
from dhscc_modules.dhscc_utils import binary_to_indices
from receiver.dqpsk_demodem import DQPSKModem
import socket
import pickle

# DHSCC receiver imports
try:
    from dhscc_receiver import DHSCC_Receiver
    DHSCC_AVAILABLE = True
    print("DHSCC receiver module loaded successfully")
except ImportError as e:
    DHSCC_AVAILABLE = False
    print(f"⚠ DHSCC receiver not available: {e}")

# PyQt5 GUI imports
try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
    from PyQt5.QtCore import QTimer, pyqtSignal, QObject, QThread
    import pyqtgraph as pg
    PYQT_AVAILABLE = True
    print("PyQt5 GUI模块加载成功")
except ImportError as e:
    PYQT_AVAILABLE = False
    print(f"⚠ PyQt5不可用: {e}，将使用matplotlib备用方案")

class DHSCCMonitor(QMainWindow):
    """DHSCC信号监控界面"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('DHSCC Processing Monitor')
        self.setGeometry(100, 100, 1200, 600)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QHBoxLayout(central_widget)

        # 创建左侧面板（星座图）
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 星座图标题
        constellation_title = QLabel('DQPSK Constellation')
        constellation_title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 5px;")
        left_layout.addWidget(constellation_title)

        # 创建星座图
        self.constellation_plot = pg.PlotWidget()
        self.constellation_plot.setBackground('w')
        self.constellation_plot.showGrid(x=True, y=True, alpha=0.3)
        self.constellation_plot.setLabel('left', 'Quadrature')
        self.constellation_plot.setLabel('bottom', 'In-phase')
        self.constellation_plot.setXRange(-1.25, 1.25)
        self.constellation_plot.setYRange(-1.25, 1.25)
        self.constellation_plot.setAspectLocked(True)
        # 创建星座图散点
        self.constellation_scatter = pg.ScatterPlotItem(
            size=3,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(0, 100, 255, 120)
        )
        self.constellation_plot.addItem(self.constellation_scatter)
        left_layout.addWidget(self.constellation_plot)

        # SNR显示
        self.snr_label = QLabel('SNR: -- dB')
        self.snr_label.setStyleSheet("font-size: 12px; margin: 5px;")
        left_layout.addWidget(self.snr_label)

        # 创建右侧面板（时域波形）
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 时域波形标题
        time_title = QLabel('Time Domain Signal')
        time_title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 5px;")
        right_layout.addWidget(time_title)

        # 创建时域波形图
        self.time_plot = pg.PlotWidget()
        self.time_plot.setBackground('w')
        self.time_plot.showGrid(x=True, y=True, alpha=0.3)
        self.time_plot.setLabel('left', 'Amplitude')
        self.time_plot.setLabel('bottom', 'Sample')
        self.time_plot.setXRange(0, 2500)
        self.time_plot.setYRange(-1.5, 1.5)

        # 创建时域波形曲线
        self.time_curve = self.time_plot.plot(pen=pg.mkPen('b', width=2))
        right_layout.addWidget(self.time_plot)

        # 频谱图标题
        spectrum_title = QLabel('Frequency Spectrum')
        spectrum_title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 5px;")
        right_layout.addWidget(spectrum_title)

        # 创建频谱图
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setBackground('w')
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spectrum_plot.setLabel('left', 'Power (dB)')
        self.spectrum_plot.setLabel('bottom', 'Frequency (MHz)')

        # 创建频谱曲线
        self.spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen('r', width=2))
        right_layout.addWidget(self.spectrum_plot)

        # 添加面板到主布局
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 1)

        # 初始化数据缓冲区
        self.time_buffer = np.zeros(2500, dtype=np.complex64)
        self.buffer_index = 0

        # 定时器用于定期更新显示
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(100)  # 100ms更新一次

    def update_constellation(self, points):
        """更新星座图"""
        if points is not None and len(points) > 0:
            # points 直接是复数数组
            real_parts = np.real(points)
            imag_parts = np.imag(points)

            self.constellation_scatter.setData(real_parts, imag_parts)

    def update_time_signal(self, samples):
        """更新时域信号"""
        if samples is not None and len(samples) > 0:
            # 更新循环缓冲区
            new_samples = np.array(samples)
            buffer_size = len(self.time_buffer)
            new_length = len(new_samples)

            if new_length <= buffer_size:
                # 新数据可以完全放入缓冲区
                if self.buffer_index + new_length <= buffer_size:
                    self.time_buffer[self.buffer_index:self.buffer_index + new_length] = new_samples
                    self.buffer_index += new_length
                else:
                    # 需要环绕
                    first_part = buffer_size - self.buffer_index
                    self.time_buffer[self.buffer_index:] = new_samples[:first_part]
                    remaining = new_length - first_part
                    self.time_buffer[:remaining] = new_samples[first_part:]
                    self.buffer_index = remaining
            else:
                # 新数据比缓冲区大，直接复制最后的部分
                self.time_buffer = new_samples[-buffer_size:]
                self.buffer_index = 0

            # 更新显示
            x_data = np.arange(buffer_size)
            self.time_curve.setData(x_data, self.time_buffer.real)

    def update_spectrum(self, samples):
        """更新频谱图"""
        if samples is None:
            return

        try:
            # 支持传入统一 payload
            if isinstance(samples, dict) and samples.get('type') == 'gui_update':
                payload = samples
                samples = payload.get('spectrum_samples')
                samp_rate = payload.get('samp_rate')
                center_freq = payload.get('center_freq')
            else:
                samp_rate = getattr(self, 'samp_rate', None)
                center_freq = getattr(self, 'center_freq', None)

            if samples is None or len(samples) == 0:
                return

            n = len(samples)
            # 计算 FFT 并移频，使 0 频在中心
            fft_data = np.fft.fftshift(np.fft.fft(samples, n))
            freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / (samp_rate if samp_rate else 1.0)))
            power = 20 * np.log10(np.abs(fft_data) + 1e-10)

            # 横轴转换为 MHz，若提供 center_freq 则为绝对频率
            if center_freq is not None:
                freqs_hz = freqs + center_freq
                freqs_mhz = freqs_hz / 1e6
            else:
                freqs_mhz = freqs / 1e6

            self.spectrum_curve.setData(freqs_mhz, power)

            # 自动设定 x 轴范围为采样带宽
            try:
                if samp_rate is not None:
                    bw_mhz = (samp_rate / 1e6) / 2.0
                    if center_freq is not None:
                        self.spectrum_plot.setXRange((center_freq / 1e6) - bw_mhz, (center_freq / 1e6) + bw_mhz)
                    else:
                        self.spectrum_plot.setXRange(-bw_mhz, bw_mhz)
            except Exception:
                pass
        except Exception as e:
            print(f"频谱更新错误: {e}")

    def update_snr(self, snr):
        """更新SNR显示"""
        if snr is not None:
            self.snr_label.setText(f'SNR: {snr:.1f} dB')
        else:
            self.snr_label.setText('SNR: -- dB')

    def update_display(self):
        """定期更新显示（占位符）"""
        pass


class GUIManager(QObject):
    """GUI管理器，处理多线程通信"""

    # 定义信号
    update_constellation_signal = pyqtSignal(object)
    update_time_signal = pyqtSignal(object)
    update_spectrum_signal = pyqtSignal(object)
    update_snr_signal = pyqtSignal(float)

    def __init__(self, gui_queue):
        super().__init__()
        self.gui_queue = gui_queue
        self.running = True

        # 连接信号到槽
        self.update_constellation_signal.connect(self._update_constellation_slot)
        self.update_time_signal.connect(self._update_time_slot)
        self.update_spectrum_signal.connect(self._update_spectrum_slot)
        self.update_snr_signal.connect(self._update_snr_slot)

    def _update_constellation_slot(self, points):
        """更新星座图槽函数"""
        if hasattr(self, 'monitor'):
            self.monitor.update_constellation(points)

    def _update_time_slot(self, samples):
        """更新时域信号槽函数"""
        if hasattr(self, 'monitor'):
            self.monitor.update_time_signal(samples)

    def _update_spectrum_slot(self, samples):
        """更新频谱槽函数"""
        if hasattr(self, 'monitor'):
            # 如果传入的是统一 GUI 负载，直接传递 payload，monitor 会解析
            if isinstance(samples, dict) and samples.get('type') == 'gui_update':
                payload = samples
                # 将采样率和中心频率传给 monitor
                self.monitor.samp_rate = payload.get('samp_rate')
                self.monitor.center_freq = payload.get('center_freq')
                self.monitor.update_spectrum(payload)
            else:
                self.monitor.update_spectrum(samples)

    def _update_snr_slot(self, snr):
        """更新SNR槽函数"""
        if hasattr(self, 'monitor'):
            self.monitor.update_snr(snr)

    def process_gui_updates(self):
        """处理GUI更新队列"""
        while self.running:
            try:
                update_data = self.gui_queue.get(timeout=0.1)
                # 支持统一负载 gui_update
                if isinstance(update_data, dict) and update_data.get('type') == 'gui_update':
                    self.update_constellation_signal.emit(update_data.get('constellation'))
                    self.update_time_signal.emit(update_data.get('time_signal'))
                    self.update_spectrum_signal.emit(update_data)  # 传入整个 payload
                    self.update_snr_signal.emit(update_data.get('snr'))
                else:
                    update_type = update_data.get('type')

                    if update_type == 'constellation':
                        self.update_constellation_signal.emit(update_data.get('points'))
                    elif update_type == 'time_signal':
                        self.update_time_signal.emit(update_data.get('samples'))
                    elif update_type == 'spectrum':
                        self.update_spectrum_signal.emit(update_data.get('samples'))
                    elif update_type == 'snr':
                        self.update_snr_signal.emit(update_data.get('snr'))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"GUI更新处理错误: {e}")


class ProcessingProgram:
    """DHSCC处理程序"""

    def __init__(self, args):
        self.args = args
        self.running = threading.Event()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化DQPSK解调器
        self.modem = DQPSKModem()

        # 初始化DHSCC接收器
        self.dhscc_receiver = None
        self.upper_layer_queue = None
        if DHSCC_AVAILABLE and hasattr(args, 'enable_dhscc') and args.enable_dhscc:
            self.init_dhscc_receiver(args)

        # UDP communication parameters
        self.enable_udp = getattr(args, 'enable_udp', False)
        self.udp_port = getattr(args, 'udp_port', 50001)

        # 数据缓冲区
        self.data_buffer = queue.Queue(maxsize=10000)  # 加大缓冲区

        # GUI更新队列
        self.gui_queue = queue.Queue(maxsize=100)
        self.accumulated_samples=[]
        # IPC相关
        self.ipc_queue = None
        self.ipc_host = args.host
        self.ipc_port = args.port
        self.control_queue = None

        # 处理统计
        self.processed_blocks = 0
        self.total_samples = 0

    def init_dhscc_receiver(self, args):
        """初始化DHSCC接收器"""
        try:
            print("初始化DHSCC接收器...")

            # DHSCC配置 - 使用5G解码器
            polar_config = {
                'k_polar': 128,  # 码率1/2，k=128
                'n_polar': 256,  # 匹配发送端的N=256
                'channel_type': 'uplink'
            }

            # 模型检查点路径
            model_checkpoint = getattr(args, 'dhscc_checkpoint', 'models/final_receiver_model.pth')

            # 创建DHSCC接收器 - 使用5G解码器
            self.dhscc_receiver = DHSCC_Receiver(
                polar_config=polar_config,
                model_checkpoint=model_checkpoint,
                intermediate_dim=getattr(args, 'intermediate_dim', 128),
                num_classes=getattr(args, 'num_classes', 10),
                use_5g_decoder=True  # 启用5G解码器
            )

            print("✓ DHSCC接收器初始化成功")

        except Exception as e:
            print(f"❌ DHSCC接收器初始化失败: {e}")
            self.dhscc_receiver = None

    def upload_result(self, frame_index, decoded_bits, prediction, confidence):
        """上传处理结果到上层"""
        if self.upper_layer_queue is None:
            return

        try:
            result = {
                'frame_index': frame_index,
                'decoded_bits': decoded_bits,
                'prediction': prediction,
                'confidence': confidence,
                'timestamp': time.time()
            }
            self.upper_layer_queue.put(result, timeout=0.1)
            print(f"✓ 上传结果: 帧索引 {frame_index}, 预测 {prediction}, 置信度 {confidence:.3f}")
        except queue.Full:
            print("⚠ 上层队列满，跳过结果上传")
        except Exception as e:
            print(f"❌ 上传结果失败: {e}")

    def upload_batch_results(self, frame_index, decoded_bits, predictions, confidences):
        """上传batch中所有样本的处理结果到上层"""
        if self.upper_layer_queue is None:
            return

        try:
            for batch_idx, (prediction, confidence) in enumerate(zip(predictions, confidences)):
                result = {
                    'frame_index': frame_index,
                    'batch_index': batch_idx,
                    'decoded_bits': decoded_bits[batch_idx] if isinstance(decoded_bits, np.ndarray) and len(decoded_bits.shape) > 1 else decoded_bits,
                    'prediction': prediction,
                    'confidence': confidence,
                    'timestamp': time.time()
                }
                self.upper_layer_queue.put(result, timeout=0.1)
        except queue.Full:
            print("⚠ 上层队列满，跳过结果上传")
        except Exception as e:
            print(f"❌ 上传batch结果失败: {e}")

    def _dqpsk_differential_decode(self, symbols):
        """
        DQPSK差分解码 - 参考usrp_multiprocess/dqpsk_system.py
        将接收到的DQPSK符号进行差分解码，恢复出原始QPSK符号
        
        Args:
            symbols: DQPSK调制的复数符号数组
            
        Returns:
            decoded_symbols: 解码后的QPSK符号数组
        """
        if len(symbols) == 0:
            return np.array([])
        
        # DQPSK参考符号 (与发送端保持一致)
        diff_ref_symbol = 1 + 0j
        
        decoded = []
        # 第一个符号使用参考符号解码
        decoded.append(symbols[0] * np.conj(diff_ref_symbol))
        
        # 后续符号进行差分解码
        for i in range(1, len(symbols)):
            decoded.append(symbols[i] * np.conj(symbols[i-1]))
        
        return np.array(decoded, dtype=np.complex64)

    def _symbols_to_bits(self, symbols):
        """
        将QPSK符号转换为比特序列 - 参考usrp_multiprocess/dqpsk_system.py
        使用Gray编码映射
        
        Args:
            symbols: QPSK符号数组
            
        Returns:
            bits: 二进制比特数组
        """
        # QPSK星座点 (Gray编码，与发送端保持一致)
        constellation = np.array([
            1 + 1j,  # 00
            -1 + 1j, # 01  
            1 - 1j,  # 10
            -1 - 1j  # 11
        ], dtype=np.complex64) / np.sqrt(2)
        
        bits = []
        for symbol in symbols:
            # 找到最近的星座点
            distances = np.abs(symbol - constellation)
            idx = np.argmin(distances)
            
            # 转换为比特（Gray编码）
            if idx == 0:
                bits.extend([0, 0])
            elif idx == 1:
                bits.extend([0, 1])
            elif idx == 2:
                bits.extend([1, 0])
            else:  # idx == 3
                bits.extend([1, 1])
        
        return np.array(bits)

    def constellation_to_llr(self, constellation):
        """
        将DQPSK星座点转换为LLR值，用于DHSCC处理
        使用正确的软信息LLR计算

        Args:
            constellation: 复数星座点数组 (640符号)

        Returns:
            llrs: LLR数组 (1024,) 用于极化码解码
        """
        # 步骤1: 定义理想星座点和比特映射
        ideal_points = np.array([
            1+1j,   # 00
            -1+1j,  # 01
            1-1j,   # 10
            -1-1j   # 11
        ], dtype=np.complex64) / np.sqrt(2)

        bit_mapping = np.array([
            [0, 0],  # 1+1j -> 00
            [0, 1],  # -1+1j -> 01
            [1, 0],  # 1-1j -> 10
            [1, 1]   # -1-1j -> 11
        ])

        # 步骤2: 计算LLR (使用软信息)
        llr_values = []
        noise_var = 0.1  # 噪声方差，需要根据实际情况调整

        for sym in constellation:
            # 计算到每个理想点的距离平方
            dist_squared = np.abs(sym - ideal_points)**2
            
            # 计算每个符号的概率 (高斯噪声模型)
            symbol_probs = np.exp(-dist_squared / (2 * noise_var))
            
            # 对于每个比特位，计算LLR
            for bit_pos in range(2):
                # 计算bit=0的所有符号的概率之和
                prob_bit0 = sum(symbol_probs[i] for i, bits in enumerate(bit_mapping) if bits[bit_pos] == 0)
                # 计算bit=1的所有符号的概率之和
                prob_bit1 = sum(symbol_probs[i] for i, bits in enumerate(bit_mapping) if bits[bit_pos] == 1)
                
                # 处理零概率情况
                if prob_bit0 == 0:
                    prob_bit0 = 1e-10
                if prob_bit1 == 0:
                    prob_bit1 = 1e-10
                    
                # LLR = log(P(bit=0|y) / P(bit=1|y))
                llr = np.log(prob_bit0 / prob_bit1)
                llr_values.append(llr)

        # 步骤3: 转换为numpy数组并重塑为(5, 256)
        llr_array = np.array(llr_values, dtype=np.float32)
        llrs = llr_array.reshape(5, 256)

        
        return llrs

    def _symbols_to_bits(self, symbols):
        """
        将QPSK符号转换为比特序列 - 参考usrp_multiprocess/dqpsk_system.py
        使用Gray编码映射
        
        Args:
            symbols: QPSK符号数组
            
        Returns:
            bits: 二进制比特数组
        """
        # QPSK星座点 (Gray编码，与发送端保持一致)
        constellation = np.array([
            1 + 1j,  # 00
            -1 + 1j, # 01  
            1 - 1j,  # 10
            -1 - 1j  # 11
        ], dtype=np.complex64) / np.sqrt(2)
        
        bits = []
        for symbol in symbols:
            # 找到最近的星座点
            distances = np.abs(symbol - constellation)
            idx = np.argmin(distances)
            
            # 转换为比特（Gray编码）
            if idx == 0:
                bits.extend([0, 0])
            elif idx == 1:
                bits.extend([0, 1])
            elif idx == 2:
                bits.extend([1, 0])
            else:  # idx == 3
                bits.extend([1, 1])
        
        return np.array(bits)

    def _constellation_to_bits(self, constellation):
        """将星座点转换为比特（硬解调）"""
        if constellation is None:
            return None
        return self.modem._symbols_to_bits(constellation)

    def set_queue(self, ipc_queue):
        """设置IPC队列"""
        self.ipc_queue = ipc_queue

    def connect_queue(self):
        """连接到队列服务器"""
        try:
            # 注册队列管理器
            class QueueManager(BaseManager):
                pass
            QueueManager.register('get_queue')
            QueueManager.register('get_control_queue')

            # 连接到服务器
            manager = QueueManager(address=(self.ipc_host, self.ipc_port), authkey=b'queue_key')
            manager.connect()

            # 获取队列对象
            self.ipc_queue = manager.get_queue()
            try:
                self.control_queue = manager.get_control_queue()
                print("✅ 已连接控制队列")
            except Exception:
                self.control_queue = None
            print("✅ 处理程序队列连接成功")
            return True

        except Exception as e:
            print(f"❌ 处理程序队列连接失败: {e}")
            return False

    def data_receive_thread_func(self):
        """数据接收线程"""
        print("数据接收线程启动")

        while self.running.is_set():
            try:
                if self.ipc_queue:
                    # 从队列接收数据块
                    data_block = self.ipc_queue.get(timeout=0.1)
                    if data_block is not None:
                        # 放入处理缓冲区
                        try:
                            self.data_buffer.put(data_block, timeout=0.1)
                            self.total_samples += len(data_block)
                        except queue.Full:
                            print("处理缓冲区满，丢弃数据块")

                time.sleep(0.01)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"数据接收线程错误: {e}")
                time.sleep(0.1)

        print("数据接收线程结束")

    def data_processing_thread_func(self):
        """数据处理线程：直接处理每个数据块"""
        print("数据处理线程启动")

        while self.running.is_set():
            try:
                # 从队列获取数据（原始样本数据）
                payload = self.data_buffer.get(timeout=0.1)
                samples = payload if isinstance(payload, np.ndarray) else payload.get('samples', np.array([]))

                # 使用滑动窗口逐步处理累积的样本数据
                self.accumulated_samples = np.concatenate([self.accumulated_samples, samples])
                
                window_size = 1800
                step_size =600  # 滑动步长，可根据需要调整
                
                while len(self.accumulated_samples) >= window_size:
                    # 提取当前窗口
                    window = self.accumulated_samples[:window_size]
                    
                    # 处理窗口数据
                    processed = self._process_accumulated_samples(window)
                    
                    # 滑动窗口
                    self.accumulated_samples = self.accumulated_samples[step_size:]


            except queue.Empty:
                continue
            except Exception as e:
                print(f"数据处理线程错误: {e}")
                time.sleep(0.1)

        print("数据处理线程结束")

    def calculate_evm(self, symbols):
        """计算星座点的EVM（均方误差）"""
        if symbols is None or len(symbols) == 0:
            return None
        ideal_points = np.array([1+1j, 1-1j, -1+1j, -1-1j])
        errors = [np.min(np.abs(s - ideal_points)) for s in symbols]
        return np.mean(np.square(errors))

    def weighted_constellation_average(self, symbol_list, evm_list):
        """对多组星座点做加权平均，权重为EVM倒数"""
        epsilon = 1e-6
        weights = np.array([1/(evm+epsilon) for evm in evm_list])
        weights /= np.sum(weights)
        S_array = np.stack(symbol_list, axis=0)
        S_avg = np.sum(weights[:, None] * S_array, axis=0)
        return S_avg

    def process_data_block(self, data_block):
        """处理数据块：顺序调用dqpsk_modem进行解调，返回帧序号和星座点"""
        try:
            # 步骤1: 匹配滤波
            rx_symbols = self.modem.matched_filter(data_block)
            #print(f"接收符号数: {len(rx_symbols)},len(data_block): {len(data_block)}")
            # PSS同步
            timing_offset = self.modem._enhanced_pss_sync(rx_symbols)
            if timing_offset is None or timing_offset < 0 or timing_offset + 128 > len(rx_symbols):
                #print("⚠ 无法检测到同步，跳过该数据块")
                return None, None  # 检测不到同步直接跳过
            #print(f"检测到同步，时序偏移: {timing_offset}")
            rx_symbols = rx_symbols[timing_offset:]
            # 步骤2: 频率校正
            corrected_samples = self.modem.frequency_correction(rx_symbols, timing_offset)
            #print(f"频率校正后样本数: {len(corrected_samples)}")
            # 步骤3: 差分解码
            demod_symbols = self.modem.differential_decode(corrected_samples)


            # 步骤4: 解码帧序号
            frame_index = self.modem.decode_frame_index(demod_symbols)
            if frame_index is None:
                return None, None, None  # 无法解码帧序号，跳过
            #print(f"解码帧序号: {frame_index}")

            # 步骤5: 计算SNR
            snr = self.modem.calculate_snr(demod_symbols)

            # 步骤6: 提取数据符号（从前导后开始，640符号）
            preamble_len = 128  # PSS(32) + SSS(32) + RS(64) = 128
            frame_index_len = 4  # 帧序号4符号
            data_len = 640  # 数据640符号
            
            # 计算数据符号的起始位置
            data_start = preamble_len + frame_index_len  # 128 + 4 = 132
            data_end = data_start + data_len  # 132 + 640 = 772
            #print(f"len(demod_symbols): {len(demod_symbols)}, data_start: {data_start}, data_end: {data_end}")
            if data_end <= len(demod_symbols):
                data_symbols = demod_symbols[data_start:data_end]
            else:
                # 如果数据不完整，返回None
                return None, None, None 

            # 步骤7: 生成星座图（使用数据符号）
            constellation = data_symbols

            # 发送到GUI更新队列
            try:
                # 星座图
                self.gui_queue.put({
                    'type': 'constellation',
                    'points': constellation
                }, timeout=0.1)

                # 时域信号（校正后的）
                self.gui_queue.put({
                    'type': 'time_signal',
                    'samples': corrected_samples[:2500]
                }, timeout=0.1)

                # 频谱（校正后的信号）
                self.gui_queue.put({
                    'type': 'spectrum',
                    'samples': corrected_samples[-4096:] if len(corrected_samples) >= 4096 else corrected_samples
                }, timeout=0.1)

                # SNR
                if snr is not None:
                    self.gui_queue.put({
                        'type': 'snr',
                        'snr': snr
                    }, timeout=0.1)

            except queue.Full:
                pass  # GUI队列满，跳过

            self.processed_blocks += 1
            self.total_samples += len(data_block)

            return frame_index,snr, constellation

        except Exception as e:
            print(f"数据处理错误: {e}")
            return None, None

    def constellation_to_llr(self, constellation):
        """
        将DQPSK星座点转换为LLR值，用于DHSCC处理
        使用正确的软信息LLR计算

        Args:
            constellation: 复数星座点数组 (640符号)

        Returns:
            llrs: LLR数组 (1024,) 用于极化码解码
        """
        # 步骤1: 定义理想星座点和比特映射
        ideal_points = np.array([
            1+1j,   # 00
            -1+1j,  # 01
            1-1j,   # 10
            -1-1j   # 11
        ], dtype=np.complex64) / np.sqrt(2)

        bit_mapping = np.array([
            [0, 0],  # 1+1j -> 00
            [0, 1],  # -1+1j -> 01
            [1, 0],  # 1-1j -> 10
            [1, 1]   # -1-1j -> 11
        ])

        # 步骤2: 计算LLR (使用软信息)
        llr_values = []
        noise_var = 0.01  # 减小噪声方差以获得更连续的LLR值

        for sym in constellation:
            # 计算到每个理想点的距离平方
            dist_squared = np.abs(sym - ideal_points)**2

            # 计算每个符号的概率 (高斯噪声模型)
            symbol_probs = np.exp(-dist_squared / (2 * noise_var))

            # 对于每个比特位，计算LLR
            for bit_pos in range(2):
                # 计算bit=0的所有符号的概率之和
                prob_bit0 = sum(symbol_probs[i] for i, bits in enumerate(bit_mapping) if bits[bit_pos] == 0)
                # 计算bit=1的所有符号的概率之和
                prob_bit1 = sum(symbol_probs[i] for i, bits in enumerate(bit_mapping) if bits[bit_pos] == 1)

                # 处理零概率情况
                if prob_bit0 == 0:
                    prob_bit0 = 1e-10
                if prob_bit1 == 0:
                    prob_bit1 = 1e-10

                # LLR = log(P(bit=0|y) / P(bit=1|y))
                llr = np.log(prob_bit0 / prob_bit1)
                llr_values.append(llr)

        # 步骤3: 转换为numpy数组并重塑为(5, 256)
        llr_array = np.array(llr_values, dtype=np.float32)
        llrs = llr_array.reshape(5, 256)


        return llrs

    def start(self):
        """启动处理程序"""
        if self.running.is_set():
            print("处理程序已在运行")
            return

        print("启动处理程序...")

        # 在UDP模式下跳过队列连接
        if not self.enable_udp:
            # 连接队列
            if not self.connect_queue():
                print("队列连接失败，处理程序无法启动")
                return
        else:
            print("UDP模式：跳过队列连接")

        self.running.set()

        # 在非UDP模式下启动数据接收和处理线程
        if not self.enable_udp:
            # 启动数据接收线程
            self.receive_thread = threading.Thread(target=self.data_receive_thread_func, daemon=True)
            self.receive_thread.start()

            # 启动数据处理线程
            self.processing_thread = threading.Thread(target=self.data_processing_thread_func, daemon=True)
            self.processing_thread.start()

        # 启动UDP服务器线程（如果启用）
        if self.enable_udp:
            self.udp_thread = threading.Thread(target=self.udp_server_thread, daemon=True)
            self.udp_thread.start()
            print("UDP服务器线程已启动")

        # 启动GUI
        if PYQT_AVAILABLE:
            self.start_gui()
        else:
            print("GUI不可用，仅运行后台处理")

        print("处理程序启动完成")

        # 保持运行并显示统计信息
        try:
            while self.running.is_set():
                print(f"\r已处理数据块: {self.processed_blocks}, 总样本数: {self.total_samples}", end="")
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n处理程序被用户中断")

    def start_gui(self):
        """启动GUI"""
        app = QApplication(sys.argv)

        # 创建监控界面
        self.monitor = DHSCCMonitor()

        # 创建GUI管理器
        self.gui_manager = GUIManager(self.gui_queue)

        # 将monitor引用传递给gui_manager
        self.gui_manager.monitor = self.monitor

        # 启动GUI更新处理线程
        gui_thread = threading.Thread(target=self.gui_manager.process_gui_updates, daemon=True)
        gui_thread.start()

        # 在GUI增加增益控制滑条（位于右侧面板上）
        try:
            slider = pg.QtWidgets.QSlider(pg.QtCore.Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(76)
            slider.setValue(20)

            def on_slider_change(val):
                # 将命令发到控制队列
                try:
                    if self.control_queue is not None:
                        self.control_queue.put({'cmd': 'set_gain', 'val': float(val)})
                except Exception as e:
                    print(f"发送增益命令失败: {e}")

            slider.valueChanged.connect(on_slider_change)
            # 将滑条添加到界面底部
            from PyQt5.QtWidgets import QVBoxLayout
            if isinstance(self.monitor.centralWidget().layout(), QVBoxLayout):
                self.monitor.centralWidget().layout().addWidget(slider)
        except Exception:
            pass

        # 显示界面
        self.monitor.show()

        # 运行应用
        app.exec_()

    def stop(self):
        """停止处理程序"""
        if not self.running.is_set():
            print("处理程序未运行")
            return

        print("停止处理程序...")
        self.running.clear()

        # 等待线程结束
        if hasattr(self, 'receive_thread') and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=2.0)

        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)

        print("处理程序停止完成")

    def _generate_5g_frozen_pos(self, k=128, n=256):
        """
        生成5G Polar码的frozen positions
        
        Args:
            k: 信息比特数 (不含CRC)
            n: 码长
            
        Returns:
            frozen_pos: 冻结位置数组
        """
        # 对于码率1/2的普通5G Polar码，不使用CRC
        k_polar = k  # 不添加CRC
        
        # 加载5G channel ranking
        try:
            csv_path = os.path.join(os.path.dirname(__file__), 'dhscc_modules', 'polar_5G.csv')
            ch_order = np.genfromtxt(csv_path, delimiter=";")
            ch_order = ch_order.astype(int)
        except FileNotFoundError:
            # 如果找不到CSV文件，使用简单的降序排列作为fallback
            print("Warning: Could not find polar_5G.csv, using fallback frozen positions")
            return np.arange(n - k_polar)
        
        # 选择前n个channel positions
        if len(ch_order) >= n:
            ch_order_n = ch_order[:n]
        else:
            # 如果CSV中没有足够的positions，扩展它
            ch_order_n = ch_order
            while len(ch_order_n) < n:
                ch_order_n = np.vstack([ch_order_n, ch_order])
            ch_order_n = ch_order_n[:n]
        
        # 按可靠性排序（从小到大，较小的索引更可靠）
        ch_order_n = ch_order_n[np.argsort(ch_order_n[:, 0])]
        
        # 最差的(n - k_polar)个位置作为frozen positions
        frozen_pos = ch_order_n[:n - k_polar, 1]  # 第二列是位置索引
        
        return frozen_pos.astype(int)

    def udp_server_thread(self):
        """UDP服务器线程：接收并处理UDP数据"""
        print(f"UDP服务器启动，监听端口 {self.udp_port}")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind(('0.0.0.0', self.udp_port))
            
            while self.running.is_set():
                try:
                    # 接收数据（设置超时）
                    sock.settimeout(0.1)
                    data, addr = sock.recvfrom(65536)
                    
                    # 反序列化
                    try:
                        unpacked_data = pickle.loads(data)
                        if unpacked_data is None:
                            print("UDP接收到空数据，跳过")
                            continue
                        idx, tx_signal = unpacked_data
                        #idx, binary_bits = unpacked_data
                    except (pickle.UnpicklingError, ValueError, TypeError) as e:
                        print(f"UDP数据反序列化失败: {e}，数据长度: {len(data)}")
                        continue
                    
                    frame_ind, constellation = self.process_data_block(tx_signal)
                    frame_llr = self.constellation_to_llr(constellation)
                    frame_llr = torch.from_numpy(frame_llr).to(self.device)
                    selected_indices_shape = (frame_llr.shape[0], 8, 4)
                    if self.dhscc_receiver is not None:
                        predictions = self.dhscc_receiver.decode_and_classify(
                            frame_llr,
                            channel_noise=0.0,
                            selected_indices_shape=selected_indices_shape
                        )
                        print(f"UDP帧 {idx} 处理完成, prediction: {predictions}")
                    else:
                        print(f"UDP帧 {idx} 跳过：DHSCC接收器未初始化")
                        
                except Exception as e:
                    #print(f"UDP处理错误: {e}")
                    continue
                    
        except Exception as e:
            print(f"UDP服务器错误: {e}")
        finally:
            sock.close()
            print("UDP服务器线程结束")

    def soft_qpsk_demodulate(self, symbols):
        """
        软QPSK解调：计算LLR值用于DHSCC处理
        
        Args:
            symbols: 接收到的复数QPSK符号数组
            
        Returns:
            llrs: LLR数组 (5, 256) 用于极化码解码
        """
        # 理想QPSK星座点 (Gray编码)
        ideal_points = np.array([
            1+1j,   # 00
            -1+1j,  # 01
            1-1j,   # 10
            -1-1j   # 11
        ], dtype=np.complex64) / np.sqrt(2)

        # 比特映射
        bit_mapping = np.array([
            [0, 0],  # 1+1j -> 00
            [0, 1],  # -1+1j -> 01
            [1, 0],  # 1-1j -> 10
            [1, 1]   # -1-1j -> 11
        ])

        # 噪声方差（基于默认SNR 20dB）
        noise_var = 10**(-20/10)  # ~0.01
        
        llr_values = []
        for sym in symbols:
            # 计算到每个理想点的距离平方
            dist_squared = np.abs(sym - ideal_points)**2
            
            # 计算每个符号的概率 (高斯噪声模型)
            symbol_probs = np.exp(-dist_squared / (2 * noise_var))
            
            # 对于每个比特位，计算LLR
            for bit_pos in range(2):
                # 计算bit=0的所有符号的概率之和
                prob_bit0 = sum(symbol_probs[i] for i, bits in enumerate(bit_mapping) if bits[bit_pos] == 0)
                # 计算bit=1的所有符号的概率之和
                prob_bit1 = sum(symbol_probs[i] for i, bits in enumerate(bit_mapping) if bits[bit_pos] == 1)
                
                # 处理零概率情况
                prob_bit0 = max(prob_bit0, 1e-10)
                prob_bit1 = max(prob_bit1, 1e-10)
                
                # LLR = log(P(bit=0|y) / P(bit=1|y))
                llr = np.log(prob_bit0 / prob_bit1)
                llr_values.append(llr)

        # 转换为numpy数组并重塑为(5, 256)
        llr_array = np.array(llr_values, dtype=np.float32)
        llrs = llr_array.reshape(5, 256)
        
        return llrs

    def _process_accumulated_samples(self, samples):
        """处理累积的样本数据 - 直接处理数据块"""
        try:
            # 直接使用process_data_block处理输入的samples
            frame_index, snr, constellation = self.process_data_block(samples)

            # 如果检测不到帧头，立即跳出
            if frame_index is None or constellation is None or snr < 0:
                return False

            # 转换为LLR
            frame_llr = self.constellation_to_llr(constellation)
            frame_llr = torch.from_numpy(frame_llr).to(self.device)
            selected_indices_shape = (frame_llr.shape[0], 8, 4)
            with open('snr_log.txt', 'a') as f:
                f.write(f'{snr}\n')

            # DHSCC处理
            if self.dhscc_receiver is not None and snr >=5:
                predictions = self.dhscc_receiver.decode_and_classify(
                    frame_llr,
                    channel_noise=0.0,
                    selected_indices_shape=selected_indices_shape
                )
                print(f"帧 {frame_index} 处理完成, prediction: {predictions}")
            else:
                print(f"帧 {frame_index} 跳过：DHSCC接收器未初始化或SNR不足")

            self.processed_blocks += 1
            return True

        except Exception as e:
            print(f"累积样本处理错误: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="DHSCC处理程序")
    parser.add_argument("--mode", type=str, default="hardware", choices=["hardware", "simulation"], help="运行模式")
    parser.add_argument("--rate", type=float, default=10e6, help="采样率 (Hz)")
    parser.add_argument("--center_freq", type=float, default=915e6, help="中心频率 (Hz)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="队列服务器主机地址")
    parser.add_argument("--port", type=int, default=50000, help="队列服务器端口")
    parser.add_argument("--ipc_mode", type=str, default="queue", choices=["queue", "udp"], help="IPC模式")

    # DHSCC相关参数
    parser.add_argument("--enable_dhscc", action="store_true", default=True, help="启用DHSCC接收器")
    parser.add_argument("--dhscc_checkpoint", type=str, default="models/final_receiver_model.pth", help="DHSCC模型检查点路径")
    parser.add_argument("--intermediate_dim", type=int, default=128, help="DHSCC中间表示维度")
    parser.add_argument("--num_classes", type=int, default=10, help="分类类别数")

    # UDP通信参数
    parser.add_argument("--enable_udp", action="store_true", help="启用UDP通信模式")
    parser.add_argument("--udp_port", type=int, default=50001, help="UDP监听端口")

    args = parser.parse_args()

    # 创建处理程序
    processing_program = ProcessingProgram(args)

    # 设置信号处理器
    def signal_handler(signum, frame):
        print(f"\n接收到信号 {signum}，停止处理程序...")
        processing_program.stop()
        sys.exit(0)

    import signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # 启动处理程序
        processing_program.start()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"运行错误: {str(e)}")
    finally:
        processing_program.stop()


if __name__ == "__main__":
    main()