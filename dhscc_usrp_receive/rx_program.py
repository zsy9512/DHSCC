import numpy as np
import uhd
import threading
import queue
import time
import argparse
import collections
import os
import pickle
import socket
import struct
import multiprocessing
from multiprocessing.managers import BaseManager
import signal
import sys

class RXProgram:
    """接收程序：高速接收数据，滤除噪声，通过IPC发送到处理程序"""

    def __init__(self, record_file=None, gain=20.0, freq=2.4e9, rate=10e6,
                 buffer_size=1024*1024, send_block_size=8000):
        self.running = threading.Event()
        self.rx_enabled = threading.Event()

        # USRP相关
        self.usrp = None
        self.rx_streamer = None

        # 数据记录相关
        self.record_file = record_file
        self.record_fp = None
        if self.record_file:
            try:
                self.record_fp = open(self.record_file, 'wb')
                print(f"接收数据将保存至: {self.record_file}")
            except Exception as e:
                print(f"无法打开记录文件 {self.record_file}: {e}")
                self.record_fp = None

        # USRP参数
        self.gain = gain
        self.freq = freq
        self.rate = rate

        # 环形缓冲区设计
        self.buffer_size = max(buffer_size, 500000000)  # 确保最小500000样本
        self.rx_buffer = np.zeros(self.buffer_size, dtype=np.complex64)
        self.buffer_head = 0  # 写入指针
        self.buffer_tail = 0  # 读取指针
        self.buffer_lock = threading.Lock()  # 保护环形缓冲区

        # 发送块大小（连续读取的长度）
        self.send_block_size = send_block_size

        # IPC相关（只用queue模式）
        self.ipc_queue = None
        self.ipc_host = "127.0.0.1"
        self.ipc_port = 50000
        # 控制队列
        self.control_queue = None

        # 统计信息
        self.rx_samples_received = 0
        self.noise_discard_count = 0
        self.overflow_count = 0

        # 线程
        self.rx_thread = None
        self.ipc_send_thread = None

    def connect_queue(self):
        """连接到队列服务器"""
        try:
            from multiprocessing.managers import BaseManager

            # 注册队列管理器
            class QueueManager(BaseManager):
                pass
            QueueManager.register('get_queue')

            # 连接到服务器
            manager = QueueManager(address=(self.ipc_host, self.ipc_port), authkey=b'queue_key')
            manager.connect()

            # 获取队列对象
            self.ipc_queue = manager.get_queue()
            try:
                # 可选获取控制队列
                self.control_queue = manager.get_control_queue()
            except Exception:
                self.control_queue = None
            print("✅ 接收程序队列连接成功")
            return True

        except Exception as e:
            print(f"❌ 接收程序队列连接失败: {e}")
            return False

    def set_queue(self, ipc_queue):
        """设置IPC Queue对象"""
        self.ipc_queue = ipc_queue
        print("Queue对象已设置")

    def _get_available_space(self):
        """计算环形缓冲区的可用空间"""
        if self.buffer_head >= self.buffer_tail:
            return self.buffer_size - self.buffer_head + self.buffer_tail - 1
        else:
            return self.buffer_tail - self.buffer_head - 1

    def _write_to_buffer(self, samples):
        """写入数据到环形缓冲区"""
        if samples is None or not isinstance(samples, np.ndarray) or samples.dtype not in [np.complex64, np.complex128]:
            return

        num_samps = len(samples)
        with self.buffer_lock:
            available_space = self._get_available_space()
            if available_space >= num_samps:
                end_pos = (self.buffer_head + num_samps) % self.buffer_size
                if end_pos > self.buffer_head:
                    self.rx_buffer[self.buffer_head:end_pos] = samples
                else:
                    mid = self.buffer_size - self.buffer_head
                    self.rx_buffer[self.buffer_head:] = samples[:mid]
                    self.rx_buffer[:end_pos] = samples[mid:]
                self.buffer_head = end_pos
            else:
                self.overflow_count += 1

    def _read_from_buffer(self, num_samples):
        """从环形缓冲区读取固定长度的数据"""
        with self.buffer_lock:
            available_samples = (self.buffer_head - self.buffer_tail) % self.buffer_size
            if available_samples >= num_samples:
                end_pos = (self.buffer_tail + num_samples) % self.buffer_size
                if end_pos > self.buffer_tail:
                    data_block = self.rx_buffer[self.buffer_tail:end_pos].copy()
                else:
                    first_part = self.buffer_size - self.buffer_tail
                    data_block = np.concatenate([
                        self.rx_buffer[self.buffer_tail:],
                        self.rx_buffer[:end_pos]
                    ])
                self.buffer_tail = end_pos
                return data_block
            else:
                return None

    def _init_usrp(self):
        """初始化USRP设备"""
        try:
            self.usrp = uhd.usrp.MultiUSRP("name=MyB210_01")# 修改这里以匹配USRP接收设备名
            self.usrp.set_clock_source("internal")
            self.usrp.set_time_source("internal")
            pc_time_ns = time.time_ns()
            full_secs = pc_time_ns // 1000000000
            frac_secs = (pc_time_ns % 1000000000) / 1000000000.0
            uhd_time = uhd.types.TimeSpec(full_secs, frac_secs)
            self.usrp.set_time_now(uhd_time)
            self.usrp.set_rx_freq(uhd.types.TuneRequest(self.freq))
            self.usrp.set_rx_gain(self.gain)
            self.usrp.set_rx_rate(self.rate)

            print(f"USRP接收初始化完成: 频率={self.freq/1e6:.1f}MHz, 增益={self.gain}dB")

            rx_st_args = uhd.usrp.StreamArgs("fc32", "sc16")
            rx_st_args.channels = [0]
            self.rx_streamer = self.usrp.get_rx_stream(rx_st_args)

            print("接收流创建完成")

        except Exception as e:
            print(f"USRP初始化失败: {str(e)}")
            raise

    def rx_thread_func(self):
        """接收线程：高速接收数据，滤除噪声"""
        print(f"接收线程启动: 缓冲区大小 {self.buffer_size}")
        # 在接收线程中同时处理控制命令
        self._rx_hardware_thread_func()

    def _rx_hardware_thread_func(self):
        """硬件模式接收线程"""
        if self.rx_streamer is None:
            print("接收线程: rx_streamer未初始化")
            return

        fast_recv_buffer_size = 2040
        recv_buffer = np.zeros((1, fast_recv_buffer_size), dtype=np.complex64)
        metadata = uhd.types.RXMetadata()

        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True
        self.rx_streamer.issue_stream_cmd(stream_cmd)

        monitor_count = 0

        while self.running.is_set():
            if self.rx_enabled.is_set():
                try:
                    num_samps = self.rx_streamer.recv(recv_buffer, metadata, timeout=0.1)
                    if num_samps > 0:
                        samples = recv_buffer[0][:num_samps].copy()
                        self.rx_samples_received += num_samps

                        # # 数据记录
                        # if self.record_fp is not None:
                        #     samples.astype(np.complex64).tofile(self.record_fp)

                        # 功率检测和帧边界识别
                        step = max(1, len(samples) // 100)
                        signal_power = np.mean(np.abs(samples[::step])**2)

                        # 噪声检测（只写入高功率数据）
                        if signal_power < 0.01:
                            self.noise_discard_count += 1
                            continue
                        # 检查控制队列（非阻塞）
                        # try:
                        #     if self.control_queue is not None and not self.control_queue.empty():
                        #         cmd = self.control_queue.get_nowait()
                        #         self._handle_control_command(cmd)
                        # except Exception:
                        #     pass
                        self._write_to_buffer(samples)

                except Exception as e:
                    if monitor_count % 1000 == 0:
                        print(f"接收线程错误: {str(e)}")
            else:
                time.sleep(0.001)

        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        self.rx_streamer.issue_stream_cmd(stream_cmd)
        print("接收线程结束")

    def _handle_control_command(self, cmd):
        """处理控制命令，命令结构：{'cmd':'set_gain', 'val': 30} 等"""
        try:
            if not isinstance(cmd, dict) or 'cmd' not in cmd:
                return
            c = cmd.get('cmd')
            if c == 'set_gain':
                val = float(cmd.get('val', self.gain))
                self.gain = val
                if self.usrp is not None:
                    try:
                        self.usrp.set_rx_gain(val)
                        print(f"🔧 已设置RX增益: {val} dB")
                    except Exception as e:
                        print(f"设置RX增益失败: {e}")
            elif c == 'start_rx':
                self.rx_enabled.set()
                print("🔛 RX已启用")
            elif c == 'stop_rx':
                self.rx_enabled.clear()
                print("⏹ RX已停止")
        except Exception as e:
            print(f"处理控制命令错误: {e}")

    def ipc_send_thread_func(self):
        """IPC发送线程：从环形缓冲区连续读取固定长度数据，通过Queue发送"""
        print("Queue IPC发送线程启动")
        while self.running.is_set():
            try:
                data_block = self._read_from_buffer(self.send_block_size)
                if data_block is not None and self.ipc_queue:
                    # 只发送原始数据块，不带帧边界标志
                    try:
                        self.ipc_queue.put(data_block, timeout=0.1)
                    except queue.Full:
                        print("Queue满，丢弃数据块")
                    except Exception as e:
                        print(f"Queue发送错误: {str(e)}")
                time.sleep(0.01)
            except Exception as e:
                print(f"IPC发送线程错误: {str(e)}")
                time.sleep(0.1)
        print("IPC发送线程结束")

    def start(self):
        """启动接收器"""
        if self.running.is_set():
            print("接收器已在运行")
            return

        print("启动接收器...")

        # 初始化USRP
        try:
            self._init_usrp()
        except Exception as e:
            print(f"USRP初始化失败: {e}")
            return

        # 连接队列
        if not self.connect_queue():
            print("队列连接失败，接收器无法启动")
            return

        # 默认启用接收（可由control_queue关闭）
        self.rx_enabled.set()

        self.running.set()

        # 启动接收线程
        self.rx_thread = threading.Thread(target=self.rx_thread_func, daemon=True)
        self.rx_thread.start()

        # 启动IPC发送线程
        self.ipc_thread = threading.Thread(target=self.ipc_send_thread_func, daemon=True)
        self.ipc_thread.start()

        print("接收器启动完成")

    def stop(self):
        """停止接收器"""
        if not self.running.is_set():
            print("接收器未运行")
            return

        print("停止接收器...")
        self.running.clear()

        # 等待线程结束
        if self.rx_thread and self.rx_thread.is_alive():
            self.rx_thread.join(timeout=2.0)

        if self.ipc_thread and self.ipc_thread.is_alive():
            self.ipc_thread.join(timeout=2.0)

        # 关闭文件
        if self.record_fp:
            self.record_fp.close()
            self.record_fp = None

        print("接收器停止完成")


def main():
    """主函数：解析命令行参数并运行接收器"""
    parser = argparse.ArgumentParser(description='DHSCC USRP接收程序')
    parser.add_argument('--record_file', type=str, default=None,
                       help='记录接收数据到二进制文件 (可选)')
    parser.add_argument('--gain', type=float, default=20.0,
                       help='接收增益 (dB)')
    parser.add_argument('--freq', type=float, default=2.4e9,
                       help='接收频率 (Hz)')
    parser.add_argument('--rate', type=float, default=10e6,
                       help='采样率 (Hz)')
    parser.add_argument('--buffer_size', type=int, default=1024*1024,
                       help='环形缓冲区大小')
    parser.add_argument('--send_block_size', type=int, default=1024,
                       help='IPC发送块大小')

    args = parser.parse_args()

    # 创建接收器实例
    receiver = RXProgram(
        record_file=args.record_file,
        gain=args.gain,
        freq=args.freq,
        rate=args.rate,
        buffer_size=args.buffer_size,
        send_block_size=args.send_block_size
    )

    # 设置信号处理器
    def signal_handler(signum, frame):
        print(f"\n接收到信号 {signum}，停止接收器...")
        receiver.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # 启动接收器
        receiver.start()

        # 保持运行
        print("接收器运行中... 按Ctrl+C停止")
        while receiver.running.is_set():
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"运行错误: {str(e)}")
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()