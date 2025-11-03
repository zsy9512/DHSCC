#!/usr/bin/env python3
"""
DHSCC USRP接收系统主程序
管理三个进程：队列服务器、接收程序、处理程序
"""

import subprocess
import time
import sys
import os
import argparse
import signal
import threading
import atexit
from multiprocessing.managers import BaseManager

# Import ProcessingProgram from processing_program.py
from processing_program import ProcessingProgram

class DHSCCReceiverSystem:
    """DHSCC接收系统管理器"""

    def __init__(self, host="127.0.0.1", port=50000, record_file='received_data.bin',
                 gain=20.0, freq=2.4e9, rate=10e6, send_block_size=8000,
                 mode="hardware", center_freq=915e6, ipc_mode="queue",
                 enable_dhscc=False, dhscc_checkpoint="models/dhscc_receiver_model.pth",
                 intermediate_dim=128, num_classes=10, enable_udp=False, udp_port=50001):
        self.host = host
        self.port = port
        self.record_file = record_file
        self.gain = gain
        self.freq = freq
        self.rate = rate
        self.send_block_size = send_block_size
        self.mode = mode
        self.center_freq = center_freq
        self.ipc_mode = ipc_mode
        self.enable_dhscc = enable_dhscc
        self.dhscc_checkpoint = dhscc_checkpoint
        self.intermediate_dim = intermediate_dim
        self.num_classes = num_classes
        self.enable_udp = enable_udp
        self.udp_port = udp_port
        self.processes = []
        self.threads = []

        # 注册退出时的清理函数
        atexit.register(self._emergency_cleanup)

    def _emergency_cleanup(self):
        """紧急清理函数"""
        print("\n🧹 执行紧急清理...")
        for name, process in self.processes:
            try:
                if process.poll() is None:  # 仍在运行
                    print(f"终止进程: {name}")
                    process.terminate()
                    process.wait(timeout=5.0)
            except Exception as e:
                print(f"清理进程 {name} 时出错: {e}")

    def start_queue_server(self):
        """启动队列服务器"""
        print("🚀 启动队列服务器...")
        cmd = [
            sys.executable, "queue_server.py",
            "--host", self.host,
            "--port", str(self.port)
        ]

        process = subprocess.Popen(cmd, cwd=os.getcwd())
        self.processes.append(("Queue Server", process))

        # 等待服务器启动
        time.sleep(2)
        if process.poll() is None:
            print("✅ 队列服务器启动成功")
            return True
        else:
            print("❌ 队列服务器启动失败")
            return False

    def start_rx_program(self):
        """启动接收程序"""
        print("📡 启动接收程序...")
        cmd = [
            sys.executable, "rx_program.py",
            "--gain", str(self.gain),
            "--freq", str(self.freq),
            "--rate", str(self.rate),
            "--send_block_size", str(self.send_block_size) 
        ]

        if self.record_file:
            cmd.extend(["--record_file", self.record_file])

        process = subprocess.Popen(cmd, cwd=os.getcwd())
        self.processes.append(("RX Program", process))

        # 等待接收程序启动
        time.sleep(2)
        if process.poll() is None:
            print("✅ 接收程序启动成功")
            return True
        else:
            print("❌ 接收程序启动失败")
            return False

    def start_processing_program(self):
        """启动处理程序（集成模式）"""
        print("📊 启动处理程序（集成模式）...")

        # 创建args对象传递给ProcessingProgram
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        args = Args(
            mode=self.mode,
            rate=self.rate,
            center_freq=self.center_freq,
            host=self.host,
            port=self.port,
            ipc_mode=self.ipc_mode,
            enable_dhscc=self.enable_dhscc,
            dhscc_checkpoint=self.dhscc_checkpoint,
            intermediate_dim=self.intermediate_dim,
            num_classes=self.num_classes,
            enable_udp=self.enable_udp,
            udp_port=self.udp_port
        )

        # 创建处理程序实例
        self.processing_program = ProcessingProgram(args)

        # 启动处理程序（这会启动线程和GUI）
        try:
            self.processing_program.start()
            print("✅ 处理程序启动成功")
            return True
        except Exception as e:
            print(f"❌ 处理程序启动失败: {e}")
            return False

    def connect_queues(self):
        """连接队列到接收程序"""
        print("🔗 连接队列到接收程序...")

        try:
            # 注册队列管理器
            class QueueManager(BaseManager):
                pass
            QueueManager.register('get_queue')

            # 连接到服务器
            manager = QueueManager(address=(self.host, self.port), authkey=b'queue_key')
            manager.connect()

            # 获取队列对象
            ipc_queue = manager.get_queue()
            print("✅ 队列连接成功")

            # 这里需要一种方式将队列传递给接收程序
            # 由于接收程序已经启动，我们需要通过其他方式设置队列
            # 暂时返回队列对象，稍后处理

            return ipc_queue

        except Exception as e:
            print(f"❌ 队列连接失败: {e}")
            return None

    def monitor_processes(self):
        """监控进程状态"""
        while True:
            all_alive = True
            for name, process in self.processes:
                if process.poll() is not None:
                    print(f"⚠ 进程 {name} 已退出 (返回码: {process.returncode})")
                    all_alive = False

            if not all_alive:
                print("❌ 检测到进程异常退出，系统将停止")
                self.stop()
                break

            time.sleep(1.0)

    def start(self):
        """启动整个系统"""
        print("🎯 启动DHSCC接收系统...")
        print(f"配置: 频率={self.freq/1e6:.1f}MHz, 增益={self.gain}dB, 采样率={self.rate/1e6:.1f}MHz")

        success = True

        # 1. 启动队列服务器
        if not self.start_queue_server():
            success = False

        # 2. 启动接收程序
        if success and not self.start_rx_program():
            success = False

        # 3. 连接队列
        if success:
            ipc_queue = self.connect_queues()
            if ipc_queue is None:
                success = False

        # 4. 启动处理程序
        if success and not self.start_processing_program():
            success = False

        if success:
            print("✅ 系统启动完成")
            print("📋 运行中的进程:")
            for name, process in self.processes:
                print(f"   • {name} (PID: {process.pid})")

            # 启动监控线程
            monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
            monitor_thread.start()
            self.threads.append(monitor_thread)

            # 等待用户中断
            try:
                print("\n⏹ 按Ctrl+C停止系统")
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("\n🛑 用户请求停止")
                self.stop()

        else:
            print("❌ 系统启动失败")
            self.stop()

    def stop(self):
        """停止整个系统"""
        print("🛑 停止DHSCC接收系统...")

        # 停止处理程序
        if hasattr(self, 'processing_program'):
            try:
                print("停止处理程序...")
                self.processing_program.stop()
                print("✅ 处理程序已停止")
            except Exception as e:
                print(f"停止处理程序时出错: {e}")

        # 终止所有进程
        for name, process in self.processes:
            try:
                if process.poll() is None:  # 仍在运行
                    print(f"终止进程: {name}")
                    process.terminate()
                    # 等待最多5秒
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        print(f"强制终止进程: {name}")
                        try:
                            process.kill()
                            process.wait()
                        except Exception:
                            # On Windows sometimes process may be stubborn; try taskkill
                            if os.name == 'nt':
                                try:
                                    subprocess.run(['taskkill', '/F', '/PID', str(process.pid)], check=False)
                                except Exception:
                                    pass
                else:
                    print(f"进程 {name} 已退出")
            except Exception as e:
                print(f"停止进程 {name} 时出错: {e}")

        self.processes.clear()
        print("✅ 系统已停止")


def main():
    parser = argparse.ArgumentParser(description='DHSCC USRP接收系统')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='队列服务器主机地址')
    parser.add_argument('--port', type=int, default=50000,
                       help='队列服务器端口')
    parser.add_argument('--record_file', type=str, default='received_data.bin',
                       help='记录接收数据到二进制文件 (默认: received_data.bin)')
    parser.add_argument('--gain', type=float, default=60.0,
                       help='接收增益 (dB)')
    parser.add_argument('--freq', type=float, default=915e6,
                       help='接收频率 (Hz)')
    parser.add_argument('--rate', type=float, default=1e6,
                       help='采样率 (Hz)')
    parser.add_argument('--send_block_size', type=int, default=8000,
                       help='IPC发送块大小 (默认: 8000)')

    # Processing program arguments
    parser.add_argument("--mode", type=str, default="hardware", choices=["hardware", "simulation"],
                       help="运行模式")
    parser.add_argument("--center_freq", type=float, default=915e6,
                       help="中心频率 (Hz)")
    parser.add_argument("--ipc_mode", type=str, default="queue", choices=["queue", "udp"],
                       help="IPC模式")

    # DHSCC相关参数
    parser.add_argument("--enable_dhscc", action="store_true", default=True,
                       help="启用DHSCC接收器")
    parser.add_argument("--dhscc_checkpoint", type=str, default="models/final_receiver_model.pth",
                       help="DHSCC模型检查点路径")
    parser.add_argument("--intermediate_dim", type=int, default=128,
                       help="DHSCC中间表示维度")
    parser.add_argument("--num_classes", type=int, default=10,
                       help="分类类别数")

    # UDP通信参数
    parser.add_argument("--enable_udp", action="store_true",
                       help="启用UDP通信模式")
    parser.add_argument("--udp_port", type=int, default=50001,
                       help="UDP监听端口")

    args = parser.parse_args()

    # 创建系统管理器
    system = DHSCCReceiverSystem(
        host=args.host,
        port=args.port,
        record_file=args.record_file,
        gain=args.gain,
        freq=args.freq,
        rate=args.rate,
        send_block_size=args.send_block_size,
        mode=args.mode,
        center_freq=args.center_freq,
        ipc_mode=args.ipc_mode,
        enable_dhscc=args.enable_dhscc,
        dhscc_checkpoint=args.dhscc_checkpoint,
        intermediate_dim=args.intermediate_dim,
        num_classes=args.num_classes,
        enable_udp=args.enable_udp,
        udp_port=args.udp_port
    )

    # 设置信号处理器
    def signal_handler(signum, frame):
        print(f"\n接收到信号 {signum}，停止系统...")
        system.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # 启动系统
        system.start()
    except Exception as e:
        print(f"系统运行错误: {str(e)}")
        system.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()