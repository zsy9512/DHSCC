#!/usr/bin/env python3
"""
测试分离部署的完整端到端脚本
启动三个进程并验证实际数据流通信
"""

import multiprocessing as mp
import time
import os
import sys
import torch
import numpy as np

def test_end_to_end_separated_deployment():
    """测试分离部署的完整端到端性能评估"""

    # 设置模型路径
    checkpoint_dir = "checkpoints/direct_polar_training_20251009_210057"
    sender_model_path = os.path.join(checkpoint_dir, "final_sender_model.pth")
    receiver_model_path = os.path.join(checkpoint_dir, "final_receiver_model.pth")
    quantizer_path = os.path.join(checkpoint_dir, "final_quantizer.pth")

    # 检查模型文件是否存在
    for path in [sender_model_path, receiver_model_path, quantizer_path]:
        if not os.path.exists(path):
            print(f"错误：模型文件不存在 {path}")
            return False

    # print("✓ 找到所有模型文件")

    # 创建进程间通信队列
    sender_to_channel = mp.Queue(maxsize=50)  # 更大的队列
    channel_to_receiver = mp.Queue(maxsize=50)

    # print("✓ 创建了通信队列")

    # 启动接收端进程
    # print("启动接收端进程...")
    receiver_process = mp.Process(
        target=run_receiver_process,
        args=(channel_to_receiver, receiver_model_path, checkpoint_dir)
    )
    receiver_process.start()

    # 等待接收端初始化
    time.sleep(3)

    # 启动信道进程
    # print("启动信道进程...")
    channel_process = mp.Process(
        target=run_channel_process,
        args=(sender_to_channel, channel_to_receiver)
    )
    channel_process.start()

    # 等待信道初始化
    time.sleep(3)

    # 启动发送端进程
    # print("启动发送端进程...")
    sender_process = mp.Process(
        target=run_sender_process,
        args=(sender_to_channel, sender_model_path, quantizer_path, checkpoint_dir)
    )
    sender_process.start()

    # 等待所有进程完成
    # print("等待进程完成...")
    sender_process.join(timeout=120)  # 更长的超时时间
    channel_process.join(timeout=120)
    receiver_process.join(timeout=120)

    # 检查进程状态
    processes_alive = []
    if sender_process.is_alive():
        processes_alive.append("发送端")
        sender_process.terminate()

    if channel_process.is_alive():
        processes_alive.append("信道")
        channel_process.terminate()

    if receiver_process.is_alive():
        processes_alive.append("接收端")
        receiver_process.terminate()

    if processes_alive:
        # print(f"警告：以下进程仍在运行: {', '.join(processes_alive)}")
        return False

    # print("✓ 端到端性能测试完成")
    return True

def run_receiver_process(input_queue, model_path, checkpoint_dir):
    """运行接收端进程"""
    try:
        # 改变工作目录
        os.chdir("e:\\PhD_work\\code\\uhdcode\\DHSCC")

        # 禁用打印
        import sys
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        # 导入接收端模块
        import receiver_process
        import sys

        # 设置命令行参数
        sys.argv = [
            'receiver_process.py',
            '--mode', 'test',
            '--max_batches', '10',  # 处理10个批次进行性能评估
            '--receiver_model_path', model_path,
            '--checkpoint_dir', checkpoint_dir,
            '--batch_size', '32'  # 更大的批次大小
        ]

        # 调用接收端的主函数
        receiver_process.receiver_process(
            receiver_process.create_receiver_args().parse_args(),
            input_queue  # from_channel_queue
        )

    except Exception as e:
        # 恢复stdout以打印错误
        sys.stdout = original_stdout
        print(f"接收端错误：{e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保恢复stdout
        sys.stdout = original_stdout

def run_channel_process(input_queue, output_queue):
    """运行信道进程"""
    try:
        # 改变工作目录
        os.chdir("e:\\PhD_work\\code\\uhdcode\\DHSCC")

        # 禁用打印
        import sys
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        # 导入信道模块
        import channel_process
        import sys

        # 设置命令行参数
        sys.argv = ['channel_process.py', '--channel_snr', '18', '--K', '128', '--N', '256', '--channel_model', 'awgn']

        # 调用信道的主函数
        channel_process.channel_process(
            channel_process.create_channel_args().parse_args(),
            input_queue,   # from_sender_queue
            output_queue   # to_receiver_queue
        )

    except Exception as e:
        # 恢复stdout以打印错误
        sys.stdout = original_stdout
        print(f"信道错误：{e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保恢复stdout
        sys.stdout = original_stdout

def run_sender_process(output_queue, model_path, quantizer_path, checkpoint_dir):
    """运行发送端进程"""
    try:
        # 改变工作目录
        os.chdir("e:\\PhD_work\\code\\uhdcode\\DHSCC")

        # 禁用打印
        import sys
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        # 导入发送端模块
        import sender_process
        import sys

        # 设置命令行参数
        sys.argv = [
            'sender_process.py',
            '--mode', 'test',
            '--max_batches', '10',  # 处理10个批次进行性能评估
            '--sender_model_path', model_path,
            '--quantizer_path', quantizer_path,
            '--checkpoint_dir', checkpoint_dir,
            '--batch_size', '32',  # 更大的批次大小
            '--group_to_use', '8'  # 使用8个组，与训练最终阶段一致
        ]

        # 调用发送端的主函数
        sender_process.sender_process(
            sender_process.create_sender_args().parse_args(),
            output_queue,  # to_channel_queue
            None  # from_channel_queue (not used in sender)
        )

    except Exception as e:
        # 恢复stdout以打印错误
        sys.stdout = original_stdout
        print(f"发送端错误：{e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保恢复stdout
        sys.stdout = original_stdout

if __name__ == "__main__":
    # print("开始端到端分离部署性能测试...")
    # print("这将测试实际的数据流通过三个进程，并评估分类性能")
    success = test_end_to_end_separated_deployment()
    if success:
        # print("✓ 分离部署端到端性能测试成功！")
        # print("检查上述输出以查看分类准确率")
        pass
    else:
        print("✗ 分离部署端到端性能测试失败")
        sys.exit(1)