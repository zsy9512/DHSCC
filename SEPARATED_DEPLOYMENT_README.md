# 分离式端到端通信系统部署指南

## 概述

本系统实现了分离式部署的端到端振动信号分类通信系统，包含三个独立运行的程序：
- 发送端程序 (sender_process.py)
- 信道程序 (channel_process.py) 
- 接收端程序 (receiver_process.py)

每个程序都可以独立启动，加载各自的训练模型权重，通过IPC队列进行通信。

## 系统架构

```
[发送端进程] --> IPC队列 --> [信道进程] --> IPC队列 --> [接收端进程]
     |              |              |              |              |
   加载发送端模型   |            调制解调       |            加载接收端模型
   量化器模型      |            (CPU)          |            分类器模型
   (GPU/CPU)       |                             |            (GPU/CPU)
```

## 部署逻辑

### 1. 训练阶段
首先运行完整的端到端训练：
```bash
python end_to_end_main.py --mode train
```
训练完成后，会在 `checkpoints/end_to_end_training_*/` 目录下生成：
- `final_sender_model.pth` - 发送端模型权重
- `final_receiver_model.pth` - 接收端模型权重  
- `final_quantizer.pth` - 量化器权重

### 2. 分离式部署测试

#### 方式一：同一台机器上的分离进程（推荐用于测试）

1. **启动接收端进程**（先启动，作为数据接收端）：
```bash
python receiver_process.py --receiver_model_path checkpoints/end_to_end_training_20251008_155040/final_receiver_model.pth --mode test
```

2. **启动信道进程**（中间处理信道模拟）：
```bash
python channel_process.py --communication_mode 4 --channel_snr 10.0 --channel_model rayleigh
```

3. **启动发送端进程**（最后启动，作为数据源）：
```bash
python sender_process.py --sender_model_path checkpoints/end_to_end_training_20251008_155040/final_sender_model.pth --quantizer_path checkpoints/end_to_end_training_20251008_155040/final_quantizer.pth --mode test --max_batches 5
```

#### 方式二：跨机器部署（生产环境）

对于跨机器部署，需要修改通信方式从multiprocessing.Queue改为网络socket：

1. 修改进程间通信代码，将Queue替换为socket连接
2. 在每台机器上部署对应的程序
3. 配置网络地址和端口进行通信

### 3. 测试流程

1. **数据流向**：
   - 发送端加载振动数据 → 提取特征 → 量化 → 转换为二进制比特 → 发送给信道进程
   - 信道进程接收比特 → 极化编码 → 调制 → 信道模拟 → 解调 → 解码 → 发送给接收端进程
   - 接收端接收比特 → 重建特征 → 分类预测 → 计算准确率

2. **启动顺序**：
   - 必须按照：接收端 → 信道 → 发送端的顺序启动
   - 这是因为下游进程需要先准备好接收数据

3. **参数配置**：
   - `--communication_mode 4`: 使用数字传输+信道编码
   - `--channel_snr`: 设置信道SNR (dB)
   - `--max_batches`: 限制测试批次数（可选）

### 4. 输出结果

每个进程会显示：
- 发送端：处理进度、发送的比特数
- 信道：接收/发送的比特数、BER计算
- 接收端：分类准确率、损失值

最终接收端会输出总体测试准确率。

## 注意事项

1. **GPU/CPU兼容性**：发送端和接收端使用GPU（如果可用），信道进程强制使用CPU
2. **模型路径**：确保提供正确的模型权重文件路径
3. **数据同步**：三个进程通过IPC队列同步，确保数据按批次正确传递
4. **错误处理**：如果某个进程异常退出，其他进程会超时并自动终止

## 故障排除

1. **参数错误**：确保使用正确的参数名
   - `--mode test` (不是 `--test`)
   - `--max_batches 5` (不是 `--max_batch 5`)

2. **模型加载失败**：检查pth文件路径是否正确
   - 使用绝对路径或确保相对路径正确
   - 确认文件存在：`ls checkpoints/end_to_end_training_*/final_*.pth`

3. **进程通信失败**：确保启动顺序正确
   - 接收端 → 信道 → 发送端
   - 检查队列大小设置（默认maxsize=10）

4. **CUDA错误**：确保GPU内存充足，或设置环境变量`CUDA_VISIBLE_DEVICES`

5. **导入错误**：如果遇到模块导入问题，确保Python路径设置正确

## 测试结果

✅ **分离部署测试成功！**

运行 `python test_separated_deployment.py` 测试结果：

```
✓ 找到所有模型文件
✓ 创建了通信队列
✓ 三个进程启动成功
✓ 数据流正确传递：发送端 → 信道 → 接收端
✓ IPC通信正常工作
✓ GPU/CPU数据转换正确
✓ 端到端处理完成
```

### 关键修复

1. **参数解析**：添加了 `--mode` 和 `--max_batches` 参数
2. **导入路径**：修复了相对导入问题
3. **数据维度匹配**：调整信道编码参数 (K=128, N=256)
4. **Demapper调用**：添加了必需的 `noise_var` 参数

### 性能特点

- **GPU优化**：发送端和接收端使用GPU，信道进程使用CPU
- **IPC通信**：使用多进程队列进行进程间通信
- **模块化设计**：三个独立进程，可分别部署和扩展

## 修复记录

- 修复了参数解析器缺少 `--mode` 参数的问题
- 修复了 `receiver/polar_decoding.py` 和 `channel/polar_decoding.py` 中的导入路径错误
- 更新了模型加载逻辑以正确加载分离的pth文件</content>
<parameter name="filePath">e:\PhD_work\code\uhdcode\DHSCC\SEPARATED_DEPLOYMENT_README.md