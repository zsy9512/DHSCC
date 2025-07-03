# 无线通信接口 (Wireless Communication Interface)

## 概述

本接口文件 (`wireless_comm/interface.py`) 为不同的传输模式提供了统一的接口，整合了 `wireless_communication.py` 和 `wireless_split.py` 的功能，并完全基于模块化的 `system_integration` 架构。

## 主要特性

- **统一接口**：支持所有4种通信模式
- **模块化设计**：基于 `system_integration` 的参数配置
- **多信道支持**：AWGN、Rayleigh、UDP等
- **发送机/接收机分离**：支持分布式部署
- **量化索引转换**：支持索引到比特流的双向转换

## 支持的通信模式

### 模式1：直接模拟传输 + 重要性掩码
- 特征基于重要性权重进行剪枝
- 通过高斯噪声进行模拟传输

### 模式2：量化模拟传输
- 量化特征通过模拟信道传输
- 添加高斯噪声模拟信道效应

### 模式3：数字传输（无信道编码）
- 使用无编码的数字传输系统
- 基于 `WirelessCommunicationSystem(use_coding=False)`

### 模式4：数字传输（有信道编码）
- 使用极化码进行信道编码
- 基于 `WirelessCommunicationSystem(use_coding=True)`
- 支持自适应系统选择

## 核心类

### WirelessInterface
主要的无线通信接口类，提供：
- 索引到二进制的转换
- 不同模式的传输功能
- 发送机/接收机接口获取
- 传输统计信息

### WirelessSender / WirelessReceiver
分离的发送机和接收机接口，支持分布式部署。

## 配置系统

### 基础配置参数

```python
# 通信模式配置
comm_mode: int = 4              # 通信模式 (1-4)

# 编码参数
n: int = 256                    # 码长 (极化码需要2的幂)
k: int = 128                    # 信息位数
channel_type: str = "uplink"    # 5G极化码类型
decoder_type: str = "SC"        # 解码器类型 ('SC', 'SCL')
list_size: int = 8              # SCL解码器列表大小

# 调制参数
bits_per_symbol: int = 4        # 每符号比特数
constellation_type: str = "qam" # 星座类型 ("qam", "bpsk", "pam")
normalize: bool = True          # 星座归一化
demapping_method: str = "app"   # 解映射方法 ('app', 'maxlog')

# 信道参数
channel_model: str = "awgn"     # 信道模型 ("awgn", "rayleigh", "udp")
equalization_method: str = "mrc" # 均衡方法

# UDP参数（硬件传输）
udp_tx_host: str = "localhost"
udp_tx_port: int = 12345
udp_rx_host: str = "localhost"
udp_rx_port: int = 12346
udp_timeout: float = 10.0
udp_buffer_size: int = 8192
```

### 预定义配置场景

1. **AWGN信道配置** (`AWGNChannelConfig`)
   - 适用于理想信道条件
   - 使用16-QAM调制和SCL解码

2. **瑞利衰落信道配置** (`RayleighChannelConfig`)
   - 适用于移动衰落环境
   - 使用MMSE均衡和保守调制

3. **UDP信道配置** (`UDPChannelConfig`)
   - 适用于真实硬件传输
   - 网络配置和鲁棒性参数

4. **高吞吐量配置** (`HighThroughputConfig`)
   - 优化数据速率
   - 使用64-QAM和高码率

5. **低延迟配置** (`LowLatencyConfig`)
   - 优化传输延迟
   - 使用短码和快速解码

6. **鲁棒性配置** (`RobustConfig`)
   - 优化恶劣信道条件下的性能
   - 使用BPSK和低码率

## 使用示例

### 基本使用

```python
from wireless_comm.interface import create_wireless_interface
from wireless_comm.config import get_config_by_scenario

# 创建AWGN信道配置
config = get_config_by_scenario('awgn', comm_mode=4)

# 创建接口
interface = create_wireless_interface(config)

# 准备传输数据
indices = torch.randint(0, 16, (batch_size, num_groups, group_size))
result = interface.transmit_mode4(indices, snr_db=10.0, group_to_use=6)

print(f"BER: {result['bit_error_rate']:.6f}")
print(f"Index Error Rate: {result['index_error_rate']:.6f}")
```

### 与WirelessChannelModule集成

```python
from models.wireless_channel import WirelessChannelModule

# 配置参数
args.comm_mode = 4
args.N = 256
args.k = 128
args.channel_model = "awgn"

# 创建无线信道模块
channel_module = WirelessChannelModule(args)

# 在forward函数中使用
received_features, ber, transmit_mse, zeros_count = channel_module(
    features=features,
    channel_noise=noise_level,
    psnr_value=snr_db,
    quantized_features=quantized_features,
    indices=indices,
    group_to_use=group_to_use,
    quantizer_decode_fn=decode_function
)
```

### 分布式部署

```python
# 发送端
sender = interface.get_sender_interface()
prep_data = sender.prepare_data(indices, bits_per_index=4, group_to_use=6)
encoded_data = sender.encode_for_transmission(prep_data['binary_data'], prep_data['metadata'])

# 接收端
receiver = interface.get_receiver_interface()
decoded_binary = receiver.decode_from_transmission(received_data, metadata)
reconstructed_indices = receiver.reconstruct_indices(decoded_binary, metadata)
```

## 性能特性

### 信道编码性能
- **模式3 vs 模式4**：模式4通过极化码提供更好的错误保护
- **自适应系统**：根据传输组数自动选择合适的编码参数
- **回退机制**：编码传输失败时自动回退到无编码传输

### 调制性能
- **BPSK**：最鲁棒，适用于恶劣信道
- **QPSK**：平衡性能和鲁棒性
- **16-QAM**：标准配置，良好的性能/复杂度权衡
- **64-QAM**：高吞吐量，适用于良好信道

### 信道模型适配
- **AWGN**：理想信道，最佳性能
- **Rayleigh**：移动衰落，需要均衡和鲁棒配置
- **UDP**：真实硬件，考虑网络延迟和缓冲

## 扩展性

### 添加新的信道模型
1. 在 `config.py` 中定义新的配置类
2. 在 `system_integration.py` 中添加信道处理逻辑
3. 更新 `interface.py` 中的参数传递

### 添加新的调制方案
1. 在 `mapping.py` 中实现新的星座
2. 更新配置参数
3. 测试性能

### 支持新的编码方案
1. 在相应的编码模块中实现
2. 更新 `system_integration.py`
3. 配置新的参数选项

## 注意事项

1. **设备一致性**：确保所有张量在同一设备上（CPU/GPU）
2. **参数匹配**：码长n必须是2的幂（极化码要求）
3. **内存管理**：大批量数据时注意GPU内存使用
4. **错误处理**：包含完整的回退机制和错误报告

## 测试和验证

运行测试脚本验证功能：

```bash
# 基本功能测试
python simple_test.py

# 完整功能测试
python test_wireless_interface.py

# 配置对比
python -c "from wireless_comm.config import print_config_comparison; print_config_comparison()"
```
