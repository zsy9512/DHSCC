# 无线信道模型配置说明

## 概述

DHSCC系统支持多种无线信道模型，用于模拟不同的传输环境。信道模型的配置只在数字通信模式（模式3和4）下生效。

## 信道模型类型

### 1. AWGN信道 (`--channel_model awgn`)
- **描述**: 加性高斯白噪声信道
- **特点**: 只添加高斯噪声，没有衰落
- **适用场景**: 理想传输环境，基准测试
- **均衡方法**: 不需要均衡（identity）

### 2. Rayleigh信道 (`--channel_model rayleigh`)
- **描述**: 瑞利衰落信道
- **特点**: 包含随机衰落系数和高斯噪声
- **适用场景**: 移动通信环境，多径传播
- **均衡方法**: 支持多种均衡算法
  - `mrc`: 最大比合并（默认）
  - `zf`: 迫零均衡
  - `mmse`: 最小均方误差均衡

### 3. UDP信道 (`--channel_model udp`)
- **描述**: 通过UDP网络传输到真实硬件
- **特点**: 真实的无线传输环境
- **适用场景**: 实际部署，硬件在环测试
- **配置参数**:
  - `--udp_tx_host`: 发送端IP地址
  - `--udp_tx_port`: 发送端端口号
  - `--udp_rx_host`: 接收端IP地址
  - `--udp_rx_port`: 接收端端口号
  - `--udp_timeout`: 超时时间（秒）
  - `--udp_buffer_size`: 缓冲区大小

## 通信模式与信道模型的关系

### 模式1和2（模拟传输）
- **信道处理**: 直接在特征级别添加噪声
- **信道模型配置**: 不需要，参数被忽略
- **噪声添加**: 在`WirelessChannelModule`中直接处理

### 模式3和4（数字传输）
- **信道处理**: 通过`WirelessInterface`进行数字信号处理
- **信道模型配置**: 必需，通过`create_wireless_interface(args)`传递
- **信号处理流程**:
  1. 量化索引 → 二进制数据
  2. 信道编码（仅模式4）
  3. 调制
  4. 信道传输（AWGN/Rayleigh/UDP）
  5. 解调
  6. 信道解码（仅模式4）
  7. 二进制数据 → 量化索引

## 参数传递机制

### dhscc_main.py 中的参数定义
```python
# 信道模型基本参数
--channel_model: 信道类型选择
--equalization_method: 均衡方法选择

# 信道编码参数（模式4）
--N: Polar码码长
--k: 信息比特数
--channel_type: 5G信道类型
--decoder_type: 解码器类型

# 调制参数
--constellation_type: 星座图类型
--bits_per_symbol: 每符号比特数
--demapping_method: 软判决方法

# UDP信道参数
--udp_tx_host, --udp_tx_port: 发送端配置
--udp_rx_host, --udp_rx_port: 接收端配置
--udp_timeout: 超时配置
```

### 参数传递路径
```
dhscc_main.py (args) 
    ↓
models/wireless_channel.py (WirelessChannelModule.__init__)
    ↓ (仅模式3,4)
wireless_comm/interface.py (create_wireless_interface)
    ↓
wireless_comm/interface.py (WirelessInterface.__init__)
    ↓
wireless_comm/system_integration.py (WirelessCommunicationSystem)
```

### getattr函数说明
```python
getattr(args, 'channel_model', 'awgn')
```
- 第1个参数: `args` 对象
- 第2个参数: `'channel_model'` 属性名
- 第3个参数: `'awgn'` 默认值（当args中没有此属性时使用）

**重要**: 现在所有参数都会使用dhscc_main.py中定义的值，默认值仅在参数未定义时使用。

## 使用示例

### AWGN信道测试
```bash
python dhscc_main.py --communication_mode 3 --channel_model awgn --snr 10
```

### Rayleigh衰落信道测试
```bash
python dhscc_main.py --communication_mode 4 --channel_model rayleigh --equalization_method mrc --snr 5
```

### UDP硬件测试
```bash
python dhscc_main.py --communication_mode 4 --channel_model udp \
    --udp_tx_host 192.168.1.100 --udp_tx_port 12345 \
    --udp_rx_host 192.168.1.101 --udp_rx_port 12346
```

## 注意事项

1. **模式选择**: 信道模型配置仅在通信模式3和4下生效
2. **参数一致性**: 确保所有相关参数都在dhscc_main.py中正确定义
3. **硬件要求**: UDP模式需要相应的硬件设备支持
4. **性能影响**: 不同信道模型的计算复杂度不同，影响训练/测试速度
