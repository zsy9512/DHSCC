# Models Module

本模块包含振动信号分类的深度学习网络模型，实现基于重要性感知的无线传输和分类系统。

## 📋 目录结构

```
models/
├── README.md                    # 本文档
├── __init__.py                 # 模块初始化
├── sender_network.py           # 发送端网络
├── receiver_network.py         # 接收端网络  
├── wireless_channel.py         # 无线信道模拟
├── adaptive_group_selector.py  # 自适应组选择器
├── vibration_vlvfe_net.py      # 振动信号VLVFE网络
├── quantizer.py               # 量化器
├── core_vq.py                 # 矢量量化核心
├── vq.py                      # 矢量量化模块
└── __pycache__/               # Python缓存文件
```

## 🔄 数据流程概览

整个系统的数据流程如下：

```
原始振动信号 → 发送端网络 → 无线信道 → 接收端网络 → 分类结果
     ↓              ↓         ↓          ↓         ↓
  [B,1,T]    [B,C,T']→[B,D]  [B,D]   [B,D]→[B,C,T'] [B,Classes]
```

其中：
- `B`: 批次大小 (Batch Size)
- `T`: 原始信号时间长度
- `T'`: 特征提取后的时间长度
- `C`: 特征通道数 (通常512)
- `D`: 编码后的特征维度 (通常64)

## 🚀 发送端网络 (Sender Network)

### 主要组件

#### 1. SenderFeatureExtractor
- **功能**: 从原始振动信号中提取高维特征
- **输入**: `[batch, 1, seq_len]` 原始振动信号
- **输出**: `[batch, 512, seq_len//16]` 高维特征
- **架构**: 
  - 预处理层：1D卷积 + BatchNorm + ReLU
  - Layer 1：128通道卷积层 + 残差连接
  - Layer 2：256通道卷积层
  - Layer 3：512通道卷积层 + 残差连接

#### 2. SenderFeatureEncoder
- **功能**: 将高维特征压缩为低维表示
- **输入**: `[batch, 512, seq_len//16]` 高维特征
- **输出**: `[batch, 64]` 压缩特征 + 权重和偏置参数
- **处理流程**:
  1. 卷积降维：512 → 8通道
  2. 自适应池化：固定长度16
  3. 线性投影：128 → 64维
  4. 特征稀疏化和权重归一化

#### 3. JointImportanceEstimator_group
- **功能**: 基于信道条件和任务需求估计特征重要性
- **输入**: 编码特征 + 预分类结果
- **输出**: 最终重要性权重
- **核心算法**:
  - 胶囊网络分组处理
  - 信道重要性 × 任务重要性 = 最终重要性
  - 支持8组×16维的分组策略

### 数据流程

```python
# 发送端完整流程
def sender_forward(x, channel_noise):
    # 1. 特征提取
    extracted_features = feature_extractor(x)         # [B,1,T] → [B,512,T']
    
    # 2. 特征编码
    encoded_features, weights, bias = feature_encoder(extracted_features)  # [B,512,T'] → [B,64]
    
    # 3. 预分类 (用于任务感知重要性)
    pre_task_logits = pre_classifier(encoded_features) # [B,64] → [B,Classes]
    
    # 4. 重要性估计
    importance_weights = importance_estimator(encoded_features, pre_task_logits)  # [B,64]
    
    # 5. 特征排序和加权
    sorted_features = sort_by_importance(encoded_features, importance_weights)
    weighted_features = sorted_features * importance_weights
    
    # 6. 约束和稳定化
    final_features = tanh_constraint(weighted_features)  # [-1, 1]
    
    return final_features, importance_weights, pre_task_logits, feature_shape
```

## 📡 接收端网络 (Receiver Network)

### 主要组件

#### 1. ReceiverAdaptiveDecoder
- **功能**: 基于信道噪声条件自适应解码特征
- **输入**: `[batch, 64]` 接收特征 + 信道噪声水平
- **输出**: `[batch, 128]` 解码特征 (重塑为8×16)
- **关键机制**:
  - **噪声条件注意力**: 根据信道噪声动态调整特征权重
  - **特征细化**: 多层感知机进行特征优化
  - **自适应处理**: 不同噪声水平下的不同处理策略

#### 2. ReceiverFeatureReconstructor  
- **功能**: 将低维特征重构为高维特征
- **输入**: `[batch, 8, 16]` 解码后的低维特征
- **输出**: `[batch, 512, seq_len]` 重构的高维特征
- **方法**: 1D卷积 + BatchNorm + ReLU

#### 3. ReceiverClassifierModule
- **功能**: 对重构特征进行最终分类
- **输入**: `[batch, 512, seq_len]` 重构特征
- **输出**: `[batch, num_classes]` 分类概率
- **架构**:
  - 自适应池化统一序列长度
  - CNN特征提取器
  - 线性分类器

### 数据流程

```python
# 接收端完整流程
def receiver_forward(received_features, channel_noise, target_shape):
    # 1. 噪声自适应解码
    decoded_flat = adaptive_decoder(received_features, channel_noise)  # [B,64] → [B,128]
    
    # 2. 重塑为卷积格式
    decoded_reshaped = decoded_flat.view(batch_size, 8, 16)           # [B,128] → [B,8,16]
    
    # 3. 特征重构
    reconstructed_features = feature_reconstructor(decoded_reshaped)   # [B,8,16] → [B,512,T']
    
    # 4. 残差连接 (可选)
    if sender_feature_extractor is not None:
        reconstructed_features = sender_feature_extractor.apply_residual(reconstructed_features)
    
    # 5. 最终分类
    output = classifier_module(reconstructed_features)                 # [B,512,T'] → [B,Classes]
    
    return output
```

## 📶 无线信道模拟 (Wireless Channel)

### 支持的通信模式

1. **模式1**: 直接模拟传输 + 重要性掩码
2. **模式2**: 量化模拟传输 + 高斯噪声
3. **模式3**: 数字传输 (无信道编码)
4. **模式4**: 数字传输 + 信道编码

### 信道效应模拟

- **加性高斯白噪声 (AWGN)**
- **PSNR控制的信噪比**
- **重要性感知的特征传输**

## 🎯 自适应组选择器 (Adaptive Group Selector)

### 核心功能
- **动态组选择**: 根据信道条件自适应选择传输的特征组数
- **性能反馈**: 基于分类准确率的反馈学习
- **单调性约束**: 保证PSNR越高，使用的组数越多

### 算法流程
1. 维护每个PSNR级别的性能历史
2. 基于准确率阈值动态调整组数
3. 探索与利用的平衡策略

## 🔧 重要性估计 (Importance Estimation)

### JointImportanceEstimator_group

#### 设计理念
- **胶囊网络分组**: 将特征分为8组，每组16维
- **联合重要性**: 结合信道条件和任务需求
- **动态权重**: 根据当前信道状态调整特征重要性

#### 计算公式
```
最终重要性 = 信道重要性 × 任务重要性 × 动态调节因子
```

## 📊 关键特性

### 1. 端到端可训练
- 所有组件支持梯度反向传播
- 联合优化发送端和接收端
- 重要性估计与分类任务联合训练

### 2. 噪声自适应
- 接收端根据信道噪声动态调整处理策略
- 重要性估计考虑信道条件
- 多种通信模式适应不同场景

### 3. 特征压缩与重构
- 高效的特征压缩 (512维 → 64维)
- 智能的特征重构机制
- 残差连接保持特征完整性

### 4. 鲁棒性设计
- 数值稳定性约束 (tanh激活、clamp等)
- 正则化和dropout防止过拟合
- 多层验证和错误处理

## 🚀 使用示例

### 基本使用
```python
from models.sender_network import SenderNetwork
from models.receiver_network import ReceiverNetwork
from models.wireless_channel import WirelessChannelModule

# 初始化网络
sender = SenderNetwork(args)
receiver = ReceiverNetwork(args, sender.feature_extractor)
channel = WirelessChannelModule(args)

# 前向传播
sent_features, importance, pre_logits, shape = sender(input_signal, channel_noise)
received_features = channel(sent_features, channel_noise, psnr_value)
final_output = receiver(received_features, channel_noise, shape)
```

### 训练配置
```python
# 推荐的训练参数
args = {
    'intermediate_dim': 64,      # 中间特征维度
    'num_classes': 10,           # 分类类别数
    'comm_mode': 1,              # 通信模式
    'threshold': 0.5,            # 重要性阈值
}
```

## 📝 注意事项

1. **内存管理**: 特征提取会产生较大的中间张量，注意GPU内存使用
2. **数值稳定性**: 所有模型都包含数值稳定性保护，避免梯度爆炸
3. **模型保存**: 提供完整的状态保存和加载功能
4. **设备兼容**: 自动检测和使用GPU/CPU设备

## 🔧 依赖项

- PyTorch >= 1.8.0
- torchvision
- numpy
- wireless_comm模块 (本项目无线通信接口)

## 📚 参考文献

该模型实现基于深度学习的无线感知分类系统，结合了：
- 重要性感知特征传输
- 噪声自适应处理
- 端到端联合优化
- 胶囊网络特征分组

更多技术细节请参考项目文档和相关论文。
