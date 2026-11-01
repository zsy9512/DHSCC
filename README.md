# DHSCC: Deep Hybrid Semantic-Channel Coding

本代码库 **DHSCC (Deep Hybrid Semantic-Channel Coding)** 是以下论文的代码实现/配套代码：

Shengyu Zhang, Qi Zhang, Liuchang Yang, Yan Wang, and Tao Jiang. **Group-Level Feature-Aware Hybrid Coding for Adaptive Task-Oriented Tactile Communication**. *IEEE Transactions on Cognitive Communications and Networking*, 2026, 12: 9460–9473.

## 概述

DHSCC (Deep Hybrid Semantic-Channel Coding) 是一个深度学习框架，用于振动信号分类任务中的语义-信道混合编码。该系统通过端到端的训练策略，实现了在无线信道条件下的高效特征传输和分类性能优化。

## 主要功能

### 🎯 核心特性
- **三阶段训练策略**: 预训练(Pretrain) → 选择器训练(Selector) → 联合训练(Final)
- **多种通信模式**: 支持4种不同的传输模式(1-4)，从基线到完整信道编码
- **自适应组选择**: 基于SNR条件的智能特征组选择策略
- **端到端优化**: 从特征提取到分类的联合优化
- **多信道模型**: AWGN、Rayleigh、UDP等多种信道模拟

### 📊 支持的通信模式
- **模式1**: 模拟传输(直接高斯噪声)
- **模式2**: 模拟传输+向量量化
- **模式3**: 数字传输(无信道编码)
- **模式4**: 数字传输+信道编码

## 快速开始

### 训练模式

#### 基础训练
```bash
# 使用默认参数进行训练
python dhscc_main.py --mode train --epochs 300 --communication_mode 4

# 自定义三阶段训练参数
python dhscc_main.py --mode train \
    --epochs 200 \
    --pretrain_epochs 80 \
    --selector_epochs 150 \
    --communication_mode 4 \
    --snr 3.0
```
python dhscc_main.py --mode train \
    --epochs 20 \
    --pretrain_epochs 10 \
    --selector_epochs 15 \
    --communication_mode 4 \
    --snr 3.0
#### 高级训练配置
```bash
# 使用不同信道模型和编码参数
python dhscc_main.py --mode train \
    --epochs 300 \
    --communication_mode 4 \
    --channel_model rayleigh \
    --equalization_method mmse \
    --N 512 --k 256 \
    --snr 2.0 \
    --batch_size 64
```

### 测试模式

#### 单模型测试
```bash
# 测试训练好的模型
python dhscc_main.py --mode test \
    --model_path checkpoints_path \
    --communication_mode 4 \
    --snr_min 0.0 --snr_max 5.0 --test_snr_steps 11
```

#### 多模式对比测试
```bash
# 测试所有通信模式
python dhscc_main.py --mode test \
    --model_path checkpoints/your_model/final/best_final_model.pth \
    --test_all_modes \
    --snr_min -2.0 --snr_max 8.0 --test_snr_steps 21
```

## 参数说明

### 🔧 核心训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 300 | 总训练轮数 |
| `--pretrain_epochs` | 100 | 预训练阶段轮数 |
| `--selector_epochs` | 100 | 选择器训练阶段轮数 |
| `--final_epochs` | 100 | 联合训练阶段轮数 |
| `--batch_size` | 128 | 训练批次大小 |
| `--lr` | 1e-3 | 学习率 |
| `--intermediate_dim` | 128 | 中间特征维度 |

### 📡 通信系统参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--communication_mode` | 4 | 通信模式(0-4) |
| `--snr` | 3.0 | 训练SNR (dB) |
| `--snr_min` | 0.0 | 测试最小SNR (dB) |
| `--snr_max` | 5.0 | 测试最大SNR (dB) |
| `--test_snr_steps` | 11 | 测试SNR步数 |

### 🔗 信道编码参数 (模式4)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--N` | 256 | Polar码长度 |
| `--k` | 128 | 信息比特数 |
| `--channel_type` | uplink | 5G信道类型 |
| `--decoder_type` | SC | 解码器类型 |
| `--constellation_type` | qam | 调制方式 |

### 🌊 信道模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--channel_model` | awgn | 信道模型(awgn/rayleigh/udp) |
| `--equalization_method` | zf | 均衡方法(identity/zf/mrc/mmse) |

### 🎛️ 自适应选择器参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--selector_snr_bins` | 11 | SNR离散化数量 |
| `--selector_algorithm` | heuristic | 选择器算法 |
| `--max_groups` | 8 | 最大组数 |

### 📁 路径参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | dataset/vibcls/vibrodata | 数据目录 |
| `--checkpoint_dir` | ./checkpoints | 模型保存目录 |
| `--results_dir` | ./results | 结果保存目录 |
| `--model_path` | - | 测试时的模型路径 |

## 输出文件

### 训练输出
- `checkpoints/training_YYYYMMDD_HHMMSS/`: 训练检查点目录
  - `pretrain/`: 预训练阶段模型
  - `selector/`: 选择器训练阶段模型  
  - `final/`: 联合训练阶段模型
  - `logs/`: 训练日志
  - `plots/`: 训练曲线图

### 测试输出
- `results/`: 测试结果目录
  - `raw_test_results.pkl`: 原始测试数据
  - `analysis_results.pkl`: 分析结果
  - `complete_results.pkl`: 完整结果
  - `performance_curves.png`: 性能曲线图

## 系统架构

### 三阶段训练流程
1. **预训练阶段** (0 ~ pretrain_epochs)
   - 训练基础分类器和发送端网络
   - 不使用无线信道，focus on特征学习

2. **选择器训练阶段** (pretrain_epochs ~ selector_epochs)
   - 冻结发送端权重，训练自适应组选择器
   - 学习SNR-group_to_use映射策略

3. **联合训练阶段** (selector_epochs ~ epochs)
   - 解冻所有权重，端到端联合优化
   - 应用单调性约束到组选择策略

### 主要组件
- **发送端网络**: 特征提取、重要性评估、特征编码
- **无线通信模块**: 信道编码、调制、信道模拟、解调、信道解码
- **接收端网络**: 特征重建、分类器
- **自适应组选择器**: 基于SNR的智能特征组选择

## 依赖环境

- Python 3.8+
- PyTorch 1.9+
- NumPy, SciPy
- matplotlib
- tqdm

## 注意事项

1. **数据准备**: 确保数据文件路径正确设置
2. **GPU内存**: 建议使用至少8GB显存的GPU
3. **磁盘空间**: 训练过程会产生大量检查点文件
4. **参数兼容性**: 不同通信模式需要相应的参数配置

## 联系方式

如有问题，请通过issue或email联系开发团队。
shenguyu@hust.edu.cn


如果您在研究工作中使用本代码库、复现本文实验或基于本文方法进行开发，请引用上述论文。推荐使用以下 BibTeX：

```bibtex
@article{zhang2026group,
  author  = {Zhang, Shengyu and Zhang, Qi and Yang, Liuchang and Wang, Yan and Jiang, Tao},
  title   = {Group-Level Feature-Aware Hybrid Coding for Adaptive Task-Oriented Tactile Communication},
  journal = {IEEE Transactions on Cognitive Communications and Networking},
  year    = {2026},
  volume  = {12},
  pages   = {9460--9473}
}
