# DTJ-VIB 使用指南

## 概述

DTJ-VIB (Discrete Task-oriented Joint Source-Channel Coding for Vibration classification) 是一个专门为触觉振动信号分类设计的深度学习系统。它将原始的 DTJ 框架适配到振动信号处理任务，实现了端到端的联合信源信道编码。

## 系统架构

### 核心组件

1. **信号处理器** (`TactileSignalProcessor`)
   - 将512维的1D振动信号转换为2x16x16的伪图像
   - 支持数据归一化和增强

2. **网络架构** (`DTJSCC_VIB`)
   - 标准版：3,062,538参数
   - 小型版：268,746参数
   - 支持向量量化和注意力机制

3. **调制解调** (`QAM/PSK`)
   - 支持16-QAM和16-PSK调制
   - 可配置信噪比(SNR)

4. **损失函数** (`RIBLoss`)
   - 健壮信息瓶颈损失
   - 平衡分类精度和编码效率

## 快速开始

### 1. 环境准备

```bash
# 激活Python环境
conda activate vibtoc

# 验证系统完整性
cd comparison_models/DTJ_VIB
python test_integration.py
```

### 2. 数据准备

确保数据集按以下结构组织：
```
dataset/vibcls/
├── train.txt          # 训练文件列表
├── test.txt           # 测试文件列表
├── clsmap.txt         # 类别映射
└── vibrodata/         # WAV音频文件
    ├── file1.wav
    ├── file2.wav
    └── ...
```

### 3. 训练模型

#### 基础训练命令
```bash
python main_vib.py \
    --dataset VIB \
    --model_type standard \
    --epochs 200 \
    --batch_size 64 \
    --lr 1e-3 \
    --mod psk \
    --psnr 8.0 \
    --lam 1e-3
```

#### 参数说明
- `--model_type`: `standard` (标准版) 或 `small` (小型版)
- `--epochs`: 训练轮数 (默认 200)
- `--batch_size`: 批大小 (默认 64)
- `--lr`: 学习率 (默认 1e-3)
- `--mod`: 调制方式 (`psk` 或 `qam`)
- `--psnr`: 信噪比 (默认 8.0 dB)
- `--lam`: 熵正则化权重 (默认 1e-3)

#### 小型模型快速训练
```bash
python main_vib.py \
    --model_type small \
    --latent_d 128 \
    --epochs 100 \
    --batch_size 32 \
    --eval_start_epoch 5
```

### 4. 模型评估

#### 单SNR评估
```bash
python evaluate_vib.py \
    --name "VIB-standard-num_e16-latent_d256-modpsk-psnr8.0-lam0.001" \
    --psnr 10.0
```

#### 多SNR综合评估
```bash
python evaluate_vib.py \
    --name "VIB-standard-num_e16-latent_d256-modpsk-psnr8.0-lam0.001" \
    --comprehensive \
    --save_csv
```

## 高级配置

### 网络架构选择

#### 标准版 (DTJSCC_VIB)
- **适用场景**: 高精度要求的应用
- **参数量**: ~300万
- **特点**: 更深的网络，更强的表征能力

#### 小型版 (DTJSCC_VIB_Small)
- **适用场景**: 资源受限的环境
- **参数量**: ~27万
- **特点**: 更少参数，更快训练速度

### 调制方式对比

| 调制方式 | 优势 | 适用场景 |
|---------|------|----------|
| PSK | 对相位噪声鲁棒性强 | 一般通信环境 |
| QAM | 频谱效率高 | 高质量信道 |

### 超参数调优建议

#### 学习率策略
```bash
# 阶梯衰减 (默认)
--lr 1e-3 --scheduler_step 80 --scheduler_gamma 0.5

# 余弦退火
--lr 1e-3 --scheduler cosine

# 预热训练
--lr 1e-4 --warmup_epochs 10
```

#### 正则化调节
```bash
# 强正则化 (追求压缩率)
--lam 1e-2

# 弱正则化 (追求精度)
--lam 1e-4

# 无正则化 (基线对比)
--lam 0.0
```

## 性能优化

### 数据加载优化
```bash
# 增加数据加载进程
--num_workers 8

# 减少I/O延迟
--pin_memory
```

### 内存优化
```bash
# 梯度累积 (有效批大小 = batch_size * accumulate_steps)
--batch_size 32 --accumulate_steps 2

# 混合精度训练
--mixed_precision
```

### GPU利用率优化
```bash
# 指定GPU
--device cuda:0

# 多GPU训练 (需要额外实现)
--device cuda --multi_gpu
```

## 常见问题与解决方案

### 1. 训练不收敛
**可能原因**: 学习率过高、数据不平衡、网络过深
**解决方案**:
```bash
# 降低学习率
--lr 5e-4

# 使用小型模型
--model_type small

# 增加权重衰减
--weight_decay 1e-3
```

### 2. 过拟合
**症状**: 训练精度高但测试精度低
**解决方案**:
```bash
# 增加正则化
--lam 1e-2

# 减少模型复杂度
--latent_d 128

# 早停
--patience 20
```

### 3. 欠拟合
**症状**: 训练和测试精度都较低
**解决方案**:
```bash
# 增加模型容量
--model_type standard --latent_d 512

# 降低正则化
--lam 1e-4

# 增加训练轮数
--epochs 400
```

### 4. 内存不足
**解决方案**:
```bash
# 减少批大小
--batch_size 16

# 使用小型模型
--model_type small

# 梯度检查点 (需要实现)
--gradient_checkpointing
```

## 实验记录模板

### 实验配置记录
```yaml
实验名称: DTJ_VIB_Experiment_001
日期: 2024-XX-XX
目标: 基线性能测试

模型配置:
  - 架构: standard
  - 潜在维度: 256
  - 量化码本大小: 16

训练配置:
  - 批大小: 64
  - 学习率: 1e-3
  - 训练轮数: 200
  - 调制方式: PSK
  - 信噪比: 8.0 dB
  - 正则化权重: 1e-3

结果:
  - 最佳测试精度: XX.X%
  - 最佳SNR: XX dB
  - 训练时间: XX小时
  - 模型大小: XX MB
```

## 结果分析

### 性能指标
- **Top-1 Accuracy**: 主要分类精度指标
- **Top-3 Accuracy**: 宽松分类精度指标
- **压缩率**: 量化前后数据大小比率
- **信道鲁棒性**: 不同SNR下的性能衰减

### 可视化分析
```bash
# 生成性能曲线
python plot_results.py --results evaluation_results/

# 分析注意力图
python analyze_attention.py --model_path trained_models/best.pt
```

## 贡献指南

### 添加新功能
1. 在相应模块中添加新功能
2. 编写单元测试
3. 更新集成测试
4. 更新文档

### 代码风格
- 遵循PEP 8规范
- 使用类型注解
- 添加详细的docstring
- 保持函数功能单一

### 提交流程
1. 创建功能分支
2. 实现功能并测试
3. 提交Pull Request
4. 代码审查和合并

## 许可证

本项目遵循原DTJ项目的许可证条款。
