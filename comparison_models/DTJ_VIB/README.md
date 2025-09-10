# DTJ-VIB: Discrete Task-oriented Joint Source-Channel Coding for Vibration Classification

这是一个将DTJ（Discrete Task-oriented Joint Source-Channel Coding）应用于触觉振动信号分类的实现。该项目将512样本的1D振动信号转换为2x16x16伪图像，并使用DTJ的任务导向通信网络进行分类。

## 项目特点

- **振动信号处理**: 将512样本1D振动信号转换为2x16x16伪图像
- **任务导向通信**: 使用DTJ的联合信源信道编码进行端到端优化
- **数字调制**: 支持PSK和QAM调制方式
- **分类任务**: 针对10类触觉振动信号进行分类

## 网络架构

### 输入处理
```
WAV文件(512样本) → [2, 16, 16] 伪图像 → DTJ编码器
```

### 网络流程
```
伪图像 → CNN编码器 → 向量量化 → 调制 → AWGN信道 → 解调 → 分类器
```

## 数据集要求

数据集应遵循以下结构：
```
dataset/vibcls/
├── train.txt          # 训练集文件列表 "filename label"
├── test.txt           # 测试集文件列表 "filename label"  
├── clsmap.txt         # 类别映射 "index classname"
└── vibrodata/         # WAV文件目录
    ├── sample1.wav
    ├── sample2.wav
    └── ...
```

## 使用方法

### 训练
```bash
python main_vib.py --dataset VIB --num_embeddings 16 --epochs 200 --lr 1e-3 --lam 0.01
```

### 评估
```bash
python evaluate_vib.py --dataset VIB --name your_trained_model
```
python comparison_models/DTJ_VIB/evaluate_vib.py --dataset VIB \
  --name VIB-standard-num_e16-latent_d256-modpsk-psnr3.0-lam0.01 \
  --comprehensive
## 主要参数

- `latent_d`: 潜在特征维度 (默认: 256)
- `num_embeddings`: 码本大小 (默认: 16)
- `num_latent`: 潜在变量数量 (默认: 8)
- `mod`: 调制方式 ('psk' 或 'qam')
- `psnr`: 信道信噪比 (默认: 8.0)
- `lam`: 熵正则化权重 (默认: 1e-3)
## 网络架构细节

### DTJSCC_VIB网络
- **输入**: 2x16x16伪图像
- **编码器**: 基于ResNet的CNN编码器
- **量化器**: MaskAttentionSampler向量量化
- **解码器**: 分类头，输出10类概率

### 与原DTJ的差异
- 输入通道数: 3 → 2
- 输入尺寸: 32x32 → 16x16
- 输出类别: CIFAR10 → VIB分类
- 特征维度: 优化适配小尺寸输入

## 依赖要求

- PyTorch >= 1.8.0
- torchaudio
- tensorboardX
- numpy

## 引用

基于以下工作：
```
@article{xie2022robust,
  title={Robust Information Bottleneck for Task-Oriented Communication with Digital Modulation},
  author={Xie, Songjie and Wu, Youlong and Ma, Shuai and Ding, Ming and Shi, Yuanming and Tang, Mingjian},
  journal={arXiv preprint arXiv:2209.10382},
  year={2022}
}
