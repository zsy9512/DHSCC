import torch
import numpy as np

# 设置固定随机种子确保可重现
torch.manual_seed(42)
np.random.seed(42)

# 生成5个测试样本（与训练数据分布相似）
batch_size = 5
seq_len = 512
num_classes = 10

# 生成振动信号数据（模拟真实的振动数据分布）
test_data = torch.randn(batch_size, 1, seq_len) * 1000  # 放大以匹配实际数据范围
test_targets = torch.randint(0, num_classes, (batch_size,))  # 随机标签 0-9

# 保存到文件
data_dict = {
    'data': test_data,
    'targets': test_targets,
    'batch_size': batch_size,
    'seq_len': seq_len
}
torch.save(data_dict, 'fixed_test_samples.pt')
print("Fixed test samples saved to fixed_test_samples.pt")
print(f"Data shape: {test_data.shape}, Targets: {test_targets}")