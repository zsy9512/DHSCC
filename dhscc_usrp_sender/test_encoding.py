import torch
import numpy as np
import argparse
from sender.sender_model import SenderNet

# 设置随机种子，确保输入相同
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义args（基于代码默认）
args = argparse.Namespace()
args.intermediate_dim = 128
args.num_classes = 10

# 生成相同的随机输入
x = torch.randn(1, 1, 512).to(device)  # 假设输入形状 [batch, channels, time]
channel_noise = torch.randn(1, 1).to(device)

# 加载模型
model = SenderNet(args).to(device)
checkpoint = torch.load(r'E:\PhD_work\code\uhdcode\dhscc_usrp_sender\direct_polar_training_20251009_210057\final_sender_model.pth', map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# 运行编码
with torch.no_grad():
    final_features, sorted_importance, pre_task_logits, feature_shape, encoded_features = model(x, channel_noise)

# 打印编码结果（encoded_features）
print("dhscc_usrp_sender Encoded Features:")
print(encoded_features)