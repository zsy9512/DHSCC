# 该脚本可自定义 adaptive_selector_state 的 group_choices 并保存为新文件
import torch

# ====== 用户自定义 group_choices 列表（长度需与原始一致） ======
# 例如：group_choices = [4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 8, 8, 8, 8, 8, 8, 8]
#group_choices = [3, 3, 3,3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8, 8, 8, 8,8,8,8]  # <-- 在此处填写自定义 group_choices 列表，如 [4, 4, ...]
group_choices = [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4]
ckpt_path = "/home/zsy/workplace/DHSCC/checkpoints/training_20250728_170939/final/best_final_model.pth"
new_ckpt_path = "/home/zsy/workplace/DHSCC/checkpoints/training_20250728_170939/final/best_final_model.pth"
checkpoint = torch.load(ckpt_path, map_location="cpu")
adaptive_selector_state = checkpoint.get('adaptive_selector_state', None)
if adaptive_selector_state is not None:
    if group_choices is not None:
        if len(group_choices) != len(adaptive_selector_state['group_choices']):
            print(f"自定义 group_choices 长度({len(group_choices)})与原始({len(adaptive_selector_state['group_choices'])})不符，未做修改。")
        else:
            adaptive_selector_state['group_choices'] = group_choices
            checkpoint['adaptive_selector_state'] = adaptive_selector_state
            torch.save(checkpoint, new_ckpt_path)
            print("已用自定义 group_choices 保存为新文件:", new_ckpt_path)
    else:
        print("请在脚本中填写自定义 group_choices 列表！")
else:
    print("未找到自适应分组策略")
