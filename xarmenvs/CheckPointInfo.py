
import torch

# 加载 checkpoint 文件
checkpoint = torch.load('/home/king/Isaac/LEAP_Hand_Sim/leapsim/runs/pretrained/nn/LeapHand.pth')

# 打印 checkpoint 的键
print(checkpoint.keys())

# 查看训练轮数（epoch）
'''
if 'epoch' in checkpoint:
    print(f"训练轮数: {checkpoint['epoch']}")
else:
    print("Checkpoint 文件中未找到训练轮数信息。")
'''

# 检查是否有 epoch 和 reward 信息
if 'epoch' in checkpoint:
    print(f"训练轮数 (epoch): {checkpoint['epoch']}")
if 'reward' in checkpoint:
    print(f"奖励值 (reward): {checkpoint['reward']}")

# 打印完整的 checkpoint 内容（可选）
print("完整的 checkpoint 内容:", checkpoint)

