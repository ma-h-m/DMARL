import torch
import os

def save_all_gradients(all_gradients = None, save_dir="gradients_epoch"):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 保存梯度到一个 .pt 文件
    filepath = os.path.join(save_dir, f"all_agents.pt")
    torch.save(all_gradients, filepath)

    print(f"All gradients saved at {filepath}")
    return filepath

def save_gradient(all_gradients = None, save_dir="gradients_epoch"):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 保存梯度到一个 .pt 文件
    filepath = os.path.join(save_dir, f"agent.pt")
    torch.save(all_gradients, filepath)

    print(f"All gradients saved at {filepath}")
    return filepath