import os
import json
import matplotlib.pyplot as plt
import numpy as np

def moving_average_full(values, window_size):
    """改进版移动平均：前面的也算，窗口不足就取已有点平均"""
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window_size + 1)
        smoothed.append(np.mean(values[start:i+1]))
    return np.array(smoothed)

def moving_std_full(values, window_size):
    """改进版滑动标准差：前面的也算"""
    stds = []
    for i in range(len(values)):
        start = max(0, i - window_size + 1)
        stds.append(np.std(values[start:i+1]))
    return np.array(stds)

root_dir = 'plot'  # 根目录
method_data = {}
smooth_window = 20  # smoothing窗口大小
min_points_for_smoothing = 30  # 少于这个点数不做平滑

# 收集数据
for method in os.listdir(root_dir):
    method_path = os.path.join(root_dir, method, 'eval_results.json')
    if os.path.isfile(method_path):
        with open(method_path, 'r') as f:
            data_list = json.load(f)  # 是一个 list
            for data in data_list:
                if data.get('agent_id') == 'agent_1':
                    if method not in method_data:
                        method_data[method] = {'train_steps': [], 'avg_reward': []}
                    method_data[method]['train_steps'].append(data['train_steps'])
                    method_data[method]['avg_reward'].append(data['avg_reward'])

# 画图
plt.figure(figsize=(10, 7))
for method in sorted(method_data.keys()):
    values = method_data[method]
    steps = np.array(values['train_steps'])
    rewards = np.array(values['avg_reward'])

    # 按 steps 排序
    sorted_indices = np.argsort(steps)
    steps_sorted = steps[sorted_indices]
    rewards_sorted = rewards[sorted_indices]

    print(f"[{method}] Points: {len(rewards_sorted)}")

    if len(rewards_sorted) >= min_points_for_smoothing:
        # 进行平滑
        rewards_smoothed = moving_average_full(rewards_sorted, smooth_window)
        rewards_std = moving_std_full(rewards_sorted, smooth_window)

        # 步数不变
        steps_smoothed = steps_sorted

        # 主曲线
        plt.plot(steps_smoothed, rewards_smoothed, label=method.replace('_', ' '), linewidth=2)

        # 标准差区域
        plt.fill_between(
            steps_smoothed,
            rewards_smoothed - rewards_std,
            rewards_smoothed + rewards_std,
            alpha=0.3
        )
    else:
        # 数据点太少，不做平滑，直接画原始曲线
        plt.plot(steps_sorted, rewards_sorted, label=method.replace('_', ' '), linewidth=2, linestyle='--')
        print(f"[{method}] Not enough points for smoothing, using raw curve.")

plt.xlabel('Train Steps')
plt.ylabel('Average Reward')
plt.title('Policy_1 Evaluation Results (Smart Smoothed)')
plt.legend()
plt.grid(True)

# 保证 plot 目录存在
os.makedirs('plot', exist_ok=True)
plt.savefig('plot/output.png', dpi=300)  # 保存成高清图片
print("Plot saved as plot/output.png")