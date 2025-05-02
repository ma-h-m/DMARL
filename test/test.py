import os
import importlib.util
import random
import torch
import numpy as np
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from communication.socket_handler import  *
# from gradient_manager import *

from envs.env_wrapper import SimpleAdversaryWrapper as EnvWrapper
import time
from envs.env_wrapper import SimpleAdversaryWrapper as EnvWrapper

def load_policy(policy_path, agent_id, env_wrapper):
    # 动态加载 model.py 和 policy.py
    config_path = os.path.join(policy_path, "agent.config")
    policy_file = os.path.join(policy_path, "policy.py")
    model_file = os.path.join(policy_path, "model.py")

    with open(config_path, 'r') as f:
        config = json.load(f)

    input_dim = config.get("observation_shape", 10)
    action_dim = config.get("action_shape", 5)
    lr = config.get("optimizer_config", {}).get("lr", 1e-3)

    spec_model = importlib.util.spec_from_file_location("Model", model_file)
    model_module = importlib.util.module_from_spec(spec_model)
    spec_model.loader.exec_module(model_module)
    Model = model_module.Model

    spec_policy = importlib.util.spec_from_file_location("Policy", policy_file)
    policy_module = importlib.util.module_from_spec(spec_policy)
    spec_policy.loader.exec_module(policy_module)

    policy_instance = policy_module.Policy(input_dim, action_dim, lr, Model, env=env_wrapper.env, agent_id=agent_id)

    # Load weights
    model_pth = os.path.join(policy_path, "model.pt")
    optim_pth = os.path.join(policy_path, "optimizer.pt")
    if os.path.exists(model_pth):
        policy_instance.load(model_pth, optim_pth)

    return policy_instance
import tqdm
import matplotlib.pyplot as plt

def test_policies(env, policies_root='test/policies', num_trials=10, num_episodes_per_trial=2):
    agent_ids = env.get_agent_ids()

    agent_policies = {}
    for agent_id in agent_ids:
        agent_dir = os.path.join(policies_root, agent_id)
        if not os.path.exists(agent_dir):
            print(f"[WARN] No directory for agent {agent_id}, skipping.")
            continue
        policy_ids = [name for name in os.listdir(agent_dir) if os.path.isdir(os.path.join(agent_dir, name))]
        agent_policies[agent_id] = policy_ids

    policy_rewards = {agent_id: {policy_id: [] for policy_id in policy_ids} for agent_id, policy_ids in agent_policies.items()}

    # 加 tqdm 进度条
    for trial in tqdm.trange(num_trials, desc="Testing policies"):
        for episode in range(num_episodes_per_trial):
            selected_policies = {}
            for agent_id in agent_ids:
                if agent_id not in agent_policies or not agent_policies[agent_id]:
                    continue
                policy_id = random.choice(agent_policies[agent_id])
                policy_path = os.path.join(policies_root, agent_id, policy_id)
                selected_policies[agent_id] = (policy_id, load_policy(policy_path, agent_id, env))

            observations = env.reset()
            done = {agent_id: False for agent_id in selected_policies}
            trunc = {agent_id: False for agent_id in selected_policies}
            rewards_sum = {agent_id: 0.0 for agent_id in selected_policies}

            while not all(done[aid] or trunc[aid] for aid in selected_policies):
                actions = {
                    agent_id: policy.predict(observations[agent_id])
                    for agent_id, (policy_id, policy) in selected_policies.items()
                }
                observations, rewards, dones, truncations, _ = env.step(actions)
                if rewards == {}:
                    break
                for agent_id in selected_policies:
                    rewards_sum[agent_id] += rewards[agent_id]
                    done[agent_id] = dones[agent_id]
                    trunc[agent_id] = truncations[agent_id]

            for agent_id, (policy_id, _) in selected_policies.items():
                policy_rewards[agent_id][policy_id].append(rewards_sum[agent_id])

    # 计算平均和标准差
    policy_avg_rewards = {}
    policy_std_rewards = {}
    for agent_id, policy_dict in policy_rewards.items():
        policy_avg_rewards[agent_id] = {}
        policy_std_rewards[agent_id] = {}
        for policy_id, rewards in policy_dict.items():
            if rewards:
                policy_avg_rewards[agent_id][policy_id] = np.mean(rewards)
                policy_std_rewards[agent_id][policy_id] = np.std(rewards)
            else:
                policy_avg_rewards[agent_id][policy_id] = 0.0
                policy_std_rewards[agent_id][policy_id] = 0.0

    # 打印结果
    print("[TEST] Final Policy Scores:")
    for agent_id, policy_dict in policy_avg_rewards.items():
        for policy_id, avg_reward in policy_dict.items():
            print(f"Agent {agent_id} - Policy {policy_id}: Avg Reward = {avg_reward:.2f}")

    # 保存图
    save_policy_score_plot(policy_avg_rewards, policy_std_rewards)

    return policy_avg_rewards
def save_policy_score_plot(avg_rewards, std_rewards, save_path='test/policy_scores.png'):
    """点图 + error bar，适合负reward，清爽表达"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    all_labels = []
    all_avg = []
    all_std = []

    for agent_id in avg_rewards:
        if agent_id != "agent_0":
            continue

        # 对 policy_id 进行排序
        sorted_policy_ids = sorted(avg_rewards[agent_id].keys())

        for policy_id in sorted_policy_ids:
            label = f"{agent_id}/{policy_id}".replace('_', ' ')
            all_labels.append(label)
            all_avg.append(avg_rewards[agent_id][policy_id])
            all_std.append(std_rewards[agent_id][policy_id])

    x = np.arange(len(all_labels))

    # 画点图 + error bar
    ax.errorbar(
        x, all_avg, yerr=all_std,
        fmt='o', capsize=5, markersize=6, linestyle='None',
        color='steelblue', ecolor='gray', elinewidth=1
    )

    # 横坐标
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=9)

    # 自动调整纵坐标范围
    y_min = min(all_avg) - max(all_std) - 0.05 * abs(min(all_avg))
    y_max = max(all_avg) + max(all_std) + 0.05 * abs(max(all_avg))
    ax.set_ylim(y_min, y_max)

    ax.set_ylabel('Average Reward')
    ax.set_title('Policy Test Results (Avg Reward ± Std)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[PLOT] Policy score plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    # 初始化环境
    env = EnvWrapper()
    env.reset()

    # 设定参数
    policies_root = 'test/policies'  # 测试 policies 的根目录
    num_trials = 10                 # 试验次数
    num_episodes_per_trial = 2       # 每个 trial 中玩几局

    # 测试 policies，内部会跑 tqdm 进度条，并自动保存图
    scores = test_policies(
        env,
        policies_root=policies_root,
        num_trials=num_trials,
        num_episodes_per_trial=num_episodes_per_trial
    )

    print("[TEST] Scores:", scores)