import os
import importlib.util
import torch
import numpy as np
import json
import random
import time
import threading
from envs.env_wrapper import SimpleAdversaryWrapper as EnvWrapper
from server.gradient_manager import get_policy_lock, policy_metadata_lock

from torch.utils.tensorboard import SummaryWriter

import shutil

def write_tensorboard_logs(eval_records, log_dir="server/tb_logs"):
    # 清空旧日志目录，确保每次都是最新的点
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    writer = SummaryWriter(log_dir=log_dir)
    for record in eval_records:
        agent_id = record["agent_id"]
        policy_id = record["policy_id"]
        train_steps = record["train_steps"]
        avg_reward = record["avg_reward"]

        tag = f"{agent_id}/{policy_id}"
        writer.add_scalar(tag, avg_reward, global_step=train_steps)

    writer.close()
    print(f"[TENSORBOARD] Logs refreshed at {log_dir}")

def load_policy(policy_path, agent_id, policy_id, env_wrapper):
    lock = get_policy_lock(policy_id)
    with lock:
        config_path = os.path.join(policy_path, "agent.config")
        policy_file = os.path.join(policy_path, "policy.py")
        model_file = os.path.join(policy_path, "model.py")

    
        with open(config_path, 'r') as f:
            config = json.load(f)

        input_dim = config.get("observation_shape", 10)
        action_dim = config.get("action_shape", 5)
        lr = config.get("optimizer_config", {}).get("lr", 1e-3)

        # 动态加载 model.py
        spec_model = importlib.util.spec_from_file_location("Model", model_file)
        model_module = importlib.util.module_from_spec(spec_model)
        spec_model.loader.exec_module(model_module)
        Model = model_module.Model

        # 动态加载 policy.py
        spec_policy = importlib.util.spec_from_file_location("Policy", policy_file)
        policy_module = importlib.util.module_from_spec(spec_policy)
        spec_policy.loader.exec_module(policy_module)

        policy_instance = policy_module.Policy(input_dim, action_dim, lr, Model, env=env_wrapper.env, agent_id=agent_id)

        # 加载模型权重
        model_pth = os.path.join(policy_path, "model.pt")
        optim_pth = os.path.join(policy_path, "optimizer.pt")
        if os.path.exists(model_pth):
            policy_instance.load(model_pth, optim_pth)

        return policy_instance

def evaluate_policies(agent_policy_map, policy_id_map, env_wrapper, num_episodes=10):
    env = env_wrapper
    agent_rewards = {agent_id: [] for agent_id in agent_policy_map}

    # 加载所有策略
    policies = {
        agent_id: load_policy(path, agent_id, policy_id_map[agent_id], env_wrapper)
        for agent_id, path in agent_policy_map.items()
    }

    for episode in range(num_episodes):
        observations = env.reset()
        done = {agent_id: False for agent_id in agent_policy_map}
        trunc = {agent_id: False for agent_id in agent_policy_map}
        rewards_sum = {agent_id: 0.0 for agent_id in agent_policy_map}

        while not all(done[aid] or trunc[aid] for aid in agent_policy_map):
            actions = {
                agent_id: policies[agent_id].predict(observations[agent_id])
                for agent_id in agent_policy_map
            }
            observations, rewards, dones, truncations, _ = env.step(actions)
            if rewards == {}:
                break
            for agent_id in agent_policy_map:
                rewards_sum[agent_id] += rewards[agent_id]
                done[agent_id] = dones[agent_id]
                trunc[agent_id] = truncations[agent_id]

        for agent_id in agent_policy_map:
            agent_rewards[agent_id].append(rewards_sum[agent_id])

    # 计算平均奖励
    results = {
        agent_id: float(np.mean(agent_rewards[agent_id]))
        for agent_id in agent_policy_map
    }

    return results

def evaluate_from_metadata(metadata_path='server/policy_metadata.json', num_episodes=10, save_path='server/eval_results.json', env_wrapper=None):
    if env_wrapper is None:
        raise ValueError("env_wrapper must be passed in from the main thread to avoid threading issues.")

    agent_ids = env_wrapper.get_agent_ids()

    with policy_metadata_lock:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    # 加载已有评估结果
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            eval_records = json.load(f)
    else:
        eval_records = []

    # 按 agent_id 分组 policy
    agent_policy_dict = {}
    for policy_name, info in metadata.items():
        agent_id = info.get("agent_id")
        if agent_id:
            agent_policy_dict.setdefault(agent_id, []).append((policy_name, info))

    # 构造新的评估映射
    agent_policy_map = {}
    policy_id_map = {}
    for agent_id in agent_ids:
        if agent_id not in agent_policy_dict:
            print(f"[WARN] No policy available for {agent_id}, skipping.")
            continue
        policy_name, info = random.choice(agent_policy_dict[agent_id])
        policy_path = os.path.join("server/policies", policy_name)
        agent_policy_map[agent_id] = policy_path
        policy_id_map[agent_id] = info.get("policy_id", policy_name)

    # 执行评估
    new_results = evaluate_policies(agent_policy_map, policy_id_map, env_wrapper, num_episodes=num_episodes)

    # 更新记录
    def find_record(records, agent_id, policy_id, train_steps):
        for r in records:
            if r["agent_id"] == agent_id and r["policy_id"] == policy_id and r["train_steps"] == train_steps:
                return r
        return None
    
    for agent_id in agent_policy_map:
        policy_path = agent_policy_map[agent_id]
        policy_name = os.path.basename(policy_path)
        policy_id = policy_id_map[agent_id]
        train_steps = metadata[policy_name].get("train_steps", 0)
        avg_reward = new_results[agent_id]

        existing = find_record(eval_records, agent_id, policy_id, train_steps)
        if existing:
            count = existing.get("count", 1)
            old_avg = existing.get("avg_reward", 0.0)
            new_avg = (old_avg * count + avg_reward) / (count + 1)
            existing["avg_reward"] = new_avg
            existing["count"] = count + 1
        else:
            eval_records.append({
                "agent_id": agent_id,
                "policy_id": policy_id,
                "train_steps": train_steps,
                "avg_reward": avg_reward,
                "count": 1
            })

    # 保存更新结果
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(eval_records, f, indent=2)

    print(f"[EVAL] Evaluation results updated and saved to {save_path}")
    write_tensorboard_logs(eval_records)
    return eval_records

def start_evaluation_loop(env_wrapper, interval_seconds=60, num_episodes=10):
    def loop():
        while True:
            try:
                print(f"[EVAL THREAD] Running evaluation...")
                evaluate_from_metadata(num_episodes=num_episodes, env_wrapper=env_wrapper)
            except Exception as e:
                print(f"[EVAL THREAD] Error during evaluation: {e}")
            time.sleep(interval_seconds)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    print(f"[EVAL THREAD] Evaluation loop started (interval = {interval_seconds}s)")