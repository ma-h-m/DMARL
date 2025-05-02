import os
import shutil
import sys
import os
import torch
from tianshou.data import Batch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.env_wrapper import SimpleAdversaryWrapper as EnvWrapper
def setup_temp_policies(thread_id: int, path_prefix: str = '', remove = True):
    """
    创建 temp_files/thread_{i}/policies 目录，
    
    参数:
        thread_id (int): 当前线程编号，例如 1 表示 thread_1
    """
    # 定义目标路径
    temp_thread_path = os.path.join(path_prefix ,"temp_files", f"thread_{thread_id}")

    if os.path.exists(temp_thread_path) and remove == True:
        print(f"Removing existing temp_files/thread_{thread_id} directory")
        shutil.rmtree(temp_thread_path)
    os.makedirs(temp_thread_path, exist_ok=True)

    policies_dest_path = os.path.join(temp_thread_path, "policies")


    # new_policy_src_path = os.path.join(path_prefix, "new_policy")

    # # 创建 temp_files/thread_{i}/policies 目录
    # os.makedirs(policies_dest_path, exist_ok=True)

    # # 遍历 new_policy 下所有 policy_*
    # for policy_name in os.listdir(new_policy_src_path):
    #     src_path = os.path.join(new_policy_src_path, policy_name)
    #     dst_path = os.path.join(policies_dest_path, policy_name)

    #     # 确保是文件夹
    #     if os.path.isdir(src_path):
    #         # 移动整个 policy 文件夹
    #         # shutil.move(src_path, dst_path)
    #         shutil.copytree(src_path, dst_path)
    #         print(f"Copied {src_path} -> {dst_path}")

    # print(f"All policies Copied to {policies_dest_path}")
    return policies_dest_path
def remove_temp_policies(thread_id: int, path_prefix: str = ''):
    """
    删除 temp_files/thread_{i}/policies 目录，
    
    参数:
        thread_id (int): 当前线程编号，例如 1 表示 thread_1
    """
    # 定义目标路径
    temp_thread_path = os.path.join(path_prefix ,"temp_files", f"thread_{thread_id}")

    if os.path.exists(temp_thread_path):
        print(f"Removing existing temp_files/thread_{thread_id} directory")
        shutil.rmtree(temp_thread_path)
    else:
        print(f"temp_files/thread_{thread_id} directory does not exist")
import os
import json
import importlib.util

def generate_agent_params(policies_path: str):
    agent_info_list = []

    for policy_name in os.listdir(policies_path):
        policy_path = os.path.join(policies_path, policy_name)
        config_path = os.path.join(policy_path, "agent.config")
        policy_file = os.path.join(policy_path, "policy.py")
        model_file = os.path.join(policy_path, "model.py")  # model.py 文件路径
        agent_dict = {}
        if os.path.isdir(policy_path) and os.path.exists(config_path) and os.path.exists(policy_file):
            # 读取 agent.config
            with open(config_path, 'r') as f:
                config = json.load(f)

            # 提取配置中的信息
            policy_id = config.get("policy_id", policy_name)
            idx = config.get("idx")
            description = config.get("description", "")
            trainable = config.get("trainable", True)
            lr = config.get("optimizer_config", {}).get("lr", 1e-3)
            input_dim = config.get("observation_shape", 10)
            action_dim = config.get("action_shape", 5)
            agent_id = config.get("agent_id", None)

            # 动态加载 model.py，确保 A2CNetwork 类能够被加载
            spec_model = importlib.util.spec_from_file_location("Model", model_file)
            model_module = importlib.util.module_from_spec(spec_model)
            spec_model.loader.exec_module(model_module)
            
            # 确保 A2CNetwork 被正确导入
            Model = model_module.Model

            # 动态加载 policy.py
            spec_policy = importlib.util.spec_from_file_location("Policy", policy_file)
            policy_module = importlib.util.module_from_spec(spec_policy)
            spec_policy.loader.exec_module(policy_module)
            env_wrapper = EnvWrapper()

            # 创建 Policy 实例
            policy_instance = policy_module.Policy(input_dim, action_dim, lr, Model, env=env_wrapper.env, agent_id=agent_id)
            policy_model_pth = os.path.join(policy_path, "model.pt")
            optimizer_pth = os.path.join(policy_path, "optimizer.pt")
            # 加载模型参数
            if os.path.exists(policy_model_pth):
                policy_instance.load(policy_model_pth, optimizer_pth)
                print(f"Loaded model from {policy_model_pth}. Loading optimizer from {optimizer_pth}.")
            else:
                print(f"Model file {policy_model_pth} does not exist.")
            # 将 A2CNetwork 作为 policy_instance 的属性或传入创建
            # policy_instance.model = A2CNetwork(input_dim, action_dim)

            agent_info = {
                "agent_id": agent_id,
                "policy_id": policy_id,
                "idx": idx,
                "description": description,
                "path": policy_path,
                "trainable": trainable,
                "policy_instance": policy_instance,

            }
            

            agent_info_list.append(agent_info)

    output_path = os.path.join(os.path.dirname(policies_path), "agent_params.json")
    # with open(output_path, 'w') as f:
    #     json.dump({"policies": agent_info_list}, f, indent=2)
    with open(output_path, 'w') as f:
    # 创建一个新的列表，其中排除了 'trainable' 属性
        agent_info_list_filtered = [
            {key: value for key, value in agent_info.items() if key != "policy_instance"}
            for agent_info in agent_info_list
        ]
        json.dump({"policies": agent_info_list_filtered}, f, indent=2)

    print(f"Saved agent_params.json to {output_path}")
    return agent_info_list

def sample_trajectory(agent_info_list, batch_size=4096):
    env_wrapper = EnvWrapper()
    max_steps = batch_size
    trajectories = {agent_info["agent_id"]: [] for agent_info in agent_info_list}
    while len(trajectories[agent_info_list[0]["agent_id"]]) < batch_size:
        # 生成一个新的轨迹
        observations = env_wrapper.reset()
        step_count = 0
        while step_count < max_steps:
            
            actions = {}
            for agent_info in agent_info_list:
                policy_instance = agent_info["policy_instance"]
                action = policy_instance.predict(observations[agent_info["agent_id"]])
                actions[agent_info["agent_id"]] = action
            next_observations, rewards, dones, truncations, info = env_wrapper.step(actions)
            
            if next_observations == {}:
                observations = env_wrapper.reset()
                continue
            done_all = all(dones.get(agent_info['agent_id'], False) or truncations.get(agent_info['agent_id'], False) for agent_info in agent_info_list)
            if done_all:
                observations = env_wrapper.reset()
                continue
            for agent_info in agent_info_list:
                trajectories[agent_info["agent_id"]].append({
                    "obs": observations[agent_info["agent_id"]],
                    "act": actions[agent_info["agent_id"]],
                    "rew": rewards[agent_info["agent_id"]],
                    "obs_next": next_observations[agent_info["agent_id"]],
                    "terminated": dones[agent_info["agent_id"]],
                    "truncated": truncations[agent_info["agent_id"]],
                    "info": info[agent_info["agent_id"]],
                })
            observations = next_observations
            step_count += 1

    return trajectories
import random

def train(epochs = 10, agent_info_list = None, batch_size = 4096):


    all_gradients = {}
    
    avg_reward = 0
    for i in range(epochs):
        traj = sample_trajectory(agent_info_list, batch_size = batch_size)
        # print (i)
        print(f"\nEpoch {i + 1}")
        for agent_info in agent_info_list:
            if not agent_info["trainable"]:
                continue
            # if random.random() < 0.5:
            #     continue
            agent_data = traj[agent_info["agent_id"]]
            batch = Batch({
                "obs": [step["obs"] for step in agent_data],
                "act": [step["act"] for step in agent_data],
                "rew": [step["rew"] for step in agent_data],
                "obs_next": [step["obs_next"] for step in agent_data],
                "terminated": [step["terminated"] for step in agent_data],
                "truncated": [step["truncated"] for step in agent_data],
                "info": [step["info"] for step in agent_data],
            })
            policy_instance = agent_info["policy_instance"]
            # 训练策略
            # print(batch_data)
            gradients_this_epoch = policy_instance.train(batch)
            if agent_info["agent_id"] not in all_gradients:
                all_gradients[agent_info["agent_id"]] = {
                    key: value.detach().cpu().clone() for key, value in gradients_this_epoch.items()
                }
            else:
                for key in all_gradients[agent_info["agent_id"]]:
                    all_gradients[agent_info["agent_id"]][key] += gradients_this_epoch[key].detach().cpu()# all_gradients[agent_info["agent_id"]] = gradients
            agent_id = agent_info["agent_id"]
            rewards = [step["rew"] for step in agent_data]
            avg_reward = sum(rewards) / len(rewards)
            # print(rewards)
            print(f"Agent {agent_id} trained with average reward: {avg_reward:.4f}")
    return all_gradients
    # print(traj)

