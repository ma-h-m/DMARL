import os
import importlib.util
import torch

class PolicyWrapper:
    """
    这是策略包装器类，封装了策略的加载、推理、训练和保存等功能。

    API:

    1. __init__(self, policy_path):
       - 初始化策略包装器，加载并实例化 `policy.py` 中的 `Policy` 类。
       - 参数: `policy_path` (str) - 策略所在目录的路径。
       - 示例:
         ```python
         policy = PolicyWrapper("new_policy/policy_1")
         ```

    2. load_policy(self):
       - 加载指定目录下的 `policy.py` 文件并实例化 `Policy` 类。
       - 返回: 实例化的 `Policy` 类对象。

    3. infer(self, obs):
       - 使用加载的策略进行推理，根据输入的状态（`obs`）生成动作。
       - 参数: `obs` (list 或 numpy array) - 输入的状态值。
       - 返回: 动作（tensor）
       - 示例:
         ```python
         obs = [[0.1, 0.2, 0.3, 0.4]]
         action = policy.infer(obs)
         ```

    4. train(self, batch_data):
       - 用于训练策略，接受一批训练数据（包括状态、动作、奖励、下一状态和终止标志）。
       - 参数: `batch_data` (dict) - 包含以下字段的字典:
         - `"obs"` (tensor): 状态（当前时刻的观测）
         - `"action"` (tensor): 动作
         - `"reward"` (tensor): 奖励
         - `"next_obs"` (tensor): 下一状态
         - `"done"` (tensor): 终止标志
       - 示例:
         ```python
         batch_data = {
             "obs": [[0.1, 0.2, 0.3, 0.4]],
             "action": [[1]],
             "reward": [[1.0]],
             "next_obs": [[0.2, 0.3, 0.4, 0.5]],
             "done": [[False]]
         }
         policy.train(batch_data)
         ```

    5. save(self):
       - 保存策略模型和优化器的状态字典。
       - 保存的文件包括:
         - `model.pth`: 模型权重
         - `optimizer.pth`: 优化器状态（如果存在）
       - 示例:
         ```python
         policy.save()
         ```

    6. load(self):
       - 加载已保存的策略模型和优化器的状态字典。
       - 示例:
         ```python
         policy.load()
         ```
    """
    def __init__(self, policy_path):
        """
        初始化策略包装器，加载 policy.py 并实例化 Policy 类
        :param policy_path: 策略目录（如 "new_policy/policy_1"）
        """
        self.policy_path = policy_path
        self.policy = self.load_policy()
    
    def load_policy(self):
        """
        加载指定目录下的 policy.py，并实例化 Policy 类
        """
        policy_file = os.path.join(self.policy_path, "policy.py")
        if not os.path.exists(policy_file):
            raise FileNotFoundError(f"Policy file not found: {policy_file}")

        spec = importlib.util.spec_from_file_location("policy", policy_file)
        policy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(policy_module)
        return policy_module.Policy()  # 确保 policy.py 里有一个 `Policy` 类

    def infer(self, obs):
        """
        使用策略进行推理
        :param obs: 输入观测值 (list 或 numpy 数组)
        :return: 策略动作 (tensor)
        
        示例:
        ```
        policy = PolicyWrapper("new_policy/policy_1")
        obs = [[0.1, 0.2, 0.3, 0.4]]  # 假设状态空间是4维
        action = policy.infer(obs)
        print(action)
        ```
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        return self.policy.forward(obs_tensor)

    def train(self, batch_data):
        """
        训练策略
        :param batch_data: 训练数据，包含 "obs", "action", "reward", "next_obs", "done"
        :type batch_data: dict
        
        示例:
        ```
        batch_data = {
            "obs": [[0.1, 0.2, 0.3, 0.4]],
            "action": [[1]],
            "reward": [[1.0]],
            "next_obs": [[0.2, 0.3, 0.4, 0.5]],
            "done": [[False]]
        }
        policy.train(batch_data)
        ```
        """
        obs = torch.tensor(batch_data["obs"], dtype=torch.float32)
        action = torch.tensor(batch_data["action"], dtype=torch.long)
        reward = torch.tensor(batch_data["reward"], dtype=torch.float32)
        next_obs = torch.tensor(batch_data["next_obs"], dtype=torch.float32)
        done = torch.tensor(batch_data["done"], dtype=torch.bool)

        self.policy.update(obs, action, reward, next_obs, done)

    def save(self):
        """
        保存策略模型和优化器
        
        示例:
        ```
        policy.save()
        ```
        """
        torch.save(self.policy.model.state_dict(), os.path.join(self.policy_path, "model.pth"))
        if self.policy.optim:
            torch.save(self.policy.optim.state_dict(), os.path.join(self.policy_path, "optimizer.pth"))

    def load(self):
        """
        加载策略模型和优化器
        
        示例:
        ```
        policy.load()
        ```
        """
        model_path = os.path.join(self.policy_path, "model.pth")
        optimizer_path = os.path.join(self.policy_path, "optimizer.pth")
        
        if os.path.exists(model_path):
            self.policy.model.load_state_dict(torch.load(model_path))
        
        if self.policy.optim and os.path.exists(optimizer_path):
            self.policy.optim.load_state_dict(torch.load(optimizer_path))