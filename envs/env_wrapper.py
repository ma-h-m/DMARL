import gym
from pettingzoo.mpe import simple_adversary_v3  # 引入 simple_adversary 环境
from collections import defaultdict

## Adversarial environment
## Usage:
    # from envs.simple_adversary_wrapper import SimpleAdversaryWrapper

    # # 初始化环境
    # env = SimpleAdversaryWrapper()

    # # 重置环境
    # observations = env.reset()

    # # 训练过程中，执行多步
    # actions = {agent_id: action for agent_id, action in zip(env.get_agent_ids(), some_action_list)}
    # observations, rewards, dones, info = env.step(actions)

    # # 结束时关闭环境
    # env.close()

class SimpleAdversaryWrapper:
    def __init__(self, render_mode='ansi'):
        """
        Initializes the SimpleAdversaryWrapper with the environment.
        """
        self.env = simple_adversary_v3.parallel_env(render_mode = render_mode)  # 初始化 simple_adversary 环境
        # self.env.reset()  # Reset the environment to its initial state

    def reset(self):
        """
        Resets the environment to the initial state.
        
        Returns:
            dict: A dictionary with agent ids as keys and initial observations as values.
        """
        obs, _ = self.env.reset()  # Get initial observations from the environment
        return {agent: obs[agent] for agent in self.env.agents}  # 返回各个智能体的初始观测

    def step(self, actions):
        """
        Takes a step in the environment with the given actions for each agent.
        
        Args:
            actions (dict): A dictionary of actions, where keys are agent ids, and values are actions.
        
        Returns:
            tuple: (observations, rewards, done, info) for all agents.
        """
        # 将动作应用到环境

        obs, rewards, dones, truncations , infos = self.env.step(actions)  # 环境返回新的观测，奖励，是否完成等
        # observations, rewards, terminations, truncations, infos
        # 返回所有智能体的观察、奖励、完成状态和其他信息
        return (
            {agent: obs[agent] for agent in self.env.agents},  # 观测
            {agent: rewards[agent] for agent in self.env.agents},  # 奖励
            {agent: dones[agent] for agent in self.env.agents},  # 完成状态
            {agent: truncations[agent] for agent in self.env.agents},   #  截断状态
            {agent: infos[agent] for agent in self.env.agents}   # 其他信息
        )

    def get_agent_ids(self):
        """
        Returns the list of agent ids present in the environment.
        
        Returns:
            list: A list of agent ids.
        """
        return self.env.agents  # 获取所有智能体的ID

    def render(self, mode="human"):
        """
        Renders the environment for visualization.
        
        Args:
            mode (str): The rendering mode, either "human" or "rgb_array".
        """
        return self.env.render(mode)  # 使用环境自带的渲染功能

    def close(self):
        """
        Cleans up any resources used by the environment.
        """
        self.env.close()  # 关闭环境

# tmp_env = SimpleAdversaryWrapper()
# print(tmp_env.env.agents)
# print(tmp_env.env.action_space)