import torch
import torch.optim as optim
from tianshou.policy import A2CPolicy
from tianshou.data import Batch
from tianshou.data import ReplayBuffer
# from .model import Model
from torch.distributions import Categorical
import torch.nn.functional as F
from torch import nn
import uuid
def dist_fn(logits: torch.Tensor) -> Categorical:
    return Categorical(logits=logits)
# from tianshou.data import policy_within_training_step, torch_train_mode
from tianshou.utils.torch_utils import policy_within_training_step, torch_train_mode
from torch.utils.data import DataLoader, TensorDataset
class Policy:
    def __init__(self, input_dim, action_dim, lr=1e-3, Model=None, env = None, agent_id = None):
        """
        Initialize the policy using Tianshou's A2CPolicy with a custom model.

        Parameters:
        - input_dim (int): The size of the input (state space).
        - action_dim (int): The number of possible actions.
        - lr (float): The learning rate for the optimizer.
        """
        # Initialize the combined model (Actor + Critic)
        self.model = Model(input_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Initialize the Tianshou A2C Policy with our custom model
        self.policy = A2CPolicy(
            actor=self.model.actor,
            critic=self.model.critic,
            optim=self.optimizer,
            dist_fn=dist_fn,
            action_space=env.action_space(agent_id),  # Use the action space from the environment
            action_scaling=False
        )
        self.action_dim = action_dim
        self.max_grad_norm = 1

    def train(self, batch_data, batch_size=32, repeat=1):
        """
        使用给定的batch数据训练模型，并返回上传的梯度。

        参数:
        - batch_data (dict): 包含以下内容的字典：
            - 'obs' (torch.Tensor): 状态观测的batch。
            - 'act' (torch.Tensor): 行为的batch。
            - 'rew' (torch.Tensor): 奖励的batch。
            - 'next_obs' (torch.Tensor): 下一状态观测的batch。
            - 'done' (torch.Tensor): 完成标志的batch（指示回合是否结束）。

        返回:
        - gradients (dict): 每个参数的梯度字典。
        """
        buffer = ReplayBuffer(size=len(batch_data['obs']))
        for i in range(len(batch_data['obs'])):
            buffer.add(
                Batch(
                    obs=batch_data['obs'][i],
                    act=batch_data['act'][i],
                    rew=batch_data['rew'][i],
                    obs_next=batch_data['obs_next'][i],
                    terminated=batch_data['terminated'][i],
                    truncated=batch_data['truncated'][i],
                    info={},
                )
            )
        
        client_grads = {}

        # 在训练过程中我们启用梯度计算并手动计算损失
        with torch.set_grad_enabled(True), policy_within_training_step(self.policy), torch_train_mode(self.policy):
            for i in range(repeat):
                # 分批次处理数据
                batch, indices = buffer.sample(0)
                self.policy.updating = True
                batch = self.policy.process_fn(batch, buffer, indices)

                for minibatch in batch.split(batch_size, merge_last= True):
                    
                    # 处理当前的minibatch数据
                    dist = self.policy(minibatch).dist  # 计算分布
                    log_prob = dist.log_prob(minibatch.act)  # 计算log概率
                    log_prob = log_prob.reshape(len(minibatch.adv), -1).transpose(0, 1)  # 调整形状
                    actor_loss = -(log_prob * minibatch.adv).mean()  # 计算actor的损失

                    # 计算critic网络的损失
                    value = self.policy.critic(minibatch.obs).flatten()  # 计算值函数
                    vf_loss = F.mse_loss(minibatch.returns, value)  # 计算critic的损失

                    # 计算熵损失
                    ent_loss = dist.entropy().mean()

                    # 总损失
                    total_loss = actor_loss + self.policy.vf_coef * vf_loss - self.policy.ent_coef * ent_loss

                    self.optimizer.zero_grad()
                    # 对总损失进行反向传播，计算梯度
                    total_loss.backward()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(
                            self.policy._actor_critic.parameters(),
                            max_norm = self.max_grad_norm
                        )
                        

                    # 计算actor网络的梯度
                    for name, param in self.policy.actor.named_parameters():
                        if param.grad is not None:
                            full_name = f"actor.{name}"
                            if full_name not in client_grads:
                                client_grads[full_name] = param.grad.clone()
                            else:
                                client_grads[full_name] += param.grad

                    # 计算critic网络的梯度
                    for name, param in self.policy.critic.named_parameters():
                        if param.grad is not None:
                            full_name = f"critic.{name}"
                            if full_name not in client_grads:
                                client_grads[full_name] = param.grad.clone()
                            else:
                                client_grads[full_name] += param.grad

                    # 禁止更新参数，防止执行优化步骤
                    self.policy.updating = False

        return client_grads
        # print(client_grads)
        # print(buffer)
        # self.policy.learn(batch_data, batch_size = batch_size, repeat = repeat)
        # with policy_within_training_step(self.policy), torch_train_mode(self.policy):
        #     self.policy.update(sample_size=0, buffer=buffer, batch_size=batch_size, repeat=repeat).pprint_asdict()
        # for name, param in self.policy.actor.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: grad norm = {param.grad.norm().item()}")
        # # After update, get the gradients of the model parameters
        # gradients = []
        # for param in self.model.parameters():
        #     if param.grad is not None:
        #         gradients.append(param.grad.clone())  # Save the gradient for uploading

        # return gradients

    def predict(self, state):
        """
        Predict the action for a given state using the policy.

        Parameters:
        - state (torch.Tensor): The input state for which the action is to be predicted.

        Returns:
        - action (torch.Tensor): The chosen action based on the policy.
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # action = policy(Batch(obs=np.array([obs]))).act[0]
        # return self.policy(Batch(obs=state)).act[0]
        return self.policy.compute_action(obs = state)
        # return self.policy.select(state)

    def save(self, model_path, optimizer_path):
        """
        Save the model and optimizer state.

        Parameters:
        - model_path (str): Path to save the model weights.
        - optimizer_path (str): Path to save the optimizer state.
        """
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)

    def load(self, model_path, optimizer_path):
        """
        Load the model and optimizer state.

        Parameters:
        - model_path (str): Path to load the model weights.
        - optimizer_path (str): Path to load the optimizer state.
        """
        self.model.load_state_dict(torch.load(model_path))
        self.optimizer.load_state_dict(torch.load(optimizer_path))