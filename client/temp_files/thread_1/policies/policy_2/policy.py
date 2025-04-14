import torch
import torch.optim as optim
from tianshou.policy import A2CPolicy
from tianshou.data import Batch
from tianshou.data import ReplayBuffer
# from .model import Model
from torch.distributions import Categorical
def dist_fn(logits: torch.Tensor) -> Categorical:
    return Categorical(logits=logits)
# from tianshou.data import policy_within_training_step, torch_train_mode
from tianshou.utils.torch_utils import policy_within_training_step, torch_train_mode
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

    def train(self, batch_data, batch_size = 32, repeat = 1):
        """
        Train the model using the provided batch data and return gradients for uploading.

        Parameters:
        - batch_data (dict): A dictionary containing the following:
            - 'obs' (torch.Tensor): Batch of state observations.
            - 'action' (torch.Tensor): Batch of actions taken.
            - 'reward' (torch.Tensor): Batch of rewards received.
            - 'next_obs' (torch.Tensor): Batch of next state observations.
            - 'done' (torch.Tensor): Batch of done flags (indicating if episode ended).

        Returns:
        - gradients (list): List of gradients for each parameter.
        """
        buffer = ReplayBuffer(size = len(batch_data['obs']))
        for i in range(len(batch_data['obs'])):
            buffer.add(
                Batch (
                    obs = batch_data['obs'][i],
                    act = batch_data['act'][i],
                    rew = batch_data['rew'][i],
                    obs_next = batch_data['obs_next'][i],
                    terminated = batch_data['terminated'][i],
                    truncated = batch_data['truncated'][i],
                    info = {},
                )
            )
        # print(buffer)
        # self.policy.learn(batch_data, batch_size = batch_size, repeat = repeat)
        with policy_within_training_step(self.policy), torch_train_mode(self.policy):
            self.policy.update(sample_size=0, buffer=buffer, batch_size=10, repeat=6).pprint_asdict()

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