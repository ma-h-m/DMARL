import torch
import torch.optim as optim
import numpy as np
from tianshou.policy import A2CPolicy
from tianshou.data import Batch
from .model import A2CNetwork

class Policy:
    def __init__(self, input_dim, action_dim, lr=1e-3):
        """
        Initialize the policy using Tianshou's A2CPolicy with a custom model.

        Parameters:
        - input_dim (int): The size of the input (state space).
        - action_dim (int): The number of possible actions.
        - lr (float): The learning rate for the optimizer.
        """
        self.model = A2CNetwork(input_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Initialize the Tianshou A2C Policy
        self.policy = A2CPolicy(self.model, self.optimizer)
        self.action_dim = action_dim


    def train(self, batch_data):
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
        # Convert the batch data dictionary into Tianshou's Batch
        batch = Batch(obs=torch.tensor(batch_data['obs'], dtype=torch.float32),
                      act=torch.tensor(batch_data['action'], dtype=torch.long),
                      rew=torch.tensor(batch_data['reward'], dtype=torch.float32),
                      obs_next=torch.tensor(batch_data['next_obs'], dtype=torch.float32),
                      done=torch.tensor(batch_data['done'], dtype=torch.float32))

        # Perform the update with Tianshou's policy
        self.policy.update(batch)

        # After update, get the gradients of the model parameters
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.clone())  # Save the gradient for uploading

        return gradients

    def predict(self, state):
        """
        Predict the action for a given state using the policy.

        Parameters:
        - state (torch.Tensor): The input state for which the action is to be predicted.

        Returns:
        - action (torch.Tensor): The chosen action based on the policy.
        """
        state = torch.tensor(state, dtype=torch.float32)
        return self.policy.select(state)

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