import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class A2CNetwork(nn.Module):
    """
    A2C Network implementation using PyTorch.
    The network contains a shared feature extractor followed by a value head and a policy head.
    """

    def __init__(self, input_dim, action_dim):
        """
        Initialize the A2C network.

        Parameters:
        - input_dim (int): The size of the input (state space).
        - action_dim (int): The number of possible actions.
        """
        super(A2CNetwork, self).__init__()

        # Define the shared layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)

        # Value function head
        self.value_head = nn.Linear(128, 1)

        # Policy function head (actor)
        self.policy_head = nn.Linear(128, action_dim)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        - x (torch.Tensor): The input state.

        Returns:
        - value (torch.Tensor): The predicted value of the state.
        - policy (torch.Tensor): The predicted action probabilities (logits).
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = self.value_head(x)
        policy = self.policy_head(x)

        return value, policy

    def get_action(self, state):
        """
        Get action based on the current policy.

        Parameters:
        - state (torch.Tensor): The input state.

        Returns:
        - action (torch.Tensor): The chosen action based on the policy.
        """
        _, policy = self(state)
        probs = F.softmax(policy, dim=-1)
        action = torch.multinomial(probs, 1)  # Sample from the action distribution
        return action