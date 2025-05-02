import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Actor network responsible for producing the action probabilities (for discrete) or parameters (for continuous).
    """
    def __init__(self, input_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, 128)
        self.policy_head = nn.Linear(128, action_dim)  # Discrete action space: raw action values
        self.action_type = "discrete"  # Change to "continuous" if using continuous actions

    def forward(self, x, state = None, info = None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
     
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        policy = self.policy_head(x)  # For discrete: logits, for continuous: action parameters
        return policy, state


class Critic(nn.Module):
    """
    Critic network responsible for predicting the state value.
    """
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, 128)
        self.value_head = nn.Linear(128, 1)  # Output state value

    def forward(self, x, state = None, info = None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
            
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        value = self.value_head(x)
        return value


class Model(nn.Module):
    """
    Model that combines Actor and Critic networks.
    """
    def __init__(self, input_dim, action_dim):
        super(Model, self).__init__()
        self.actor = Actor(input_dim, action_dim)
        self.critic = Critic(input_dim)

    def forward(self, x):
        """
        Forward pass through both actor and critic.
        """
        policy = self.actor(x)  # Get action probabilities or parameters
        value = self.critic(x)  # Get state value
        return policy, value