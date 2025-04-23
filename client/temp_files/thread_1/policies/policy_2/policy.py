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

    def train(self, batch_data, batch_size=32, repeat=1):
        """
        Train the model using the provided batch data and return gradients for uploading.
        This version splits the data into mini-batches, shuffles the data each time, and accumulates gradients.

        Parameters:
        - batch_data (dict): A dictionary containing the following:
            - 'obs' (torch.Tensor): Batch of state observations.
            - 'action' (torch.Tensor): Batch of actions taken.
            - 'reward' (torch.Tensor): Batch of rewards received.
            - 'next_obs' (torch.Tensor): Batch of next state observations.
            - 'done' (torch.Tensor): Batch of done flags (indicating if episode ended).
        
        Returns:
        - gradients (dict): Dictionary containing the accumulated gradients for each parameter.
        """
        # Convert batch_data to a tensor dataset for easy shuffling and batching
            # Convert batch_data to torch tensors (if they're numpy arrays)
        obs_tensor = torch.tensor(batch_data['obs'], dtype=torch.float32)
        act_tensor = torch.tensor(batch_data['act'], dtype=torch.long)
        rew_tensor = torch.tensor(batch_data['rew'], dtype=torch.float32)
        obs_next_tensor = torch.tensor(batch_data['obs_next'], dtype=torch.float32)
        terminated_tensor = torch.tensor(batch_data['terminated'], dtype=torch.bool)
        truncated_tensor = torch.tensor(batch_data['truncated'], dtype=torch.bool)

        dataset = TensorDataset(obs_tensor, act_tensor, rew_tensor, obs_next_tensor, terminated_tensor, truncated_tensor)

        # Create a DataLoader to handle batching and shuffling
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        client_grads = {}

        # Perform the training update using mini-batches
        with policy_within_training_step(self.policy), torch_train_mode(self.policy):
            for _ in range(repeat):
                # Iterate over mini-batches (shuffled each time)
                for mini_batch in data_loader:
                    # Unpack the mini_batch
                    obs, act, rew, obs_next, terminated, truncated = mini_batch

                    # Create a ReplayBuffer for this mini-batch
                    buffer = ReplayBuffer(size=batch_size)
                    for j in range(len(obs)):
                        buffer.add(
                            Batch(
                                obs=obs[j].numpy(),  # Convert PyTorch Tensor to NumPy array
                                act=act[j].numpy(),
                                rew=rew[j].numpy(),
                                obs_next=obs_next[j].numpy(),
                                terminated=terminated[j].numpy(),
                                truncated=truncated[j].numpy(),
                                info={},
                            )
                        )

                    # Update the model using the mini-batch
                    self.policy.update(sample_size=0, buffer=buffer, batch_size=batch_size, repeat=1)

                    # Collect gradients for actor
                    for name, param in self.policy.actor.named_parameters():
                        if param.grad is not None:
                            full_name = f"actor.{name}"
                            if full_name not in client_grads:
                                client_grads[full_name] = param.grad.clone()
                            else:
                                client_grads[full_name] += param.grad

                    # Collect gradients for critic
                    for name, param in self.policy.critic.named_parameters():
                        if param.grad is not None:
                            full_name = f"critic.{name}"
                            if full_name not in client_grads:
                                client_grads[full_name] = param.grad.clone()
                            else:
                                client_grads[full_name] += param.grad

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