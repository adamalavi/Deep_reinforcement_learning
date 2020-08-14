import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_state=[64, 32], hidden_adv=[128, 32]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1_adv = nn.Linear(state_size, hidden_adv[0])
        self.fc2_adv = nn.Linear(hidden_adv[0], hidden_adv[1])
        self.out_adv = nn.Linear(hidden_adv[1], action_size)
        
        self.fc1_state = nn.Linear(state_size, hidden_state[0])
        self.fc2_state = nn.Linear(hidden_state[0], hidden_state[1])
        self.out_state = nn.Linear(hidden_state[1], 1)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x_adv = F.relu(self.fc1_adv(state))
        x_adv = F.relu(self.fc2_adv(x_adv))
        x_adv = self.out_adv(x_adv)
        
        x_state = F.relu(self.fc1_state(state))
        x_state = F.relu(self.fc2_state(x_state))
        x_state = self.out_state(x_state)
        
        return x_adv.sub_(x_adv.mean()).add_(x_state)
