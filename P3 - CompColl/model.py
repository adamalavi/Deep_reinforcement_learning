import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_size=256, fc2_size=128):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size*2, fc1_size)
        self.bn1 = nn.LayerNorm(fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.bn2 = nn.LayerNorm(fc2_size)
        self.mu = nn.Linear(fc2_size, action_size)
        
        self.init_weights()
        
    def init_weights(self):
        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc1.bias.data.uniform_(-f1, f1)
        self.fc2.bias.data.uniform_(-f2, f2)
        self.mu.weight.data.uniform_(-3e-3, 3e-3) # Refer the research paper to know why a fixed value is used here
        self.mu.bias.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        
        if len(state) == self.state_size:
            state = state.unsqueeze(0)
        
        x = self.bn1(self.fc1(state))
        x = F.relu(x)
        x = self.bn2(self.fc2(x))
        return F.tanh(self.mu(x))
        
class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_size=256, fc2_size=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size*2, fc1_size)
        self.bn1 = nn.LayerNorm(fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.bn2 = nn.LayerNorm(fc2_size)
        
        self.action_layer = nn.Linear(action_size*2, fc2_size)
        self.q = nn.Linear(fc2_size, 1)
        
        self.init_weights()
        
    def init_weights(self):
        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc1.bias.data.uniform_(-f1, f1)
        self.fc2.bias.data.uniform_(-f2, f2)
        self.q.weight.data.uniform_(-3e-3, 3e-3) # Refer the research paper to know why a fixed value is used here
        self.q.bias.data.uniform_(-3e-3, 3e-3)    

    def forward(self, state, action):
        state_value = self.bn1(self.fc1(state))
        state_value = F.relu(state_value)
        state_value = self.bn2(self.fc2(state_value))
        
        action_value = F.relu(self.action_layer(action))
        state_action_value = F.relu(torch.add(state_value, action_value))
        return self.q(state_action_value)
        
        
        
        
        
        
        
        
        
        
        
        
        