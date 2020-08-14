import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 5e-2              # for soft update of target parameters
LR_CRITIC = 1e-3        # learning rate of critic
LR_ACTOR = 1e-3         # learning rate of actor
UPDATE_EVERY = 1        # Updating learning params
LEARN_NUM = 5           # No. of learning passes
EPS_START = 5.5         # Decay factor for noise
EPS_FINAL = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, num_agents=20):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        print("Running on: "+str(device))
        
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(seed)
        self.eps = EPS_START
        self.eps_decay = 0.0005
        # Actor network
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        
        # Critic network
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        self.noise = OUNoise((num_agents, action_size), seed)
    
    def step(self, state, action, reward, next_state, done, agent_id):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step += 1
        # Learn every UPDATE_EVERY time steps.
        if (self.t_step%UPDATE_EVERY)== 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                for _ in range(LEARN_NUM):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA, agent_id)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((self.num_agents, self.action_size))
        
        self.actor_local.eval()
        with torch.no_grad():
            for i, state in enumerate(states):
                actions[i,:] = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            actions += self.eps * self.noise.sample()
        return np.clip(actions, -1, 1)

    def learn(self, experiences, gamma, agent_id):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        # ------------------- update critic network ------------------- #
        target_actions = self.actor_target.forward(next_states)
        # Construct next actions vector relative to the agent
        if agent_id == 0:
            target_actions = torch.cat((target_actions, actions[:,2:]), dim=1)
        else:
            target_actions = torch.cat((actions[:,:2], target_actions), dim=1)
            
        next_critic_value = self.critic_target.forward(next_states, target_actions)
        critic_value = self.critic_local.forward(states, actions)
        # Q targets for current state
        # If the episode is over, the reward from the future state will not be incorporated
        Q_targets = rewards + (gamma * next_critic_value * (1 - dones))

        critic_loss = F.mse_loss(critic_value, Q_targets)
        # Minimizing loss
        self.critic_local.train()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optim.step()
        
        self.critic_local.eval()
        
        # ------------------- update actor network ------------------- #
        self.actor_local.train()
        self.actor_optim.zero_grad()
        mu = self.actor_local.forward(states)
        # Construct mu vector relative to each agent
        if agent_id == 0:
            mu = torch.cat((mu, actions[:,2:]), dim=1)
        else:
            mu = torch.cat((actions[:,:2], mu), dim=1)
            
        actor_loss = -self.critic_local(states, mu).mean()
        actor_loss.backward()
        self.actor_optim.step()
        
        self.actor_local.eval()
        
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

        # update noise decay parameter
        self.eps -= self.eps_decay
        self.eps = max(self.eps, EPS_FINAL)
        self.noise.reset()
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def reset(self):
        self.noise.reset()

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2): # default: mu=0., theta=0.15, sigma=0.2
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(0, 1, self.size)
        self.state = x + dx
        return self.state
    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)