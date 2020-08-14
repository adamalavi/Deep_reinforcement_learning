import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.policy = defaultdict(lambda: np.zeros(self.nA))
        
        # Expected SARSA parameters - best average reward = 9.317
#         self.epsilon = 1.0
#         self.eps_decay = 0.9999
#         self.eps_min = 0.0001
        
#         self.alpha = 0.2
#         self.gamma = 0.99

        # SARSA max - best average reward = 9.419
        self.epsilon = 1.0
        self.eps_decay = 0.995
        self.eps_min = 0.0001
        
        self.alpha = 0.2
        self.gamma = 0.9999

#         self.epsilon = 1.0
#         self.eps_decay = 0.99
#         self.eps_min = 0.0001
        
#         self.alpha = 0.2
#         self.alph_decay = 0.999999
#         self.alph_min = 0.05
#         self.gamma = 0.9999

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.epsilon = max(self.epsilon*self.eps_decay, self.eps_min)
        self.policy = np.ones(self.nA) * (self.epsilon/self.nA)
        self.policy[np.argmax(self.Q[state])] = 1 - self.epsilon + (self.epsilon/self.nA)
        return np.random.choice(np.arange(self.nA), p = self.policy)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
#         self.alpha = max(self.alpha*self.alph_decay, self.alph_min)
        if not done:
            self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + (self.gamma * np.max(self.Q[next_state])) - self.Q[state][action])
        if done:
            self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * 0 - self.Q[state][action])