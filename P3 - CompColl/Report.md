# Project Report - Competition and collaboration

### Approach
The approach used for completing the projects is Deep deterministic policy gradient aimed at solving a multi-agent problem. It is a type of actor-critic method where two neural network architectures are used, one works as the actor and the other as the critic. The job of the actor is to predict the optimal policy mu which will be used by the agent to perform the task. The job of the critic is to predict a value which will be used to tune the weights of the actor network. The network used here incorporates various techniques like experience replay and fixed Q-target just like vanilla DQN. This algorithm worked pretty well and I was able to achieve the target average score of +0.5 over 100 consecutive steps in around 1300 episodes.

### Network architecture
The actor network architecture was fairly simple with two fully connected layers of size 256 and 128 followed by another linear output layer of output size 2 in this case for every agent as there are two actions i.e move and jump. A final tanh activation layer was applied to get a value between -1 and 1. Batch normalisation was used after both the layers to obtain quicker convergence. All the layer weights and biases were initialised as mentioned in the research paper on [DDPG](https://arxiv.org/pdf/1509.02971.pdf)
The critic network architecture was a little more complicated. First a state value output was obtained by activating the state input by passing it through two fully connected convolutional layers, again of size 256 and 128. ReLu activation and batch normalisation was used between the layers. After that another fully connected layer is used to obtain the action value output from the action input which has an output size same as that of the second fully connected layer. The two values are then added after ReLu activation and then a final linear layer is obtained to get a single output value.

### Hyperparameters 

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

### Future modifications
Trying other algorithms like A2C, A3C, TRPO, etc might give better results and may be tested.