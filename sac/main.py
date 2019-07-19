import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from datetime import datetime
import time
from tensorboardX import SummaryWriter


env_name = 'Pendulum-v0'
n_episodes = 1000
hidden_size = 256
replay_buffer_size = 1e6
train_batch_size = 256
gamma = 0.99
target_smoothing_coeff = 0.005
lr = 3e-4

evaluate_freq = 50
seed = 0
load_path = None
save_freq = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExpReplay:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.pointer = 0
        self.max_len = replay_buffer_size

    def add(self, exp_tuple):
        if len(self.states) < self.max_len:
            self.states.append(0)
            self.actions.append(0)
            self.rewards.append(0)
            self.dones.append(0)

        state_seq, action_seq, reward_seq, done_seq = exp_tuple
        self.states[self.pointer] = state_seq
        self.actions[self.pointer] = action_seq
        self.rewards[self.pointer] = reward_seq
        self.dones[self.pointer] = done_seq
        self.pointer = int((self.pointer + 1) % self.max_len)

    def sample(self, batch_size):
        st_b, ac_b, rew_b, dn_b = \
            zip(*random.sample(list(zip(self.states,self.actions,self.rewards, self.dones)),
                               batch_size))
        return st_b, ac_b, rew_b, dn_b

    def __len__(self):
        return len(self.states)

class Actor(nn.Module):
    def __init__(self, input_size, output_size, action_scale):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.output_size = output_size
        self.action_scale = action_scale
        self.t = 0

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.mul(F.tanh(self.fc3(x)), self.action_scale)

        return x

    def get_action(self, state):

        state_th = torch.tensor(state).float().to(device)
        action_th, hidden = self.forward(state_th)
        action = action_th.max(dim=-1)[1].item()

        return action

class QNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, s, a):

        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(torch.cat(x, a)))
        x = self.fc3(x)
        return x

class VNet(nn.Module):
    def __init__(self, state_size):
        super(VNet, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':

    env = gym.make(env_name)
    dir_name = '/tmp/rl_implementations/sac/{}-{}'.format(env_name, datetime.today().strftime("%Y-%d-%b-%H-%M-%S"))
    writer = SummaryWriter(dir_name)

    torch.manual_seed(seed)
    random.seed(seed)
    env.seed(seed)

    time_start = time.time()

    policy = Actor(env.observation_space.shape[0], env.action_space.shape[0],env.action_space.high)
    q_net1 = QNet(env.observation_space.shape[0], env.action_space.shape[0])
    q_net2 = QNet(env.observation_space.shape[0], env.action_space.shape[0])
    v_net = VNet(env.observation_space.shape[0])
    v_target_net = VNet(env.observation_space.shape[0])
    v_target_net.eval()
    v_target_net.load_state_dict(v_net.parameters())

    optimizer = optim.Adam(policy.parameters(), q_net1.parameters(), q_net2.parameters(), v_net.parameters())

    for ep in range(n_episodes):
        pass
