import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import random
from datetime import datetime
import time
from tensorboardX import SummaryWriter

import pdb


env_name = 'Pendulum-v0'
n_episodes = 5000
hidden_size = 256
replay_buffer_size = 1e6
train_batch_size = 256
gamma = 0.99
target_smoothing_coeff = 0.005
lr = 3e-4
std_dev = 1

evaluate_freq = 25
seed = 0
load_path = None
save_freq = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExpReplay:
    def __init__(self):
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.pointer = 0
        self.max_len = replay_buffer_size

    def add(self, exp_tuple):
        if len(self.states) < self.max_len:
            self.states.append(0)
            self.actions.append(0)
            self.rewards.append(0)
            self.next_states.append(0)
            self.dones.append(0)

        state, action, reward, next_state, done = exp_tuple
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = next_state
        self.dones[self.pointer] = done
        self.pointer = int((self.pointer + 1) % self.max_len)

    def sample(self, batch_size):
        st_b, ac_b, rew_b, nst_b, dn_b = \
            zip(*random.sample(list(zip(self.states,self.actions,self.rewards, self.next_states, self.dones)),
                               batch_size))
        return st_b, ac_b, rew_b, nst_b, dn_b

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
        x = torch.mul(torch.tanh(self.fc3(x)), torch.tensor(self.action_scale).to(device))

        return x

    def get_action(self, state):

        state_th = torch.tensor(state).float().to(device)
        action_mean_th = self.forward(state_th)
        action_th_sampled = Normal(action_mean_th, std_dev).sample()
        action = action_th_sampled.cpu().detach().numpy()
        return action


class QNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, s, a):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(torch.cat((x, a), dim=-1)))
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

def optimize_model(policy, q1_net, q2_net, v_net, v_target_net, memory, optimizer):
    if len(memory) < train_batch_size:
        return
    st_b, ac_b, rew_b, nst_b, dn_b = memory.sample(train_batch_size)
    states_th = torch.tensor(st_b).float().to(device)
    actions_th = torch.tensor(ac_b).to(device)
    rewards_th = torch.tensor(rew_b).to(device)
    next_states_th = torch.tensor(nst_b).float().to(device)
    dones_th = torch.tensor(dn_b).float().to(device)

    Q_vals = q1_net(states_th, actions_th)
    V_vals = v_net(states_th)
    V_next_state_vals = v_target_net(next_states_th)
    pi_action_means = policy(states_th)
    pi_action_log_probs = Normal(pi_action_means, std_dev).log_prob(actions_th)

    newly_sampled_actions = Normal(pi_action_means, std_dev).sample()
    newly_sampled_action_log_probs = Normal(pi_action_means, std_dev).log_prob(newly_sampled_actions)
    newly_sampled_Q_vals = q1_net(states_th, newly_sampled_actions)

    J_v = torch.sum((V_vals - (Q_vals - pi_action_log_probs))**2)
    J_q = torch.sum((Q_vals - (rewards_th + gamma*V_next_state_vals*(1-dones_th)))**2)
    J_pi = torch.sum(newly_sampled_action_log_probs - newly_sampled_Q_vals)

    loss = J_v + J_q + J_pi
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def evaluate_episode(env, policy):
    done = False
    state = env.reset()
    rew_ep = 0
    while not done:
        env.render()
        action = policy.get_action(state)
        next_state, reward, done, _ = env.step(action)
        rew_ep += reward
        if next_state is not None:
            state = next_state

    print("Evaluation Reward: {}".format(rew_ep))
    return rew_ep

if __name__ == '__main__':

    env = gym.make(env_name)
    dir_name = '/tmp/rl_implementations/sac/{}-{}'.format(env_name, datetime.today().strftime("%Y-%d-%b-%H-%M-%S"))
    writer = SummaryWriter(dir_name)

    torch.manual_seed(seed)
    random.seed(seed)
    env.seed(seed)

    time_start = time.time()

    policy = Actor(env.observation_space.shape[0], env.action_space.shape[0],env.action_space.high).to(device)
    q1_net = QNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    q2_net = QNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    v_net = VNet(env.observation_space.shape[0]).to(device)
    v_target_net = VNet(env.observation_space.shape[0]).to(device)
    v_target_net.eval()
    v_target_net.load_state_dict(v_net.state_dict())

    optimizer = optim.Adam([*policy.parameters(), *q1_net.parameters(), *q2_net.parameters(), *v_net.parameters()])
    memory = ExpReplay()

    for ep in range(n_episodes):
        state = env.reset()
        done = False
        rew_ep = 0

        while not done:
            action = policy.get_action(state)
            next_state, reward, done, info = env.step(action)
            rew_ep += reward

            exp_tuple = (state, action, reward, next_state, done)
            memory.add(exp_tuple)

            optimize_model(policy, q1_net, q2_net, v_net, v_target_net, memory, optimizer)

            for t_par, par in zip(v_target_net.parameters(), v_net.parameters()):
                t_par.data = par.data*target_smoothing_coeff + t_par.data*(1-target_smoothing_coeff)

            state = next_state


        if ep % evaluate_freq == 0:
            rew_eval = evaluate_episode(env, policy)