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
import numpy as np

import pdb

from rl_implementations.sac.networks import Actor, QNet

env_name = 'Pendulum-v0'
n_episodes = 10000
replay_buffer_size = 1e6
train_batch_size = 256
gamma = 0.99
target_smoothing_coeff = 0.005
lr = 3e-4
reward_scaling = 5
entropy_coeff = 1

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



def optimize_model(policy, q1_net, q2_net, q1_target_net, q2_target_net,
                   memory, actor_optimizer, q1_net_optimizer, q2_net_optimizer):
    if len(memory) < train_batch_size:
        return 0, 0, 0  # dummy losses for consistency in presenting results

    st_b, ac_b, rew_b, nst_b, dn_b = memory.sample(train_batch_size)
    states_th = torch.tensor(st_b).float().to(device)
    actions_th = torch.tensor(ac_b).to(device)
    rewards_th = torch.tensor(rew_b).unsqueeze(1).to(device)
    next_states_th = torch.tensor(nst_b).float().to(device)
    dones_th = torch.tensor(dn_b).float().unsqueeze(1).to(device)

    Q1_vals = q1_net(states_th, actions_th)
    Q2_vals = q2_net(states_th, actions_th)
    next_action_means, next_action_stds = policy(next_states_th)
    next_actions = Normal(next_action_means, next_action_stds).sample()

    with torch.no_grad():
        Q1_next_state_vals = q1_target_net(next_states_th, next_actions)
        Q2_next_state_vals = q2_target_net(next_states_th, next_actions)
        Q_next_state_minvals = torch.min(Q1_next_state_vals, Q2_next_state_vals)
        Q_target = rewards_th + gamma*Q_next_state_minvals*(1 - dones_th)

    pi_action_means, pi_action_logstd = policy(states_th)
    pi_action_stds = torch.exp(pi_action_logstd)

    z = Normal(torch.zeros_like(pi_action_means), torch.ones_like(pi_action_stds)).sample()
    newly_sampled_actions = pi_action_means + z*pi_action_stds
    newly_sampled_action_log_probs = Normal(pi_action_means, pi_action_stds).log_prob(newly_sampled_actions)
    newly_sampled_Q1_vals = q1_net(states_th, newly_sampled_actions)

    J_q1 = torch.mean((Q1_vals - Q_target)**2)
    J_q2 = torch.mean((Q2_vals - Q_target)**2)
    q1_net_optimizer.zero_grad()
    J_q1.backward()
    q1_net_optimizer.step()
    q2_net_optimizer.zero_grad()
    J_q2.backward()
    q2_net_optimizer.step()

    J_pi = torch.mean(entropy_coeff*newly_sampled_action_log_probs - newly_sampled_Q1_vals)
    actor_optimizer.zero_grad()
    J_pi.backward()
    actor_optimizer.step()
    return J_q1, J_pi


def evaluate_episode(env, policy):
    done = False
    state = env.reset()
    rew_ep = 0
    sum_std = 0
    steps = 0
    while not done:
        env.render()
        action_mean_th, action_logstd_th = policy.forward(torch.tensor(state).float().to(device))
        steps += 1
        sum_std += torch.exp(action_logstd_th).detach().cpu().numpy()
        action = torch.tanh(action_mean_th).cpu().detach().numpy()*env.action_space.high
        next_state, reward, done, _ = env.step(action)
        rew_ep += reward
        state = next_state

    print("Evaluation Reward: {}, Average Std: {}".format(rew_ep, sum_std/steps))
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
    q1_target_net = QNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    q2_target_net = QNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    q1_target_net.eval()
    q1_target_net.load_state_dict(q1_net.state_dict())
    q2_target_net.eval()
    q2_target_net.load_state_dict(q2_net.state_dict())

    actor_optimizer = optim.Adam(policy.parameters(), lr=lr)
    q1_net_optimizer = optim.Adam(q1_net.parameters(), lr=lr)
    q2_net_optimizer = optim.Adam(q2_net.parameters(), lr=lr)
    memory = ExpReplay()
    steps = 0

    for ep in range(n_episodes):
        state = env.reset()
        done = False
        rew_ep = 0

        while not done:
            steps += 1
            action = policy.get_action(state)
            next_state, reward, done, info = env.step(action)
            rew_ep += reward

            exp_tuple = (state, action, reward_scaling*reward, next_state, done)
            memory.add(exp_tuple)

            loss = optimize_model(policy, q1_net, q2_net, q1_target_net, q2_target_net, memory,
                                  actor_optimizer, q1_net_optimizer, q2_net_optimizer)

            for t_par, par in zip(q1_target_net.parameters(), q1_net.parameters()):
                t_par.data = par.data*target_smoothing_coeff + t_par.data*(1-target_smoothing_coeff)
            for t_par, par in zip(q2_target_net.parameters(), q2_net.parameters()):
                t_par.data = par.data*target_smoothing_coeff + t_par.data*(1-target_smoothing_coeff)

            state = next_state

        writer.add_scalar('train/Q-loss', loss[0], ep)
        writer.add_scalar('train/Pi-loss', loss[1], ep)
        writer.add_scalar('train/Total loss', sum(loss), ep)
        writer.add_scalar('train/steps', steps, ep)

        if ep % evaluate_freq == 0:
            rew_eval = evaluate_episode(env, policy)
            writer.add_scalar('eval/rewards', rew_eval, ep)