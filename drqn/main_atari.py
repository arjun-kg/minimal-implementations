import gym
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from scipy.misc import imresize
import time
from datetime import datetime
from tensorboardX import SummaryWriter

import pdb

env_name = 'PongNoFrameskip-v4'
n_episodes = 50000
seq_len = 10
train_batch_size = 32
optim_steps_per_ep = 10
gamma = 0.99
target_update_freq = 2
eps_start = 1
eps_end = 0.1
temperature = 1e7
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
        self.max_len = 1e5

    def add(self, exp_tuple):
        if len(self.states) < self.max_len:
            self.states.append(0)
            self.actions.append(0)
            self.rewards.append(0)
            self.dones.append(0)

        state_seq, action_seq, reward_seq, done = exp_tuple
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


class DRQN(nn.Module):
    def __init__(self, output_size):
        super(DRQN, self).__init__()

        self.cnn1 = nn.Conv2d(1, 32, 8, 4)
        self.cnn2 = nn.Conv2d(32, 64, 3, 2)
        self.cnn3 = nn.Conv2d(64, 64, 3, 1)
        self.lstm1 = nn.LSTM(7*7*64, 512, batch_first=True)
        self.fc1 = nn.Linear(512, output_size)

        self.output_size = output_size
        self.eps = eps_start
        self.t = 0

    def get_action(self, state, hidden, eval=False):

        state_th = torch.tensor(state).float().to(device)
        action_th, hidden = self.forward(state_th, hidden)
        action = action_th.max(dim=-1)[1].item()

        rnd_num = np.random.rand()

        if not eval:
            self.eps = eps_end + (eps_start - eps_end)*np.exp(-self.t/temperature)
            self.t += 1
            if rnd_num < self.eps:
                action = np.random.randint(self.output_size)
        return action, hidden

    def forward(self, x, hidden):
        if len(x.shape) > 2:
            n_b = x.size(0)
            n_t = x.size(1)

        else:
            n_b = 1
            n_t = 1

        x = x.contiguous().view(n_b * n_t, 1, x.size(-2), x.size(-1))
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))

        x, hidden = self.lstm1(x.view(n_b, n_t, -1), hidden)
        x = self.fc1(x)

        return x, hidden


def optimize_model(policy, target_net, memory, optimizer):
    if len(memory) < train_batch_size:
        return

    st_b, ac_b, rew_b, dn_b = memory.sample(train_batch_size)

    states_th = torch.tensor(st_b).float().unsqueeze(2).to(device)
    actions_th = torch.tensor(ac_b).unsqueeze(-1).to(device)
    rewards_th = torch.tensor(rew_b).to(device)
    dones_th = torch.tensor(dn_b).float().to(device)

    hidden = (torch.zeros(1, train_batch_size, 512).to(device),
              torch.zeros(1, train_batch_size, 512).to(device))
    state_action_values, _ = policy.forward(states_th[:, :-1], hidden)
    state_action_values = state_action_values.gather(-1, actions_th)

    next_state_action_values, _ = target_net.forward(states_th[:, 1:], hidden)
    next_state_action_values = next_state_action_values.max(2)[0]

    target_values = rewards_th + gamma*next_state_action_values*(1-dones_th)
    loss = nn.MSELoss()(state_action_values.squeeze(-1), target_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def evaluate_episode(env, policy):
    done = False
    state = env.reset()
    p_state = preprocess(state)
    hidden = (torch.zeros(1, 1, 512).to(device),
              torch.zeros(1, 1, 512).to(device))
    steps = 0
    rew_ep = 0
    while not done:
        env.render()
        action, hidden = policy.get_action(p_state, hidden, eval=True)
        next_state, reward, done, _ = env.step(action + 2)
        rew_ep += reward
        if next_state is not None:
            p_state = preprocess(next_state)
        steps += 1

    print("Evaluation Reward: {}".format(rew_ep))
    return rew_ep


def preprocess(im):
    # image expects 210 x 160 x 3, outputs 84 x 84 grayscale, cropped to have main play area
    im = 0.2989*im[:, :, 0] + 0.5870*im[:, :, 1] + 0.1140*im[:, :, 2]
    im = imresize(im, (110, 84))
    im = im[18:102, :]
    return im


if __name__ == '__main__':

    env = gym.make(env_name)
    dir_name = '/tmp/rl_implementations/drqn/{}-{}'.format(env_name, datetime.today().strftime("%Y-%d-%b-%H-%M-%S"))
    writer = SummaryWriter(dir_name)

    torch.manual_seed(seed)
    random.seed(seed)
    env.seed(seed)

    time_start = time.time()

    policy = DRQN(2).to(device)
    if load_path is not None:
        policy.load_state_dict(torch.load(load_path))
        print("Policy loaded from '{}'".format(load_path))
    target_net = DRQN(2).to(device)
    target_net.load_state_dict(policy.state_dict())
    target_net.eval()
    memory = ExpReplay()
    optimizer = optim.Adam(policy.parameters())
    optim_steps = 0

    for ep in range(n_episodes):
        time_ep_start = time.time()
        state = env.reset()
        p_state = preprocess(state)
        done = False
        hidden = (torch.zeros(1, 1, 512).to(device),
                  torch.zeros(1, 1, 512).to(device))
        state_seq = [p_state]
        action_seq = []
        reward_seq = []
        done_seq = []

        rew_ep = 0

        while not done:
            action, hidden = policy.get_action(p_state, hidden)
            next_state, reward, done, info = env.step(action+2)  # pong actions are 2 and 3
            rew_ep += reward
            if next_state is not None:
                p_next_state = preprocess(next_state)
            else:
                p_next_state = None

            if len(action_seq) < seq_len:
                state_seq.append(p_next_state)
                action_seq.append(action)
                reward_seq.append(reward)
                done_seq.append(done)
            else:
                exp_tuple = (state_seq, action_seq, reward_seq, done_seq)
                memory.add(exp_tuple)
                state_seq = [p_state, p_next_state]
                action_seq = [action]
                reward_seq = [reward]
                done_seq = [done]

                hidden = (torch.zeros(1, 1, 512).to(device),
                          torch.zeros(1, 1, 512).to(device))

            p_state = p_next_state

            if done:
                pad_len = seq_len - len(action_seq)
                state_seq.extend([np.zeros_like(p_state) for _ in range(pad_len)])
                action_seq.extend([0 for _ in range(pad_len)])
                reward_seq.extend([0 for _ in range(pad_len)])
                done_seq.extend([True for _ in range(pad_len)])
                exp_tuple = (state_seq, action_seq, reward_seq, done_seq)
                memory.add(exp_tuple)

        for _ in range(optim_steps_per_ep):
            loss = optimize_model(policy, target_net, memory, optimizer)
            optim_steps += 1
            writer.add_scalar('train/loss', loss, optim_steps)
        if ep % target_update_freq == 0:
            target_net.load_state_dict(policy.state_dict())

        if ep % save_freq == 0 and ep != 0:
            torch.save(policy.state_dict(), dir_name+'/policy_net_{}'.format(ep))
            print("Saved policy after episode {} in '{}'".format(ep, dir_name+'/policy_net_{}'.format(ep)))

        print("\nEpisode: {}\nTrain Timesteps: {}\nOptimization Steps: {}\nTime Elapsed: {}\nEpsilon: "
              "{}\nEpisode Reward: {}\n"
              .format(ep, policy.t, optim_steps, time.time() - time_start, policy.eps, rew_ep))
        writer.add_scalar('train/epsilon', policy.eps, ep)
        writer.add_scalar('train/rew', rew_ep, ep)

        if ep % evaluate_freq == 0:
            rew_eval = evaluate_episode(env, policy)
            writer.add_scalar('eval/rew', rew_eval, ep)

    print("Training completed in {} seconds".format(time.time() - time_start))