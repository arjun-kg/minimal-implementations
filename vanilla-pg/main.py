import torch
import torch.nn as nn
import gym
import pdb
import numpy as np
import torch.optim as optim
from tensorboardX import SummaryWriter
from datetime import datetime

n_episodes = 5000
hidden_size = 6
train_flag = True
gamma = 0.99
eps_start = 0
eps_end = 0.0
temperature = n_episodes/10
seed = 0

env = gym.make('CartPole-v1')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
eval_every = 25
np.random.seed(seed)
env.seed(seed)
torch.manual_seed(seed)

w1 = nn.Parameter(torch.zeros(input_size, hidden_size))
nn.init.xavier_uniform_(w1)
b1 = nn.Parameter(torch.zeros(1, hidden_size))
w2 = nn.Parameter(torch.zeros(hidden_size, output_size))
nn.init.xavier_uniform_(w2)
b2 = nn.Parameter(torch.zeros(1, output_size))

trainable_params = [w1, w2, b1, b2]
optimizer = optim.Adam(trainable_params)
writer = SummaryWriter('/tmp/rl_implementations/vpg/{}'.format(datetime.today().strftime("%Y-%d-%b-%H-%M-%S")))

for ep in range(n_episodes):
    done = False
    state = env.reset()
    states = [state]
    action_log_probs = []
    rews = []
    eps_now = eps_end + (eps_start - eps_end)*np.exp(-ep/temperature)

    while not done:
        x = nn.ELU()(torch.mm(torch.tensor(state).view(-1,input_size).float(), w1) + b1)
        out = torch.tanh(torch.mm(x, w2) + b2)

        action_prob = torch.softmax(out, dim=-1)
        action_log_prob = torch.log_softmax(out, dim=-1)  # taking log of action_prob is numerically unstable
        # action = torch.multinomial(action_prob, 1).item()
        action = action_prob.max(-1)[1].item()
        if eps_now > np.random.rand():
            action = np.random.randint(output_size)
        state, rew, done, _ = env.step(action)

        rews.append(rew)
        states.append(state)
        selected_action_log_prob = torch.index_select(action_log_prob, dim=-1, index=torch.tensor(action)).view(1, 1)
        action_log_probs.append(selected_action_log_prob)

    if train_flag:
        returns = [rews[-1]]
        for rew in reversed(rews[:-1]):
            returns.append(rew + gamma*returns[-1])
        returns.reverse()
        # policy loss
        loss = -torch.mean(torch.stack(action_log_probs).squeeze(1).squeeze(1)*torch.tensor(returns).float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    writer.add_scalar('train/Epsilon', eps_now, ep)
    writer.add_scalar('train/Loss', loss, ep)

    if ep % eval_every == 0:
        state = env.reset()
        done = False
        eval_rews = 0
        while not done:
            # env.render()
            x = nn.ELU()(torch.mm(torch.tensor(state).view(-1, input_size).float(), w1) + b1)
            out = torch.tanh(torch.mm(x, w2) + b2)

            # action_probs = torch.softmax(out, dim=-1)
            # action = torch.multinomial(action_probs, 1).item()
            action = out.max(-1)[1].item()
            state, rew, done, _ = env.step(action)
            eval_rews += rew

        print("Episodes: {}, Eval rews: {}, Eps now: {}".format(ep, eval_rews, round(eps_now,2)))
        writer.add_scalar('eval/rewards', eval_rews, ep)
