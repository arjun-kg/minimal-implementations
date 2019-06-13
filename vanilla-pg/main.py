import torch
import torch.nn as nn
import gym 
import pdb 
import numpy as np 
import torch.optim as optim 


n_episodes = 10000
hidden_size = 256
train_flag = True
gamma = 0.99
lr = 1e-4
eps_start = 0.8
eps_end = 0.2
temperature = n_episodes/10

env = gym.make('Pendulum-v0')
input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]
action_scale = env.action_space.high
learn_std = False
eval_every = 20


#policy = nn.Sequential(nn.Linear(input_size, hidden_size), 
#                                         nn.ReLU(), 
#                                         nn.Linear(hidden_size, output_size), 
#                                         nn.Tanh())

w1 = nn.Parameter(torch.randn(input_size, hidden_size))
b1 = nn.Parameter(0.01*torch.ones(1, hidden_size))
w2 = nn.Parameter(torch.randn(hidden_size, output_size))
b2 = nn.Parameter(0.01*torch.ones(1, output_size))

#optimizer = optim.Adam(policy.parameters())

log_action_std = nn.Parameter(-2*torch.ones(output_size))

for ep in range(n_episodes):
    done = False
    state = env.reset()
    
    states = [state]
    action_log_probs = []
    rews = []
    eps_now = eps_end + (eps_start - eps_end)*np.exp(-ep/temperature)
    while not done:
        x = torch.relu(torch.mm(torch.tensor(state).view(-1,input_size).float(), w1) + b1)
        x = torch.tensor(action_scale)*torch.tanh(torch.mm(x, w2) + b2)
        action_mean = x               
        action_std = torch.exp(log_action_std)
        action = torch.normal(action_mean,action_std)

        if eps_now > np.random.rand():
            action = action_scale*(2*torch.rand(output_size) - 1)
        log_prob = torch.distributions.Normal(action_mean, action_std).log_prob(action)
        state, rew, done, _ = env.step(action.detach())
        
        rews.append(rew)
        states.append(state)
        action_log_probs.append(log_prob)

    if train_flag:
        returns = [rews[-1]]
        for rew in reversed(rews[:-1]):
            returns.append(rew.float() + gamma*returns[-1].float())
        
        returns.reverse()
        
        loss = torch.mean(torch.stack(action_log_probs)*torch.tensor(returns).float())
        loss.backward()
        
        for par in [w1, b1, w2, b2]:
            par.data.add_(lr*par.grad.data)
            par.grad.data.zero_()
        if learn_std:
            log_action_std.data.add_(lr*log_action_std.grad.data)
            log_action_std.grad.data.zero_()
        
    if ep % eval_every == 0:
        state = env.reset()
        done = False
        action_std = torch.exp(log_action_std)            
        eval_rews = 0
        while not done:
            env.render()
            x = torch.relu(torch.mm(torch.tensor(state).view(-1,input_size).float(), w1) + b1)
            x = torch.tensor(action_scale)*torch.tanh(torch.mm(x, w2) + b2)
            action_mean = x
            action = torch.normal(action_mean,action_std)
            state, rew, done, _ = env.step(action.detach())
            eval_rews += rew
        
        print("Episodes: {}, Eval rews: {}, Eps now: {}".format(ep, eval_rews, round(eps_now,2)))
            
        