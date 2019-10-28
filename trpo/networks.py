import torch
import torch.nn as nn
from torch.distributions import Categorical
n_hidden = 256
init_w = 1e-3


class PolicyNet(nn.Module):
    def __init__(self, n_obs, n_act, type='discrete'):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(n_obs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc_out = nn.Linear(n_hidden, n_act)
        self.fc_out.weight.data.uniform_(-init_w, init_w)
        self.fc_out.bias.data.uniform_(-init_w, init_w)

        self.type = type

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc_out(x)

        return x

    def get_action(self, state_np):
        state_th = torch.tensor(state_np).float()
        action_th = self.forward(state_th)

        if self.type == 'discrete':
            action_sampled_th = Categorical(logits=action_th).sample()
        else:
            raise NotImplementedError

        action_sampled_np = action_sampled_th.cpu().detach().numpy()
        return action_sampled_np


class ValueNet(nn.Module):
    def __init__(self, n_obs):
        super(ValueNet, self).__init__()

        self.fc1 = nn.Linear(n_obs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc_out = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc_out(x)

        return x
