import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal

hidden_size = 256
logstd_min = -20
logstd_max = 2
init_w = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, input_size, output_size, action_scale):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, output_size)
        self.fc_logstd = nn.Linear(hidden_size, output_size)
        self.fc_mean.weight.data.uniform_(-init_w, init_w)
        self.fc_mean.bias.data.uniform_(-init_w, init_w)
        self.fc_logstd.weight.data.uniform_(-init_w, init_w)
        self.fc_logstd.bias.data.uniform_(-init_w, init_w)
        self.output_size = output_size
        self.action_scale = action_scale

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_m = self.fc_mean(x)
        x_ls = torch.clamp(self.fc_logstd(x), min=logstd_min, max=logstd_max)

        return x_m, x_ls

    def get_action(self, state):

        state_th = torch.tensor(state).float().to(device)
        action_mean_th, action_logstd_th = self.forward(state_th)
        action_th_sampled = torch.tanh(Normal(action_mean_th, torch.exp(action_logstd_th)).sample())*\
                            torch.tensor(self.action_scale).to(device)
        action = action_th_sampled.cpu().detach().numpy()
        return action


class QNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, s, a):
        x = torch.cat((s,a), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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

