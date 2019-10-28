from rl_implementations.utils.networks import MLP
from rl_implementations.trpo.utils import normal_log_density
import torch.optim as optim
from torchviz import make_dot
from torch.distributions import Normal
import torch
import pdb

logstd_min = -20
logstd_max = 2
gamma = 0.99
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TRPOAgent:
    def __init__(self, n_obs, n_act):
        self.policy_net = MLP((n_obs,), (n_act, n_act), 3, 256, (lambda x: x, lambda x: torch.clamp(x, min=logstd_min, max=logstd_max))).to(device)
        self.value_net = MLP((n_obs,), (1,), 3, 256, (lambda x: x,)).to(device)

        # self.policy_net_optimizer =
        self.value_net_optimizer = optim.Adam(self.value_net.parameters(), lr)

    def get_action(self, state, eval=False):
        state_th = torch.tensor(state).to(device).float()
        action_mean_th, action_log_std_th = self.policy_net(state_th)
        action_std_th = action_log_std_th.exp()
        if eval:
            return action_mean_th.cpu().detach().numpy()
        else:
            z = torch.normal(torch.zeros(action_mean_th.size()), torch.ones(action_mean_th.size())).to(device)
            action_th = action_mean_th + action_std_th*z
            logprob_th = normal_log_density(action_th, action_mean_th, action_log_std_th, action_std_th)
            return action_th.detach().cpu().numpy(), logprob_th

    def optimize_model(self, trajectories):
        states, actions, log_probs, rewards, dones = trajectories

        states_th = torch.tensor(states).to(device).float()
        actions_th = torch.tensor(actions).to(device)
        log_probs_th = torch.cat(log_probs).to(device)
        rewards_th = torch.tensor(rewards).to(device).unsqueeze(-1)
        dones_th = torch.tensor(dones).to(device).unsqueeze(-1)

        values_th = self.value_net(states_th)

        # rewards to go:
        returns = []
        advantages = []
        prev_ret = 0
        st_p = len(states) - 1  # states pointer, since that list has a different size
        for i in range(len(rewards)-1, -1, -1):
            v_next = values_th[st_p]
            prev_ret = rewards[i] + gamma*prev_ret*(1-dones[i]) + gamma*v_next*dones[i]
            returns.append(prev_ret)
            adv = rewards[i] + gamma*v_next - values_th[st_p-1]
            advantages.append(adv)

            if dones[i]:
                st_p -= 2
            else:
                st_p -= 1
        returns.reverse()
        advantages.reverse()

        advantages_th = torch.tensor(advantages).to(device)
        returns_th = torch.tensor(returns).to(device)

        loss = -torch.mean(log_probs_th*advantages_th)
        g = torch.autograd.grad(loss, self.policy_net.parameters(), retain_graph=True, allow_unused=True)
        g = torch.cat([grad.view(-1) for grad in g]).detach()
        x = self.conjugate_gradient(states_th, g, 100)

        pdb.set_trace()
        v_net_loss = torch.mean((returns_th - values_th)**2)
        self.value_net_optimizer.zero_grad()
        v_net_loss.backward()
        self.value_net_optimizer.step()

    def get_kl(self, x):
        mean, log_std = self.policy_net(x)
        std = torch.exp(log_std)
        mean_detached = mean.detach()
        log_std_detached = log_std.detach()
        std_detached = std.detach()
        kl = log_std - log_std_detached + (std_detached.pow(2) + (mean_detached - mean).pow(2)) / (2.0 * std.pow(2)) - 0.5
        return kl.sum()

    def Fvp_direct(self, states, v):
        damping = 1e-2
        pa_sum = self.get_kl(states)
        # compute the first derivative of the loss wrt the network parameters and flatten into a vector
        grads = torch.autograd.grad(pa_sum, self.policy_net.parameters(), create_graph=True)
        grads_flat = torch.cat([grad.view(-1) for grad in grads])
        # compute the dot product with the input vector
        grads_v = torch.sum(grads_flat * v)
        # now compute the derivative again.
        grads_grads_v = torch.autograd.grad(grads_v, self.policy_net.parameters(), create_graph=False)
        flat_grad_grad_v = torch.cat([grad.contiguous().view(-1) for grad in grads_grads_v]).data
        return flat_grad_grad_v + v * damping

    def conjugate_gradient(self, states, b, n_steps, residue_limit=1e-4):
        x = torch.zeros(b.size()).to(device)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(n_steps):
            Avp = self.Fvp_direct(states, p)
            alpha = rdotr / torch.dot(p, Avp)
            x += alpha * p
            r -= alpha * Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residue_limit:
                break
        return x
