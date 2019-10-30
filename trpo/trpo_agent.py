from rl_implementations.utils.networks import MLP
from rl_implementations.trpo.utils import normal_log_density, set_flat_params_to, get_flat_params_from
import torch.optim as optim
import math
from torch.distributions import Normal
import torch
import pdb

logstd_min = -20
logstd_max = 2
gamma = 0.99
lr = 1e-3
max_kl = 1e-2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TRPOAgent:
    def __init__(self, n_obs, n_act):
        self.policy_net = MLP((n_obs,), (n_act, n_act), 3, 256, (lambda x: x, lambda x: torch.clamp(x, min=logstd_min, max=logstd_max))).to(device)
        self.value_net = MLP((n_obs,), (1,), 3, 256, (lambda x: x,)).to(device)

        self.value_net_optimizer = optim.Adam(self.value_net.parameters(), lr)
        self.train_steps = 0

    def get_action(self, state, eval=False):
        state_th = torch.tensor(state).to(device).float()
        action_mean_th, action_log_std_th = self.policy_net(state_th)
        action_std_th = action_log_std_th.exp()
        if eval:
            return action_mean_th.cpu().detach().numpy()
        else:
            action_th = torch.normal(action_mean_th, action_std_th)
            return action_th.detach().cpu().numpy()

    def get_log_probs(self, states, actions):
        action_means_th, action_log_stds_th = self.policy_net(states)
        action_stds_th = action_log_stds_th.exp()
        log_probs = normal_log_density(actions, action_means_th, action_log_stds_th, action_stds_th)
        return log_probs


    def optimize_model(self, trajectories, writer):
        self.train_steps += 1
        states, states2, actions, rewards, dones = trajectories

        states_th = torch.tensor(states).to(device).float()
        states2_th = torch.tensor(states2).to(device).float()

        actions_th = torch.tensor(actions).to(device)
        # rewards_th = torch.tensor(rewards).to(device).unsqueeze(-1)
        # dones_th = torch.tensor(dones).to(device).unsqueeze(-1)

        log_probs_th = self.get_log_probs(states2_th, actions_th)
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

            if dones[i] and i != len(rewards) - 1:
                st_p -= 2
            else:
                st_p -= 1
        returns.reverse()
        advantages.reverse()
        advantages_th = torch.cat(advantages).to(device)
        advantages_th = advantages_th/advantages_th.norm()
        returns_th = torch.tensor(returns).to(device)

        loss = -torch.mean(torch.exp(log_probs_th - log_probs_th.detach())*advantages_th)
        g = torch.autograd.grad(loss, self.policy_net.parameters(), retain_graph=True)
        loss_grad = torch.cat([grad.view(-1) for grad in g]).detach()
        loss_grad = loss_grad/loss_grad.norm()
        s = self.conjugate_gradient(states2_th, -loss_grad, 10)

        shs = 0.5 * (s.dot(self.Fvp_direct(states2_th, s)))
        lm = math.sqrt(max_kl / shs)
        fullstep = s * lm
        expected_improve = -loss_grad.dot(fullstep)
        prev_params = get_flat_params_from(self.policy_net)
        success, new_params = self.line_search(advantages_th, states2_th, actions_th, prev_params, fullstep, expected_improve)
        set_flat_params_to(self.policy_net, new_params)

        v_net_loss = torch.mean((returns_th.detach() - values_th)**2)
        self.value_net_optimizer.zero_grad()
        v_net_loss.backward()
        self.value_net_optimizer.step()

        writer.add_scalar('train/v_loss', v_net_loss, self.train_steps)
        writer.add_scalar('train/pi_loss', loss, self.train_steps)


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
        flat_grad_grad_v = torch.cat([grad.contiguous().view(-1) for grad in grads_grads_v]).detach()
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
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
            if rdotr < residue_limit:
                break
        return x

    def line_search(self, advantages_th, states_th, actions_th, x, fullstep, expected_improve_full, max_backtracks=10, accept_ratio=0.1):
        log_probs_th = self.get_log_probs(states_th, actions_th)
        loss_val = -torch.mean(torch.exp(log_probs_th - log_probs_th.detach())*advantages_th)
        # pdb.set_trace()
        for stepfrac in [.5 ** i for i in range(max_backtracks)]:
            x_new = x + stepfrac * fullstep
            set_flat_params_to(self.policy_net, x_new)

            new_log_probs_th = self.get_log_probs(states_th, actions_th)
            new_loss_val = -torch.mean(torch.exp(new_log_probs_th - log_probs_th.detach())*advantages_th)
            actual_improve = loss_val - new_loss_val
            expected_improve = expected_improve_full * stepfrac
            ratio = actual_improve / expected_improve

            if ratio > accept_ratio:
                print("YASS")
                return True, x_new
        return False, x
