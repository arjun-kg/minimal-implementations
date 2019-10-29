import torch
import math
import numpy as np
import pdb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density

def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))

    flat_params = torch.cat(params)
    return flat_params

def evaluate(agent, env):
    state = env.reset()
    ep_rew = 0
    done = False
    while not done:
        action = agent.get_action(state, eval=True)
        new_obs, rew, done, _ = env.step(np.tanh(action))
        ep_rew += rew
        state = new_obs
        # env.render()
        # print(np.tanh(action))
    return ep_rew
