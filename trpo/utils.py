import torch
import math
import pdb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density


