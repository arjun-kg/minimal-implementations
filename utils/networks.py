import torch.nn as nn
import torch


class MLP(nn.Module):
    """Multi Layer Perceptron
    Params:
    n_in (tuple of ints): Inputs are concatenated before feeding into neural net
    n_out (tuple of ints): Last layer branches out into different outputs as in n_out
    n_hidden (int): Number of hidden layers
    hidden_size (int): Size of every hidden layer
    out_fns (tuple of lambda expressions): activation fns that need to be applied at every output

    """
    def __init__(self, n_in, n_out, n_hidden, hidden_size, out_fns):
        super(MLP, self).__init__()

        self.n_out = n_out
        self.out_fns = out_fns
        self.net = nn.Sequential()
        self.net.add_module('input_layer', nn.Linear(sum(n_in), hidden_size))
        self.net.add_module('input_activ', nn.ReLU())
        for i in range(n_hidden):
            self.net.add_module('hidden_layer{}'.format(i), nn.Linear(hidden_size, hidden_size))
            self.net.add_module('hidden_activ{}'.format(i), nn.ReLU())

        self.output_layers = nn.ModuleList([nn.Linear(hidden_size, n_out[i]) for i in range(len(n_out))])

    def forward(self, *inputs):
        if len(inputs) > 1:
            x = torch.cat(inputs, dim=-1) # TODO: What will happen if inputs is len 1 if I concatenate?
        else:
            x = inputs[0]

        x = self.net(x)
        x_outs = [self.out_fns[i](self.output_layers[i](x)) for i in range(len(self.n_out))]

        if len(self.n_out) == 1:
            return x_outs[0]
        return x_outs