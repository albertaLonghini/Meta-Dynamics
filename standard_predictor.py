import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, params):
        super(Network, self).__init__()

        n_actions = params['t_steps']
        n_neurons = params['n_neurons']
        n_layers = params['n_layers']
        subset = params['subset']
        dlo_only = params['dlo_only']
        obj_only = params['obj_only']
        obj_input = params['obj_input']

        if subset == 0:
            in_dim = 117
        else:
            in_dim = 78

        if dlo_only == 1:
            out_dim = 32 * 2
            if obj_input == 0:
                in_dim = 66
        else:
            if obj_only == 1:
                out_dim = 3*2
            else:
                out_dim = 78

        if n_layers == 0:

            self.body = nn.Sequential(
                nn.Linear(in_dim, n_neurons),
                nn.LeakyReLU(),
                nn.Linear(n_neurons, n_neurons),
                nn.LeakyReLU()
            )

            self.head = nn.Sequential(
                nn.Linear(n_neurons+(n_actions*2), n_neurons),
                nn.LeakyReLU(),
                nn.Linear(n_neurons, out_dim),
            )

        else:

            self.body = nn.Sequential(
                nn.Linear(in_dim, n_neurons),
                nn.LeakyReLU(),
                nn.Linear(n_neurons, n_neurons),
                nn.LeakyReLU(),
                nn.Linear(n_neurons, n_neurons),
                nn.LeakyReLU()
            )

            self.head = nn.Sequential(
                nn.Linear(n_neurons + (n_actions * 2), n_neurons),
                nn.LeakyReLU(),
                nn.Linear(n_neurons, n_neurons),
                nn.LeakyReLU(),
                nn.Linear(n_neurons, out_dim),
            )


    def forward(self, x, a):

        h = self.body(x)
        h = torch.cat([h, a], -1)
        y = self.head(h)

        return y


class regressor(nn.Module):

    def __init__(self, params):

        super(regressor, self).__init__()

        self.dlo_only = params['dlo_only']
        self.obj_only = params['obj_only']
        self.obj_input = params['obj_input']
        self.displacement = params['displacement']

        self.model_theta = Network(params)

    def get_loss(self, x, a, y):

        y_hat = self.model_theta.forward(x, a)

        # predic displacement istead of final position
        if self.displacement == 1:
            if self.dlo_only == 0 and self.obj_only == 0:
                y_hat = y_hat + x
            else:
                if self.dlo_only == 1:
                    if self.obj_input == 0:
                        y_hat = y_hat + x[:, 2:]
                    else:
                        y_hat = y_hat + x[:, 2:-6]
                else:
                    y_hat = y_hat + x[:, 66:-6]

        err = (y - y_hat) ** 2

        if self.dlo_only == 1 or self.obj_only == 1:
            return torch.mean(err), None
        else:
            push_err = torch.mean(err[:, :2]).detach().cpu().item()
            dlo_err = torch.mean(err[:, 2:66]).detach().cpu().item()
            obj_pos_err = torch.mean(err[:, 66:72]).detach().cpu().item()
            obj_or_err = torch.mean(err[:, 72:]).detach().cpu().item()

            return torch.mean(err), np.array([push_err, dlo_err, obj_pos_err, obj_or_err])
