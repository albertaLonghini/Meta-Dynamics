import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class MamlParams(nn.Module):
    def __init__(self, n_actions, n_neurons, n_layers, subset, dlo_only, lr):
        super(MamlParams, self).__init__()

        if subset == 0:
            in_dim = 117
        else:
            in_dim = 78

        if dlo_only == 1:
            out_dim = 32*2
        else:
            out_dim = 78

        self.n_layers = n_layers

        if n_layers == 0:

            self.theta_shapes = [[n_neurons, in_dim], [n_neurons],
                                 [n_neurons, n_neurons], [n_neurons],
                                 [n_neurons, n_neurons+(n_actions*2)], [n_neurons],
                                 [out_dim, n_neurons], [out_dim]]

        else:

            self.theta_shapes = [[n_neurons, in_dim], [n_neurons],
                                 [n_neurons, n_neurons], [n_neurons],
                                 [n_neurons, n_neurons], [n_neurons],
                                 [n_neurons, n_neurons + (n_actions * 2)], [n_neurons],
                                 [n_neurons, n_neurons], [n_neurons],
                                 [out_dim, n_neurons], [out_dim]]

        # self.batch_norm1 = nn.BatchNorm2d(self.filters, track_running_stats=False)
        # self.batch_norm2 = nn.BatchNorm2d(self.filters, track_running_stats=False)
        # self.batch_norm3 = nn.BatchNorm2d(self.filters, track_running_stats=False)
        # self.batch_norm4 = nn.BatchNorm2d(self.filters, track_running_stats=False)

        # self.max_pool = nn.MaxPool2d(2)

        self.lr = nn.ParameterList([nn.Parameter(torch.tensor(lr))] * len(self.theta_shapes))

        self.theta_0 = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.theta_shapes])
        for i in range(len(self.theta_0)):
            if self.theta_0[i].dim() > 1:
                torch.nn.init.kaiming_uniform_(self.theta_0[i])

    def get_theta(self):
        return self.theta_0

    def forward(self, x, a, theta=None):

        if theta is None:
            theta = self.theta_0

        if self.n_layers == 0:

            h = F.leaky_relu(F.linear(x, theta[0], bias=theta[1]))
            h = F.leaky_relu(F.linear(h, theta[2], bias=theta[3]))
            h = torch.cat([h, a], -1)
            h = F.leaky_relu(F.linear(h, theta[4], bias=theta[5]))
            y = F.linear(h, theta[6], bias=theta[7])

        else:

            h = F.leaky_relu(F.linear(x, theta[0], bias=theta[1]))
            h = F.leaky_relu(F.linear(h, theta[2], bias=theta[3]))
            h = F.leaky_relu(F.linear(h, theta[4], bias=theta[5]))
            h = torch.cat([h, a], -1)
            h = F.leaky_relu(F.linear(h, theta[6], bias=theta[7]))
            h = F.leaky_relu(F.linear(h, theta[8], bias=theta[9]))
            y = F.linear(h, theta[10], bias=theta[11])

        return y


class maml(nn.Module):

    def __init__(self, params):

        super(maml, self).__init__()

        self.inner_lr = 1.0
        self.dlo_only = params['dlo_only']
        self.adapt_lr = params['learned_lr']

        # TODO: ???
        # self.gradient_steps = params['gradient_steps']



        self.model_theta = MamlParams(params['t_steps'], params['n_neurons'], params['n_layers'], params['subset'], params['dlo_only'], self.inner_lr)

        self.log_grads_idx = 0
        self.grads_vals = np.zeros(len(self.model_theta.get_theta()))

    def adapt(self, x, a, y, train=False):

        theta_i = self.model_theta.get_theta()

        L, _ = self.get_loss(x, a, y)
        theta_grad_s = torch.autograd.grad(outputs=L, inputs=theta_i, create_graph=True)
        if train:
            if self.adapt_lr == 1:
                theta_i = list(map(lambda p: p[1] - p[2] * p[0], zip(theta_grad_s, theta_i, self.model_theta.lr)))
            else:
                theta_i = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(theta_grad_s, theta_i)))
            for i, grad in enumerate(theta_grad_s):
                self.grads_vals[i] += torch.mean(torch.abs(grad))
        else:
            if self.adapt_lr == 1:
                theta_i = list(map(lambda p: p[1] - p[2].detach() * p[0].detach(), zip(theta_grad_s, theta_i, self.model_theta.lr)))
            else:
                theta_i = list(map(lambda p: p[1] - self.inner_lr * p[0].detach(), zip(theta_grad_s, theta_i)))

        return theta_i, L.detach().cpu().item()

    def get_loss(self, x, a, y, theta_i=None):

        y_hat = self.model_theta.forward(x, a, theta_i)

        err = (y - y_hat)**2

        if self.dlo_only == 1:
            return torch.mean(err), None
        else:
            push_err = torch.mean(err[:, :2]).detach().cpu().item()
            dlo_err = torch.mean(err[:, 2:66]).detach().cpu().item()
            obj_pos_err = torch.mean(err[:, 66:72]).detach().cpu().item()
            obj_or_err = torch.mean(err[:, 72:]).detach().cpu().item()

            return torch.mean(err), np.array([push_err, dlo_err, obj_pos_err, obj_or_err])
