# Based on https://github.com/lucidrains/siren-pytorch
import torch
from torch import nn
from math import sqrt


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    '''Implements a single SIREN layer.
    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        w0 (float): w0 parameter from SIREN paper.
        c (float): c value from SIREN paper used for weight initialization.
        is_first (bool): Whether this is first layer of model.
        use_bias (bool): Whether the bias is used.
        activation (torch.nn.Module): Activation function. If None, defaults to Sine activation.
    '''
    def __init__(self, dim_in, dim_out, w0=30., c=6., 
                 is_first=False, use_bias=False, activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)
        # Initialize layers following SIREN paper
        w_std = (1 / dim_in) if self.is_first else (sqrt(c / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias: nn.init.uniform_(self.linear.bias, -w_std, w_std)
        self.activation = Sine(w0) if activation is None else activation
        # self.bn = nn.BatchNorm1d(dim_out)

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        # out = self.bn(out)
        
        return out


class LSSRNModel(nn.Module):
    '''LSSRN model.
    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether the bias is used.
        final_activation (torch.nn.Module): Activation function.
    '''
    def __init__(self, dim_in, dim_hidden, dim_out=4, num_layers=1, w0=30.,
                 w0_initial=30., use_bias=False, activation=None, final_activation=None):
        super().__init__()
        layers = []
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden
            layers.append(SirenLayer(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                is_first=is_first,
                use_bias=use_bias,
                activation=activation
            ))
        self.net = nn.Sequential(*layers)
        final_activation = nn.Sigmoid() if final_activation is None else final_activation
        self.last_layer = SirenLayer(dim_in=dim_hidden, dim_out=dim_out, w0=w0,
                                     use_bias=use_bias, activation=final_activation)

    def forward(self, x):
        x = self.net(x)

        return self.last_layer(x)
