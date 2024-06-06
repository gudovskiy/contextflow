import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from einops import rearrange, repeat
from .flowlayer import FlowLayer
from .simple_vit import SimpleViT, posemb_sincos_2d
from utils.helpers import freeze_parameters
from .autoregressive import MaskedResidualBlock2d, MaskedResidualBlockLinear, MaskedLinear
from datasets.ts import InputDropout

kwargs_act = {'softplus': nn.Softplus(), 'relu': nn.ReLU(), 'elu': nn.ELU(), 'gelu': nn.GELU(), 'tanh': nn.Tanh(), 'selu': nn.SELU(), 'sigmoid': nn.Sigmoid()}


class MaskedCoupling(FlowLayer):
    def __init__(self, data_channels, kernel_size=(1,1), padding=(0,0), context_net=None, contextflow=False, mask_type='B'):
        super().__init__()
        D = O = data_channels
        K = kernel_size
        P = padding
        self.context_net = context_net
        self.contextflow = contextflow
        self.NN = MaskedResidualBlock2d(D, D, kernel_size=K, padding=P, D=D, mask_type=mask_type)
        if self.context_net:
            if not self.contextflow:
                self.NN = MaskedResidualBlock2d(2*D, D, kernel_size=K, padding=P, D=D, mask_type=mask_type)
            else: freeze_parameters(self.NN)
            
            self.C = C = self.context_net.C
            self.CN = MaskedResidualBlockLinear(C, D, D)

    def get_xs_logs_t(self, x, context=None):
        B, _, H, W = x.shape
        if self.context_net:
            c, logp_c = self.context_net(context)
            logp_c = logp_c * H * W
            if self.contextflow:
                h = self.NN(x) + rearrange(self.CN(c), 'b c -> b c 1 1')
            else:
                h = self.NN(torch.concat((x, repeat(self.CN(c), 'b c -> b c h w', h=H, w=W)), dim=1))
        else:
            h = self.NN(x)
            logp_c = torch.zeros(B).to(device=x.device)

        h = rearrange(h, 'b (p c) h w -> p b c h w', p=2)
        t = h[0]

        logs_range = 2.0
        log_s = logs_range * torch.tanh(h[1] / logs_range)
        s = torch.exp(log_s)
        return s, t, log_s, logp_c

    def forward(self, x, context=None):
        s, t, log_s, logp_c = self.get_xs_logs_t(x, context)
        z = x * s + t
        return z, log_s.flatten(start_dim=1).sum(-1) + logp_c
    # to update:
    def reverse(self, z, context=None):
        #s, t, log_s, _ = self.get_xs_logs_t(z, context)
        #x = (z[1] - t) / s
        x = torch.zeros_like(z)
        #for d in range(x.shape[1]):
        #    elementwise_params = self.autoregressive_net(x)
        #    x[:,d] = self._elementwise_inverse(z[:,d], elementwise_params[:,d])
        return x

    def logdet(self, input, context=None):
        _, ldj = self.forward(input, context)
        return ldj


class MaskedCouplingFC(MaskedCoupling):
    def __init__(self, data_channels, kernel_size=(1,1), padding=(0,0), context_net=None, contextflow=False):
        super().__init__(data_channels, kernel_size=(1,1), padding=(0,0), context_net=None, contextflow=False)
        self.D = data_channels

    def forward(self, x, context=None):
        x = x.view(-1, self.D, 1, 1)
        output, ldj = super().forward(x, context)
        return output.view(-1, self.D), ldj

    def reverse(self, z, context=None):
        z = z.view(-1, self.D, 1, 1)
        output = super().reverse(z, context)
        return output.view(-1, self.D)

    def logdet(self, x, context=None):
        x = x.view(-1, self.D, 1, 1)
        return super().logdet(x)
