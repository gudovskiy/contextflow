import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from einops import rearrange, repeat
from .flowlayer import FlowLayer
from .simple_vit import SimpleViT, posemb_sincos_2d
from utils.helpers import freeze_parameters
from datasets.ts import InputDropout

kwargs_act = {'softplus': nn.Softplus(), 'relu': nn.ReLU(), 'elu': nn.ELU(), 'gelu': nn.GELU(), 'tanh': nn.Tanh(), 'selu': nn.SELU(), 'sigmoid': nn.Sigmoid()}


class Coupling(FlowLayer):
    def __init__(self, data_channels, kernel_size=(1,1), padding=(0,0), context_net=None, contextflow=False):
        super().__init__()
        D = data_channels//2
        H = data_channels*2
        O = data_channels
        ACT = kwargs_act['relu']  # relu
        K = kernel_size
        P = padding
        self.context_net = context_net
        self.contextflow = contextflow
        #self.inpdp = InputDropout(0.1)
        C1 = nn.Conv2d(D, H, 1)
        C2 = nn.Conv2d(H, H, K, padding=P, padding_mode='reflect')
        C3 = nn.Conv2d(H, O, 1)
        self.NN = nn.Sequential(C1, ACT, C2, ACT, C3)
        if self.context_net:
            if not self.contextflow:
                C1 = nn.Conv2d(D+O, H, 1)  # concat
                self.NN = nn.Sequential(C1, ACT, C2, ACT, C3)
            else: freeze_parameters(self.NN)
            
            self.C = C = self.context_net.C
            self.CN = nn.Sequential(nn.Linear(C, H), ACT, nn.Linear(H, H), ACT, nn.Linear(H, O))
    
    def get_xs_logs_t(self, x1, context=None):
        B, _, H, W = x1.shape
        if self.context_net:
            c, logp_c = self.context_net(context)
            logp_c = logp_c * H * W
            if self.contextflow:
                h = self.NN(x1) + rearrange(self.CN(c), 'b c -> b c 1 1')
            else:
                h = self.NN(torch.concat((x1, repeat(self.CN(c), 'b c -> b c h w', h=H, w=W)), dim=1))
        else:
            h = self.NN(x1)
            logp_c = torch.zeros(B).to(device=x1.device)

        h = rearrange(h, 'b (p c) h w -> p b c h w', p=2)
        t = h[0]

        logs_range = 2.0
        log_s = logs_range * torch.tanh(h[1] / logs_range)
        s = torch.exp(log_s)
        return s, t, log_s, logp_c

    def forward(self, x, context=None):
        #x = self.inpdp(x)
        x = rearrange(x, 'b (p c) h w -> p b c h w', p=2)
        s, t, log_s, logp_c = self.get_xs_logs_t(x[0], context)
        z1 = x[1] * s + t
        z = rearrange([x[0], z1], 'p b c h w -> b (p c) h w', p=2)
        return z, log_s.flatten(start_dim=1).sum(-1) + logp_c
    # to update:
    def reverse(self, z, context=None):
        z = rearrange(z, 'b (p c) h w -> p b c h w', p=2)
        s, t, log_s, _ = self.get_xs_logs_t(z[0], context)
        x1 = (z[1] - t) / s
        x = rearrange([z[0], x1], 'p b c h w -> b (p c) h w', p=2)
        return x

    def logdet(self, input, context=None):
        _, ldj = self.forward(input, context)
        return ldj


class CouplingFC(Coupling):
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


class TransCoupling(FlowLayer):
    def __init__(self, in_sz, p_sz, context_net=None, contextflow=False):
        super().__init__()
        D = in_sz[0]//2
        H = in_sz[0]*2
        O = in_sz[0]
        tr_size = (in_sz[1], in_sz[2])
        P = p_sz[0]*p_sz[1]
        T = O*P  # 128  # hidden dim of transformer
        heads, layers = 1, 6
        ACT = kwargs_act['relu']  # relu
        self.context_net = context_net
        self.contextflow = contextflow
        #self.inpdp = InputDropout(0.1)
        self.NN = nn.Sequential(SimpleViT(image_size=tr_size, patch_size=p_sz, dim=T, depth=layers, heads=heads, mlp_dim=T, channels=D))
        if self.context_net:
            if not self.contextflow:
                self.NN = SimpleViT(image_size=tr_size, patch_size=p_sz, dim=T, depth=layers, heads=heads, mlp_dim=T, channels=D+O)
            else: freeze_parameters(self.NN)

            self.C = C = self.context_net.C
            self.CN = nn.Sequential(nn.Linear(C, H), ACT, nn.Linear(H, H), ACT, nn.Linear(H, O))

    def get_xs_logs_t(self, x1, context=None):
        B, _, H, W = x1.shape
        if self.context_net:
            c, logp_c = self.context_net(context)
            if self.contextflow:
                h = self.NN(x1) + rearrange(self.CN(c), 'b c -> b c 1 1')
            else:
                h = self.NN(torch.concat((x1, repeat(self.CN(c), 'b c -> b c h w', h=H, w=W)), dim=1))
        else:
            h = self.NN(x1)
            logp_c = torch.zeros(B).to(device=x1.device)

        h = rearrange(h, 'b (p c) h w -> p b c h w', p=2)
        t = h[0]
        logs_range = 2.0
        log_s = logs_range * torch.tanh(h[1] / logs_range)
        s = torch.exp(log_s)
        return s, t, log_s, logp_c

    def forward(self, x, context=None):
        #x = self.inpdp(x)
        x = rearrange(x, 'b (p c) h w -> p b c h w', p=2)
        s, t, log_s, logp_c = self.get_xs_logs_t(x[0], context)
        z1 = x[1] * s + t
        z = rearrange([x[0], z1], 'p b c h w -> b (p c) h w', p=2)
        return z, log_s.flatten(start_dim=1).sum(-1) + logp_c
    # to update:
    def reverse(self, z, context=None):
        z = rearrange(z, 'b (p c) h w -> p b c h w', p=2)
        s, t, log_s, _ = self.get_xs_logs_t(z[0], context)
        x1 = (z[1] - t) / s
        x = rearrange([z[0], x1], 'p b c h w -> b (p c) h w', p=2)
        return x

    def logdet(self, input, context=None):
        _, ldj = self.forward(input, context)
        return ldj


'''class Conv2dZero(nn.Module):
    """
    From https://github.com/ehoogeboom/emerging/blob/9545da2f87d5507a506b68e6f4a261086a4e2c47/tfops.py#L294
    """
    def __init__(self, data_channels, out_channels, bias=True, 
                 kernel_size=(3,3), stride=(1,1), padding=(1,1),
                 dilation=1, groups=1, logscale_factor=3):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.data_channels = data_channels
        self.out_channels = out_channels
        self.logscale_factor = logscale_factor

        w_shape = (out_channels, data_channels, *self.kernel_size)
        w_init = nn.init.zeros_(torch.empty(w_shape))

        self.weight = nn.Parameter(w_init)

        zeros = torch.nn.init.zeros_(torch.empty(out_channels))
        self.bias = nn.Parameter(zeros) if bias else None

        # for ReZero Trick
        self.logs = nn.Parameter(zeros)

    def forward(self, input):
        output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        output = output * torch.exp(self.logs * self.logscale_factor).view(1, -1, 1, 1)
        return output'''
