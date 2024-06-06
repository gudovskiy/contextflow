import torch
import torch.nn as nn
from .activations import FlowActivationLayer
from einops import rearrange, repeat


class ActNorm(FlowActivationLayer):
    def __init__(self, data_size, context_net=None, contextflow=False):
        super().__init__()
        D, H, W = data_size if len(data_size)==3 else (data_size[0], 1, 1)
        self.D = D
        self.H = H
        self.W = W
        self.NN_t    = nn.Parameter(torch.zeros(D))
        self.NN_logs = nn.Parameter(torch.zeros(D))
        self.register_buffer('initialized', torch.tensor(0))
        self.context_net = context_net
        self.contextflow = contextflow
        if self.context_net:
            self.C = C = self.context_net.C
            self.CN = nn.Linear(C, 2*D)
            if self.contextflow:
                self.NN_t.requires_grad_(False)
                self.NN_logs.requires_grad_(False)
                nn.init.zeros_(self.CN.weight)
                nn.init.zeros_(self.CN.bias)
    
    def initialize(self, x, t, logs, initialized):
        reduce_dims = [i for i in range(len(x.size())) if i != 1]
        with torch.no_grad():
            mean    = torch.mean(x, dim=reduce_dims)
            logstd = torch.log(torch.std(x, dim=reduce_dims) + 1e-8)
            t.data.copy_(mean)
            logs.data.copy_(logstd)
            initialized.fill_(1)
    
    def forward(self, x, context=None):
        B, _, H, W = x.shape
        
        if self.context_net:
            c, logp_c = self.context_net(context)
            c = rearrange(self.CN(c), 'b (p d) -> p b d 1 1', p=2)  # Bx2*C -> 2xBxDx1x1
            logp_c = logp_c * H * W

            if self.contextflow:
                if not self.initialized: self.initialize(x, self.NN_t, self.NN_logs, self.initialized)
                t    = c[0] + repeat(self.NN_t,    'c -> b c () ()', b=B)
                logs = c[1] + repeat(self.NN_logs, 'c -> b c () ()', b=B)
            else:
                t    = c[0]
                logs = c[1]
        else:
            if not self.initialized: self.initialize(x, self.NN_t, self.NN_logs, self.initialized)
            t    = repeat(self.NN_t,    'c -> b c () ()', b=B)
            logs = repeat(self.NN_logs, 'c -> b c () ()', b=B)
            logp_c = torch.zeros(B).to(device=x.device)
        
        ldj = logs.flatten(start_dim=1).sum(-1)
        z = (x - t) * torch.exp(-logs)
        return z, ldj + logp_c
    # to update:
    def reverse(self, z, context=None):
        B, _, H, W = z.shape
        if self.context_net:
            c = rearrange(self.context_net(context), 'b (p c) h w -> p b c h w', p=2)
            if self.contextflow:
                assert self.initialized
                t    = c[0] + repeat(self.NN_t,    'c -> b c () ()', b=B)
                logs = c[1] + repeat(self.NN_logs, 'c -> b c () ()', b=B)
            else:
                t    = c[0]
                logs = c[1]
        else:
            assert self.initialized
            t    = repeat(self.NN_t,    'c -> b c () ()', b=B)
            logs = repeat(self.NN_logs, 'c -> b c () ()', b=B)
        
        x = z * torch.exp(logs) + t
        return x

    def logdet(self, x, context=None):
        _, ldj = self.forward(x, context)
        return ldj


class ActNormFC(ActNorm):
    def __init__(self, data_size):
        super().__init__(data_size)

    def forward(self, x, context=None):
        x = x.view(-1, self.D, 1, 1)
        output, ldj = super().forward(x, context)
        return output.view(-1, self.D), ldj

    def reverse(self, x, context=None):
        x = x.view(-1, self.D, 1, 1)
        output = super().reverse(x, context)
        return output.view(-1, self.D)

    def logdet(self, x, context=None):
        x = x.view(-1, self.D, 1, 1)
        return super().logdet(x)