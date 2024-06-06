import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributions as D
from einops import rearrange, repeat, reduce
import math


class StudentDistribution(nn.Module):
    def __init__(self, size, mixtures=1):
        super().__init__()
        self.size = size
        self.softp = nn.Softplus()
        self.D = D = int(np.prod(size))
        self.M = M = mixtures
        self.m = nn.Parameter(torch.zeros( D), requires_grad=False)
        self.s = nn.Parameter(torch.ones(  D), requires_grad=False)
        self.v = nn.Parameter(2*torch.ones(D), requires_grad=False)
    
    def param_dist(self):
        mean, scale = self.m, self.softp(self.s)
        dist = D.Normal(mean, scale)
        return dist
    
    def forward(self, input, context=None):
        return self.log_prob(input, context)

    def log_prob(self, input, context=None, sum=True):
        dist = self.param_dist()
        log_prob = dist.log_prob(input.view(-1, self.D)).sum(1)
        return log_prob

    def sample(self, n_samples, context=None):
        dist = self.param_dist()
        #print('samples:', n_samples, dist)
        x = dist.sample((n_samples,)).view(n_samples, *self.size)
        #print('x:', x.shape)
        log_prob = self.log_prob(x, context)
        #print('log_prob:', log_prob.shape)
        return x, log_prob


class StudentMixtureDistribution(nn.Module):
    def __init__(self, size, mixtures=2, context_net=None, contextflow=False):
        super().__init__()
        self.size = size
        self.softp = nn.Softplus()
        D, H, W  = size
        self.M = M = mixtures
        self.K = K = 8
        self.mG = nn.Parameter(torch.randn (M, K, D, H, W))
        self.sG = nn.Parameter(torch.ones(  M, K, D, H, W))
        self.wG = nn.Parameter(torch.randn (M, K,        ))
        self.mS = nn.Parameter(torch.randn (M, K, D, H, W))
        self.sS = nn.Parameter(torch.ones(  M, K, D, H, W))
        self.wS = nn.Parameter(torch.randn (M, K,        ))
        v_init = torch.linspace(1, 10, K).view(1, K, 1, 1, 1)
        #v_init = torch.logspace(0, 10, K, base=2).view(1, K, 1, 1, 1)
        self.vS = nn.Parameter(v_init.repeat(M, 1, D, H, W))
        self.vS.requires_grad_(False)
        self.context_net = None  # context_net
        self.contextflow = False
        if self.context_net:
            self.C = C = self.context_net.C
            #if self.contextflow:
            #    self.mG.requires_grad_(False)
            #    self.sG.requires_grad_(False)
            #    self.wG.requires_grad_(False)

        #if self.context_net and self.contextflow:
        #    self.C = C = D
        #    self.mG.requires_grad_(False)
        #    self.sG.requires_grad_(False)
        #    self.wG.requires_grad_(False)

    def param_dist(self):
        meanG, scaleG, weightG        = self.mG, self.softp(self.sG), torch.softmax(self.wG, dim=0)
        meanS, scaleS, weightS, tailS = self.mS, self.softp(self.sS), torch.softmax(self.wS, dim=0), self.softp(self.vS)
        return D.MixtureSameFamily(D.Categorical(weightG), D.Independent(D.Normal(         meanG, scaleG), 3)), \
               D.MixtureSameFamily(D.Categorical(weightS), D.Independent(D.StudentT(tailS, meanS, scaleS), 3))

    def cond_dist(self, context):
        DD, H, W  = self.size
        c, logp_c = self.context_net(context)
        #print(context.shape, c.shape)
        c = rearrange(c, 'b (m p c) -> p b m c 1 1', m=self.M, p=2, c=DD)
        logp_c = logp_c * H * W
        mean, log_scale = c[0], c[1]
        return D.Independent(D.Normal(mean, log_scale.exp()), 3), logp_c

    def log_prob(self, input, context=None):
        x = repeat(input, 'b c h w -> b m c h w', m=self.M)
        distG, distS = self.param_dist()
        log_prob = distG.log_prob(x) + distS.log_prob(x)
        #log_prob = distS.log_prob(x)
        #print(self.context_net, self.contextflow)
        if self.context_net:
            print('self.context_net:', self.context_net)
            cond_dist, logp_c = self.cond_dist(context)
            cond_prob = cond_dist.log_prob(x)
            log_prob = log_prob + cond_prob  #+ logp_c.unsqueeze(-1)
        
        return log_prob

    def sample(self, n_samples, context=None):
        distG, distS = self.param_dist()
        x = distG.sample((n_samples,)) #.view(n_samples, self.M, *self.size)
        #print('x:', x.shape)
        x = x[:,1,...]
        log_prob = self.log_prob(x, context)
        return x, log_prob