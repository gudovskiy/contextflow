import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributions as TD
from einops import rearrange, repeat, reduce
import math


class StandardNormal(nn.Module):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, size, mixtures=1, context_net=None, contextflow=False):
        super(StandardNormal, self).__init__()
        assert mixtures == 1, 'mixtures should be 1 in GaussianDistribution'
        self.size = torch.Size(size)
        self.softp = nn.Softplus()
        self.M = M = mixtures
        self.K = K = 8
        self.register_buffer('buffer', torch.zeros(1))
        self.softp = nn.Softplus()
        self.context_net = context_net
        self.contextflow = contextflow
        #print(contextflow, context_net)
        if self.context_net:
            D, H, W = size
            #print(self.context_net.contexts[0])
            self.C = C = self.context_net.contexts[0]
            self.mC = nn.Parameter(torch.randn(C, 1, D, H, W))
            self.sC = nn.Parameter(torch.ones( C, 1, D, H, W))
            self.wC = nn.Parameter(torch.randn(C, 1,        ))
            print(self.mC.shape)

    def param_cond_dist(self, context):
        #c, logp_c = self.context_net(context)
        #print('c:', c.shape, c)
        #c = context[:,0]
        #c = rearrange(c, 'b (m p c) -> p b m c 1 1', m=self.M, p=2, c=DD)
        #logp_c = logp_c * H * W
        #mean, log_scale = c[0], c[1]
        #return TD.Independent(TD.Normal(mean, log_scale.exp()), 3), logp_c
        #print('context:', self.C, c.shape, torch.min(c), torch.max(c), c)
        mean, scale, weight = self.mC, self.softp(self.sC), torch.softmax(self.wC, dim=0)
        #print(mean.shape, scale.shape, weight.shape)
        return TD.MixtureSameFamily(TD.Categorical(weight), TD.Independent(TD.Normal(mean, scale), 3))

    def forward(self, input, context=None):
        return self.log_prob(input, context)

    def log_prob(self, input, context=None):
        log_base =  - 0.5 * math.log(2 * math.pi)
        log_inner = - 0.5 * input**2
        log_prob = (log_base + log_inner).flatten(start_dim=1).sum(-1).unsqueeze(-1)
        
        if self.context_net:
            #print(input.shape, context.shape, self.contextflow)
            #print('self.context_net:', self.context_net)
            x = repeat(input, 'b c h w -> b m c h w', m=self.C)
            #print('x:', x.shape)
            cond_dist = self.param_cond_dist(context)
            cond_prob = cond_dist.log_prob(x)  # BxC
            #print('cond_prob:', cond_prob.shape)
            c = context[:,0]
            log_prob = log_prob + cond_prob[:, c]  #+ logp_c.unsqueeze(-1)
        
        return log_prob

    def sample(self, n_samples, context=None):
        x = torch.randn(n_samples, *self.size, device=self.buffer.device, dtype=self.buffer.dtype)
        log_prob = self.log_prob(x, context)
        #print('x/log_prob:', x.shape, log_prob.shape)
        return x, log_prob


class GaussianDistribution(nn.Module):
    """
    Standard Normal Likelihood
    """
    def __init__(self, size, mixtures=1, context_net=None, contextflow=False):
        super().__init__()
        self.size = size
        self.softp = nn.Softplus()
        self.D = D = size[0]  # int(np.prod(size))
        assert mixtures == 1, 'mixtures should be 1 in GaussianDistribution'
        self.M = M = mixtures
        self.m = nn.Parameter(torch.zeros(D,1,1), requires_grad=False)
        self.s = nn.Parameter(torch.ones( D,1,1), requires_grad=False)
        self.context_net = context_net
        self.contextflow = contextflow
    
    def param_dist(self):
        mean, scale = self.m, self.softp(self.s)
        #print(mean.shape, scale.shape)
        dist = TD.Normal(mean, scale)
        return dist
    
    def forward(self, input, context=None):
        return self.log_prob(input, context)

    def log_prob(self, input, context=None, sum=True):
        #print(input.shape)
        dist = self.param_dist()
        #log_prob = dist.log_prob(input.view(-1, self.D)).sum(1)
        log_prob = torch.sum(dist.log_prob(input), (1,2,3))
        #print('log_prob:', log_prob.shape)
        return log_prob

    def sample(self, n_samples, context=None):
        dist = self.param_dist()
        #print('samples:', n_samples, dist)
        x = dist.sample((n_samples,)).view(n_samples, *self.size)
        #print('x:', x.shape)
        log_prob = self.log_prob(x, context)
        #print('log_prob:', log_prob.shape)
        return x, log_prob


class GaussianMixtureDistribution(nn.Module):
    """
    Standard Normal Likelihood
    """
    def __init__(self, size, mixtures=2, components=8, context_net=None, contextflow=False):
        super().__init__()
        self.size = size
        self.softp = nn.Softplus()
        D, H, W = size
        self.D = D
        self.M = M = mixtures
        self.K = K = components
        self.mG = nn.Parameter(torch.randn(M, K, D, H, W))
        self.sG = nn.Parameter(torch.ones( M, K, D, H, W))
        self.wG = nn.Parameter(torch.randn(M, K,        ))
        self.context_net = context_net
        self.contextflow = contextflow
        if self.context_net:
            self.C = C = self.context_net.C
            if self.contextflow:
                self.mG.requires_grad_(False)
                self.sG.requires_grad_(False)
                self.wG.requires_grad_(False)

    def param_dist(self, cond_mean=0.0, cond_scale=0.0):
        mean, scale, weight = self.mG + cond_mean, self.softp(self.sG+cond_scale), torch.softmax(self.wG, dim=-1)
        return TD.MixtureSameFamily(TD.Categorical(weight), TD.Independent(TD.Normal(mean, scale), 3))

    def log_prob(self, input, context=None):
        B, _, H, W = input.shape
        x = repeat(input, 'b d h w -> b m d h w', m=self.M)
        if isinstance(context, list): context = context[0]
        if self.context_net:
            c, logp_c = self.context_net(context)
            c = rearrange(c, 'b (p m k d) -> p b m k d 1 1', p=2, m=self.M, k=self.K, d=self.D)  # Bx2*M*K*D -> 2xBxMxKxDx1x1
            logp_c = logp_c * H * W
            dist = self.param_dist(cond_mean=c[0], cond_scale=c[1])
            log_prob = dist.log_prob(x)
            log_prob = log_prob + logp_c.unsqueeze(-1)
        else:
            dist = self.param_dist()
            log_prob = dist.log_prob(x)
        
        return log_prob

    def sample(self, n_samples, context=None):
        dist = self.param_dist()
        x = dist.sample((n_samples,)) #.view(n_samples, self.M, *self.size)
        #print('x:', x.shape)
        x = x[:,1,...]
        log_prob = self.log_prob(x, context)
        return x, log_prob


class MultivariateGaussianMixtureDistribution(nn.Module):
    """
    Standard Normal Likelihood
    """
    def __init__(self, size, mixtures=2, context_net=None, contextflow=False):
        super().__init__()
        self.size = size
        self.softp = nn.Softplus()
        self.D = D = int(np.prod(size))
        self.M = M = mixtures
        self.K = K = 8
        self.mG = nn.Parameter(torch.randn(M, K, D   ))
        self.sG = nn.Parameter(torch.ones( M, K, D   ))
        self.lG = nn.Parameter(torch.zeros(M, K, D, D))
        self.wG = nn.Parameter(torch.randn(M, K      ))
        self.context_net = context_net
        self.contextflow = contextflow
        if self.context_net and self.contextflow:
            self.C = C = D
            self.mG.requires_grad_(False)
            self.sG.requires_grad_(False)
            self.lG.requires_grad_(False)
            self.wG.requires_grad_(False)

    def param_dist(self):
        #mean, scale, weight = self.m, self.softp(self.s), torch.softmax(self.w, dim=0)
        mean, weight = self.mG, torch.softmax(self.wG, dim=0)
        scale = torch.tril(self.lG, diagonal=-1)
        scale = torch.diagonal_scatter(scale, self.softp(self.sG), 0, dim1=2, dim2=3)
        mix  = D.Categorical(weight)
        comp = D.MultivariateNormal(mean, scale_tril=scale)
        dist = D.MixtureSameFamily(mix, comp)
        return dist

    def cond_dist(self, context):
        c = self.context_net(context)
        c = rearrange(c, 'b (m p c) h w -> p b m c h w', m=self.M, p=2, c=self.C) #.contiguous()
        mean, scale = c[0], self.softp(c[1])
        #return D.MixtureSameFamily(D.Categorical(weight), D.Independent(D.Normal(mean, scale), 3))
        return D.Independent(D.Normal(mean, scale), 3)

    def log_prob(self, input, context=None):
        x = repeat(input, 'b c h w -> b m (c h w)', m=self.M)
        dist = self.param_dist()
        log_prob = dist.log_prob(x)
        if self.context_net and self.contextflow:
            cond_dist = self.cond_dist(context)
            cond_prob = cond_dist.log_prob(x)
            log_prob = log_prob + cond_prob
        
        return log_prob

    def sample(self, n_samples, context=None):
        dist = self.param_dist()
        #print('sampling from MultivariateGaussianMixtureDistribution:', self.w)
        x = dist.sample((n_samples,)).view(n_samples, self.M, *self.size)  #.sum(1)
        #print('x:', x.shape)
        x = x[:,0,...]
        log_px = self.log_prob(x, context)
        return x, log_px


class ConditionalGaussianDistribution(nn.Module):
    """
    Conditional Normal Likelihood
    """
    def __init__(self, size, mixtures=1, context_net=None, contextflow=False):
        super().__init__()
        self.size = size
        self.D = size[0]
        assert mixtures == 1, 'mixtures should be 1 in GaussianDistribution'
        self.M = M = mixtures
        self.context_net = context_net
        self.contextflow = contextflow

    def forward(self, x, context=None):
        return self.log_prob(x, context)

    def cond_dist(self, context):
        c, _ = self.context_net(context)
        c = rearrange(c, 'b (p c) -> p b c', p=2)
        mean, log_scale = c[0], c[1]
        return mean, log_scale

    def log_prob(self, x, context=None):
        mean, log_scale = self.cond_dist(context)
        log_base =  - 0.5 * math.log(2 * math.pi) - log_scale
        log_inner = - 0.5 * torch.exp(-2 * log_scale) * ((x - mean) ** 2)
        log_prob = (log_base+log_inner).sum(-1)
        return log_prob

    def sample(self, n_samples, context=None):
        mean, log_scale = self.cond_dist(context)
        eps = torch.randn(n_samples, *self.size, device=mean.device, dtype=mean.dtype)
        x = mean + log_scale.exp() * eps
        log_base =  - 0.5 * math.log(2 * math.pi) - log_scale
        log_inner = - 0.5 * torch.exp(-2 * log_scale) * ((x - mean) ** 2)
        log_prob = (log_base+log_inner).sum(-1)
        return x, log_prob
