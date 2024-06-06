import torch
from einops import rearrange
from .flowlayer import FlowLayer
#from .coupling import Coupling


class SplitPrior(FlowLayer):
    def __init__(self, dist):
        super().__init__()
        self.dist = dist

    def forward(self, x, context=None):
        x = rearrange(x, 'b (p c) h w -> p b c h w', p=2)
        ldj = self.dist.log_prob(x[1], context)
        return x[0], ldj

    def reverse(self, z, context=None):
        z2, _ = self.dist.sample(self.C, context)
        x = rearrange([z, z2], 'p b c h w -> b (p c) h w', p=2)
        #x = self.transform.reverse(x, context)
        return x

    def logdet(self, input, context=None):
        _, ldj = self.forward(input, context)
        return ldj


# revise:
'''class SplitPriorFC(SplitPrior):
    def __init__(self, n_dims, distribution):
        assert type(n_dims) == int
        self.n_dims = n_dims
        self.half_dims = n_dims // 2
        input_size = (n_dims, 1, 1)
        super().__init__(input_size, distribution)

    def forward(self, input, context=None):
        input = input.view(-1, self.n_dims, 1, 1)
        output, ldj = super().forward(input, context)
        return output.view(-1, self.half_dims), ldj

    def reverse(self, input, context=None):
        input = input.view(-1, self.half_dims, 1, 1)
        output = super().reverse(input, context)
        return output.view(-1, self.n_dims)

    def logdet(self, input, context=None):
        input = input.view(-1, self.n_dims, 1, 1)
        return super().logdet(input)'''