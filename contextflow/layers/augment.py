import torch
from einops import rearrange
from .flowlayer import FlowLayer


class Augment(FlowLayer):
    def __init__(self, aug_distribution, aug_size, split_dim=1):
        super().__init__()
        # deq_distribution should be a distribution with support on [0, 1]^d
        self.distribution = aug_distribution
        self.aug_size = aug_size
        self.split_dim = split_dim

    def forward(self, input, context=None):
        # Note, input is the context for distribution model.
        noise, log_qnoise = self.distribution.sample(input.size(0))
        input = torch.cat([input, noise], dim=self.split_dim)
        return input, -log_qnoise

    def reverse(self, input, context=None):
        split_proportions = (input.shape[self.split_dim] - self.aug_size, self.aug_size)
        x, _ = torch.split(input, split_proportions, dim=self.split_dim)
        return x

    def logdet(self, input, context=None):
        raise NotImplementedError
