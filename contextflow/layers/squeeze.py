import torch
from .flowlayer import FlowLayer
from einops import rearrange

class Squeeze(FlowLayer):
    def __init__(self, patch_size=(2, 2)):
        super().__init__()
        self.p = patch_size

    def forward(self, input, context=None):
        return rearrange(input, 'b c (h p1) (w p2) -> b (c p1 p2) h w', p1=self.p[0], p2=self.p[1]), self.logdet(input, context)

    def reverse(self, input, context=None):
        return rearrange(input, 'b (c p1 p2) h w -> b c (h p1) (w p2)', p1=self.p[0], p2=self.p[1])

    def logdet(self, input, context=None):
        return input.new_zeros(len(input))


class UnSqueeze(FlowLayer):
    def __init__(self, patch_size=(2, 2)):
        super().__init__()
        self.p = patch_size

    def forward(self, input, context=None):
        return rearrange(input, 'b (c p1 p2) h w -> b c (h p1) (w p2)', p1=self.p[0], p2=self.p[1]), self.logdet(input, context)

    def reverse(self, input, context=None):
        return rearrange(input, 'b c (h p1) (w p2) -> b (c p1 p2) h w', p1=self.p[0], p2=self.p[1])

    def logdet(self, input, context=None):
        return input.new_zeros(len(input))
