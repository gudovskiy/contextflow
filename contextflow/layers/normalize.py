import torch
from torch import Tensor
import numpy as np
from .flowlayer import PreprocessingFlowLayer

class Normalization(PreprocessingFlowLayer):
    def __init__(self, translation, scale, learnable=False):
        super().__init__()
        if   isinstance(translation, Tensor): translation = translation
        elif isinstance(translation, list) or isinstance(translation, tuple): translation = torch.Tensor(translation)
        else: translation = torch.Tensor([translation])
        
        if   isinstance(scale, Tensor): scale = scale
        elif isinstance(scale, list) or isinstance(scale, tuple): scale = torch.Tensor(scale)
        else: scale = torch.Tensor([scale])

        #translation = translation.view(-1,1,1)
        #scale = scale.view(-1,1,1)
        
        if learnable:
            self.translation = torch.nn.Parameter(translation)
            self.scale = torch.nn.Parameter(scale)
        else:
            self.register_buffer('translation', translation)
            self.register_buffer('scale', scale)

    def forward(self, input, context=None):
        fill = (1,) * (input.dim()-2)
        scale = self.scale.view(self.scale.shape[0], *fill)
        translation = self.translation.view(self.translation.shape[0], *fill)
        #print('s/t:', self.scale.shape, self.translation.shape, input.shape)
        output = (input / scale) + translation
        #print('output:', output.shape, torch.min(output), torch.max(output))
        return output, self.logdet(input, context)

    def reverse(self, input, context=None):
        fill = (1,) * (input.dim()-2)
        scale = self.scale.view(self.scale.shape[0], *fill)
        translation = self.translation.view(self.translation.shape[0], *fill)
        return (input - translation) * scale

    def logdet(self, input, context=None):
        B, C = input.shape[:2]
        D = input.numel()/B/C
        logdet = -1 * D * torch.log(self.scale).sum()
        ldj = logdet if self.scale.numel() > 1 else C*logdet
        #if self.scale.numel() > 1: logdet = -1 * D * torch.log(self.scale).sum()
        #else:                      logdet = -C * D * torch.log(self.scale).sum()
        return ldj.expand(B)


'''class Normalization(PreprocessingFlowLayer):
    def __init__(self, translation, scale, learnable=False):
        super().__init__()
        if   isinstance(translation, Tensor): translation = translation
        elif isinstance(translation, list) or isinstance(translation, tuple):   translation = torch.Tensor(translation)
        else: translation = torch.Tensor([translation])
        
        if   isinstance(scale, Tensor): scale = scale
        elif isinstance(scale, list) or isinstance(scale, tuple):   scale = torch.Tensor(scale)
        else: scale = torch.Tensor([scale])

        translation = translation.view(-1,1,1)
        scale = scale.view(-1,1,1)
        
        if learnable:
            self.translation = torch.nn.Parameter(translation)
            self.scale = torch.nn.Parameter(scale)
        else:
            self.register_buffer('translation', translation)
            self.register_buffer('scale', scale)

    def forward(self, input, context=None):
        #output = (input - self.translation) / self.scale
        output = (input / self.scale) + self.translation
        #print('output:', output.shape, torch.min(output), torch.max(output))
        return output, self.logdet(input, context)

    def reverse(self, input, context=None):
        return (input - self.translation) * self.scale
        #return (input * self.scale) + self.translation

    def logdet(self, input, context=None):
        B, C, H, W = input.size()
        if self.scale.numel() > 1:
            logdet = -1 * H * W * torch.log(self.scale).sum()
        else:
            logdet = -C * H * W * torch.log(self.scale).sum()
        
        return logdet.expand(B)'''