import torch
import torch.nn as nn


class FlowSequential(nn.Module):
    def __init__(self, dist, *modules):
        super().__init__()
        self.dist = dist
        self.mixtures = dist.M

        for i, module in enumerate(modules):
            self.add_module(str(i), module)
        self.sequence_modules = modules

    def __iter__(self):
        yield from self.sequence_modules

    def forward(self, input, context=None):
        M, B, device = self.mixtures, input.shape[0], input.device
        logdet = torch.zeros((B, M), device=device)
        for module in self.sequence_modules:
            output_input, layer_logdet = module(input, context)
            logdet += layer_logdet if logdet.dim() == layer_logdet.dim() else layer_logdet.unsqueeze(-1)
            input   = output_input

        logprob = self.dist.log_prob(input, context)
        return output_input, logprob + logdet

    def log_prob(self, input, context=None):
        return self.forward(input, context)[1]

    def sample(self, n_samples, context=None):
        z, _ = self.dist.sample(n_samples, context)
        input = z
        for module in reversed(self.sequence_modules):
            output = module.reverse(input, context)
            input = output

        return input


class FlowInvSequential(nn.Module):
    def __init__(self, dist, *modules):
        super().__init__()
        self.dist = dist

        for i, module in enumerate(modules):
            self.add_module(str(i), module)
        self.sequence_modules = modules

    def __iter__(self):
        yield from self.sequence_modules

    def forward(self, input, context=None):
        return self.sample(input, context)

    def log_prob(self, input, context=None):
        raise RuntimeError("InverseFlow does not support log_prob, see Flow instead.")

    def sample(self, input, context=None):
        output, logprob = self.dist.sample(input.size(0), context)
        input = output
        # regular sample
        for module in self.sequence_modules:
            output, layer_logdet = module(input, context)
            logprob -= layer_logdet
            input = output

        return output, logprob