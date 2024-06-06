import torch
from .flowlayer import FlowLayer
from collections.abc import Iterable

class PermuteAxes(FlowLayer):
    def __init__(self, permutation):
        super().__init__()
        assert isinstance(permutation, Iterable), 'permutation must be an Iterable'
        assert permutation[0] == 0, 'First element of permutation must be 0 (such that batch dimension stays intact)'
        self.permutation = permutation
        self.inverse_permutation = torch.argsort(torch.tensor(self.permutation)).tolist()

    def forward(self, input, context=None):
        return input.permute(self.permutation).contiguous(), self.logdet(input, context)

    def reverse(self, input, context=None):
        return input.permute(self.inverse_permutation).contiguous()

    def logdet(self, input, context=None):
        return input.new_zeros(len(input))


class Permute(FlowLayer):
    """
    Permutes inputs on a given dimension using a given permutation.

    Args:
        permutation: Tensor or Iterable, shape (dim_size)
        dim: int, dimension to permute (excluding batch_dimension)
    """

    def __init__(self, permutation, dim=1):
        super(Permute, self).__init__()
        assert isinstance(dim, int), 'dim must be an integer'
        assert dim >= 1, 'dim must be >= 1 (0 corresponds to batch dimension)'
        assert isinstance(permutation, torch.Tensor) or isinstance(permutation, Iterable), 'permutation must be a torch.Tensor or Iterable'
        if isinstance(permutation, torch.Tensor):
            assert permutation.ndimension() == 1, 'permutation must be a 1D tensor, but was of shape {}'.format(permutation.shape)
        else:
            permutation = torch.tensor(permutation)

        self.dim = dim
        self.register_buffer('permutation', permutation)

    @property
    def inverse_permutation(self):
        return torch.argsort(self.permutation)

    def forward(self, x, context=None):
        return torch.index_select(x, self.dim, self.permutation), torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    def reverse(self, z, context=None):
        return torch.index_select(z, self.dim, self.inverse_permutation), torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

    def logdet(self, x, context=None):
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)


class Shuffle(Permute):
    """
    Permutes inputs on a given dimension using a random, but fixed, permutation.

    Args:
        dim_size: int, number of elements on dimension dim
        dim: int, dimension to permute (excluding batch_dimension)
    """

    def __init__(self, dim_size, dim=1):
        super(Shuffle, self).__init__(torch.randperm(dim_size), dim)


class Reverse(Permute):
    """
    Reverses inputs on a given dimension.

    Args:
        dim_size: int, number of elements on dimension dim
        dim: int, dimension to permute (excluding batch_dimension)
    """

    def __init__(self, dim_size, dim=1):
        super(Reverse, self).__init__(torch.arange(dim_size - 1, -1, -1), dim)


class Exchange(Permute):
    """
    Exchanges inputs on a given dimension.

    Args:
        dim_size: int, number of elements on dimension dim
        dim: int, dimension to permute (excluding batch_dimension)
    """

    def __init__(self, dim_size, dim=1):
        super(Exchange, self).__init__(torch.cat((torch.arange(dim_size//2, dim_size), torch.arange(0, dim_size//2))), dim)


class EvenOdd(Permute):
    """
    Exchanges inputs on a given dimension.

    Args:
        dim_size: int, number of elements on dimension dim
        dim: int, dimension to permute (excluding batch_dimension)
    """

    def __init__(self, dim_size, dim=1):
        super(EvenOdd, self).__init__(torch.cat((torch.arange(0, dim_size, step=2), torch.arange(1, dim_size, step=2))), dim)

