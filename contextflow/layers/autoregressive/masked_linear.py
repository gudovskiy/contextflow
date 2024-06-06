import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def repeat_rows(x, num_reps):
    """Each row of tensor `x` is repeated `num_reps` times along leading dimension."""
    shape = x.shape
    x = x.unsqueeze(1)
    x = x.expand(shape[0], num_reps, *shape[1:])
    return merge_leading_dims(x, num_dims=2)

# Adapted from https://github.com/bayesiains/nsf/blob/master/nde/transforms/made.py
# and https://github.com/karpathy/pytorch-made/blob/master/made.py

class MaskedLinear(nn.Linear):
    """
    A linear module with a masked weight matrix.

    Args:
        in_degrees: torch.LongTensor, length matching number of input features.
        out_features: int, number of output features.
        data_features: int, number of features in the data.
        random_mask: bool, if True, a random connection mask will be sampled.
        random_seed: int, seed used for sampling random order/mask.
        is_output: bool, whether the layer is the final layer.
        data_degrees: torch.LongTensor, length matching number of data features (needed if is_output=True).
        bias: bool, if True a bias is included.
    """

    def __init__(self,
                 in_degrees,
                 out_features,
                 data_features,
                 random_mask=False,
                 random_seed=None,
                 is_output=False,
                 data_degrees=None,
                 bias=True):
        if is_output:
            assert data_degrees is not None
            assert len(data_degrees) == data_features
        
        super(MaskedLinear, self).__init__(in_features=len(in_degrees),
                                           out_features=out_features,
                                           bias=bias)
        self.out_features = out_features
        self.data_features = data_features
        self.is_output = is_output
        mask, out_degrees = self.get_mask_and_degrees(in_degrees=in_degrees,
                                                      data_degrees=data_degrees,
                                                      random_mask=random_mask,
                                                      random_seed=random_seed)
        self.register_buffer('mask', mask)
        self.register_buffer('degrees', out_degrees)

    @staticmethod
    def get_data_degrees(in_features, random_order=False, random_seed=None):
        if random_order:
            rng = np.random.RandomState(random_seed)
            return torch.from_numpy(rng.permutation(in_features) + 1)
        else:
            return torch.arange(1, in_features + 1)

    def get_mask_and_degrees(self,
                             in_degrees,
                             data_degrees,
                             random_mask,
                             random_seed):
        if self.is_output:
            out_degrees = repeat_rows(data_degrees, self.out_features // self.data_features)
            mask = (out_degrees[..., None] > in_degrees).float()

        else:
            if random_mask:
                min_in_degree = torch.min(in_degrees).item()
                min_in_degree = min(min_in_degree, self.data_features - 1)
                rng = np.random.RandomState(random_seed)
                out_degrees = torch.from_numpy(rng.randint(min_in_degree,
                                                           self.data_features,
                                                           size=[self.out_features]))
            else:
                max_ = max(1, self.data_features - 1)
                min_ = min(1, self.data_features - 1)
                out_degrees = torch.arange(self.out_features) % max_ + min_
            mask = (out_degrees[..., None] >= in_degrees).float()

        return mask, out_degrees

    def update_mask_and_degrees(self,
                                in_degrees,
                                data_degrees,
                                random_mask,
                                random_seed):
        mask, out_degrees = self.get_mask_and_degrees(in_degrees=in_degrees,
                                                      data_degrees=data_degrees,
                                                      random_mask=random_mask,
                                                      random_seed=random_seed)
        self.mask.data.copy_(mask)
        self.degrees.data.copy_(out_degrees)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MaskedResidualBlockLinear(nn.Module):
    def __init__(self, I, O, D):
        super(MaskedResidualBlockLinear, self).__init__()

        self.linear1 = MaskedLinear(MaskedLinear.get_data_degrees(1*I), 2*I, D)
        self.linear2 = MaskedLinear(MaskedLinear.get_data_degrees(2*I), 2*I, D)
        self.linear3 = MaskedLinear(MaskedLinear.get_data_degrees(2*I), 2*O, D)
        #print(self.linear1.weight.shape, self.linear1.mask.shape)
        #print(self.linear2.weight.shape, self.linear2.mask.shape)
        #print(self.linear3.weight.shape, self.linear3.mask.shape)
        #p = 0.1
        #if p>0.: self.drop = nn.Dropout2d(p)
        #else:    self.drop = nn.Identity()
    
    def forward(self, x):
        identity = x  #.repeat(1,2)
        #x = self.drop(x)
        #print(self.linear1.weight.shape, self.linear1.mask.shape, x.shape)
        x = self.linear1(F.relu(x))
        x = self.linear2(F.relu(x))
        x = self.linear3(F.relu(x))
        print('linear:', x.shape)
        return x + identity
