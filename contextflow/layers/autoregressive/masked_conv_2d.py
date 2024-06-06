import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import mask_conv2d_spatial, mask_conv2d


class _MaskedConv2d(nn.Conv2d):
    """
    A masked version of nn.Conv2d.
    """

    def register_mask(self, mask):
        """
        Registers mask to be used in forward pass.

        Input:
            mask: torch.FloatTensor
                Shape needs to be broadcastable with self.weight.
        """
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super(_MaskedConv2d, self).forward(x)


class SpatialMaskedConv2d(_MaskedConv2d):
    """
    A version of nn.Conv2d masked to be autoregressive in the spatial dimensions.
    Uses mask of shape (1, 1, height, width).

    Input:
        *args: Arguments passed to the constructor of nn.Conv2d.
        mask_type: str
            Either 'A' or 'B'. 'A' for first layer of network, 'B' for all others.
        **kwargs: Keyword arguments passed to the constructor of nn.Conv2d.
    """

    def __init__(self, *args, mask_type, **kwargs):
        super(SpatialMaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        _, _, height, width = self.weight.size()
        mask = mask_conv2d_spatial(mask_type, height, width)
        self.register_mask(mask)


class MaskedConv2d(_MaskedConv2d):
    """
    A version of nn.Conv2d masked to be autoregressive in
    the spatial dimensions and in the channel dimension.
    This is constructed specifically for data that
    has any number of input channels.
    Uses mask of shape (out_channels, in_channels, height, width).

    Input:
        *args: Arguments passed to the constructor of nn.Conv2d.
        mask_type: str
            Either 'A' or 'B'. 'A' for first layer of network, 'B' for all others.
        data_channels: int
            Number of channels in the input data, e.g. 3 for RGB images. Default: 3.
            This will be used to mask channels throughout the newtork such that
            all feature maps will have order (R, G, B, R, G, B, ...).
            In the case of mask_type B, for the central pixel:
            Outputs in position R can only access inputs in position R.
            Outputs in position G can access inputs in position R and G.
            Outputs in position B can access inputs in position R, G and B.
            In the case of mask_type A, for the central pixel:
            Outputs in position G can only access inputs in position R.
            Outputs in position B can access inputs in position R and G.
        **kwargs: Keyword arguments passed to the constructor of nn.Conv2d.
    """

    def __init__(self, *args, mask_type, data_channels=3, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        out_channels, in_channels, height, width = self.weight.size()
        mask = mask_conv2d(mask_type, in_channels, out_channels, height, width, data_channels)
        self.register_mask(mask)


class MaskedResidualBlock2d(nn.Module):
    def __init__(self, I, O, kernel_size=(1,1), padding=(0,0), D=0, mask_type='B'):
        super(MaskedResidualBlock2d, self).__init__()

        self.conv1 = MaskedConv2d(1*I, 2*I, 1, mask_type=mask_type, data_channels=D)
        self.conv2 = MaskedConv2d(2*I, 2*I, kernel_size, padding=padding, padding_mode='reflect', mask_type=mask_type, data_channels=D)
        self.conv3 = MaskedConv2d(2*I, 2*O, 1, mask_type=mask_type, data_channels=D)
        #p = 0.1
        #if p>0.: self.drop = nn.Dropout2d(p)
        #else:    self.drop = nn.Identity()
    
    def forward(self, x):
        identity = x.repeat(1,2,1,1)
        #x = self.drop(x)
        x = self.conv1(F.relu(x))
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        return x + identity


class SpatialMaskedResidualBlock2d(nn.Module):
    def __init__(self, h, kernel_size=3):
        super(SpatialMaskedResidualBlock2d, self).__init__()

        self.conv1 = nn.Conv2d(2 * h, h, kernel_size=1)
        self.conv2 = SpatialMaskedConv2d(h, h, kernel_size=kernel_size, padding=kernel_size//2, mask_type='B')
        self.conv3 = nn.Conv2d(h, 2 * h, kernel_size=1)
        #p = 0.1
        #if p>0.: self.drop = nn.Dropout2d(p)
        #else:    self.drop = nn.Identity()

    def forward(self, x):
        identity = x.repeat(1,2,1,1)
        #x = self.drop(x)
        x = self.conv1(F.relu(x))
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        
        return x + identity
