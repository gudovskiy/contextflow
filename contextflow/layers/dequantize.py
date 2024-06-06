import torch
from .flowlayer import PreprocessingFlowLayer
from einops import rearrange
from layers.activations import Sigmoid, Softplus  #, ScalarAffineBijection
import numpy as np


class Dequantization(PreprocessingFlowLayer):
    def __init__(self, dist):
        super(Dequantization, self).__init__()
        # dist should be a distribution with support on [0, 1]^d
        self.dist = dist

    def forward(self, input, context=None):
        # Note, input is the context for distribution model.
        noise, log_qnoise = self.dist.sample(input.size(0), context=input)  #, input.float())
        return input + noise, -log_qnoise

    def reverse(self, input, context=None):
        return input.floor()

    def logdet(self, input, context=None):
        raise NotImplementedError


class UniformCatDequantization(PreprocessingFlowLayer):
    '''
    A uniform dequantization layer for mixed-precision discrete data.
    This is useful for converting discrete variables to continuous [1, 2].

    Forward:
        `z = (x+u)/K, u~Unif(0,1)^D`
        where `x` is discrete, `x \in {0,1,2,...,K-1}^D`.
    Inverse:
        `x = Quantize(z, K)`

    Args:
        num_cats: list, number of K categories in quantization,
            i.e. [8, 5] for `[x0 \in {0,1,2,...,7}` ,`x1 \in {0,1,2,...,4}]`.

    References:
        [1] RNADE: The real-valued neural autoregressive density-estimator,
            Uria et al., 2013, https://arxiv.org/abs/1306.0186
        [2] Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design,
            Ho et al., 2019, https://arxiv.org/abs/1902.00275
    '''

    def __init__(self, num_cats=[1]):
        super(UniformCatDequantization, self).__init__()
        self.D = len(num_cats)
        self.register_buffer('qbins', torch.tensor(num_cats, dtype=torch.float))  # D
        self.register_buffer('ldj_per_dim', -torch.log(torch.tensor(num_cats, dtype=torch.float)))  # D
        #self.ScalarAffineBijection = ScalarAffineBijection(shift=-1.0, scale=2.0)

    def forward(self, input):
        x, context = input
        u = torch.rand(x.shape, device=self.qbins.device, dtype=self.qbins.dtype)
        z = (x.type(u.dtype) + u) / self.qbins  # BxD / D
        shape = x.shape
        batch_size = shape[0]
        num_dims = shape[1:].numel()
        ldj = (self.ldj_per_dim * num_dims).sum(-1).repeat(batch_size) #+ ldj_s
        return z, ldj

    def reverse(self, z, context=None):
        z = z * self.qbins  # BxD * D
        return z.floor().clamp(min=0, max=self.qbins-1).long()

    def logdet(self, x, context=None):
        raise NotImplementedError


class VariationalCatDequantization(PreprocessingFlowLayer):
    '''
    A variational dequantization layer for mixed-precision discrete data.
    This is useful for converting discrete variables to continuous [1, 2].

    Forward:
        `z = (x+u)/K, u~encoder(x)`
        where `x` is discrete, `x \in {0,1,2,...,K-1}^D`
        and `encoder` is a conditional distribution.
    Inverse:
        `x = Quantize(z, K)`

    Args:
        encoder: ConditionalDistribution, a conditional distribution/flow which
            outputs samples in `[0,1]^D` conditioned on `x`.
        num_cats: list, number of K categories in quantization,
            i.e. [8, 5] for `[x0 \in {0,1,2,...,7}` ,`x1 \in {0,1,2,...,4}]`.

    References:
        [1] RNADE: The real-valued neural autoregressive density-estimator,
            Uria et al., 2013, https://arxiv.org/abs/1306.0186
        [2] Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design,
            Ho et al., 2019, https://arxiv.org/abs/1902.00275
    '''

    def __init__(self, encoder, num_cats=[1]):
        super(VariationalCatDequantization, self).__init__()
        self.D = len(num_cats)
        self.register_buffer('qbins', torch.tensor(num_cats, dtype=torch.float))  # D
        self.register_buffer('ldj_per_dim', -torch.log(torch.tensor(num_cats, dtype=torch.float)))  # D
        self.encoder = encoder
        self.sigmoid = Sigmoid()
        #self.ScalarAffineBijection = ScalarAffineBijection(shift=-1.0, scale=2.0)

    def forward(self, input):
        x, context = input
        u, qu = self.encoder.sample(x, context)
        u, act_ldj = self.sigmoid(u)
        z = (x.type(u.dtype) + u) / self.qbins  # BxD / D
        shape = x.shape
        batch_size = shape[0]
        num_dims = shape[1:].numel()
        ldj = (self.ldj_per_dim * num_dims).sum(-1).repeat(batch_size) #+ ldj_s
        return z, ldj + act_ldj - qu

    def reverse(self, z, context=None):
        z = z * self.qbins  # BxD * D
        return z.long()
        #for d in range(self.D):
        #    z[:,d] = z[:,d].floor().clamp(min=0, max=self.qbins[d]-1).long()
        #return z

    def logdet(self, x, context=None):
        raise NotImplementedError


class EyeSampling(PreprocessingFlowLayer):
    def __init__(self):
        super(EyeSampling, self).__init__()
        
    def forward(self, input):
        x, context = input
        #print('EyeSampling:', x.shape, context.shape)
        return x, torch.zeros(x.shape[0], device=x.device)

    def reverse(self, z, context=None):
        return z.long()

    def logdet(self, x, context=None):
        raise NotImplementedError


class ProbSampling(PreprocessingFlowLayer):
    def __init__(self, encoder):
        super(ProbSampling, self).__init__()
        self.encoder = encoder
        self.sigmoid = Sigmoid()
        #self.ScalarAffineBijection = ScalarAffineBijection(shift=-1.0, scale=2.0)

    def forward(self, input):
        x, context = input
        #print('x/context:', x.shape, context.shape)
        u, qu = self.encoder.sample(x, context)
        u, act_ldj = self.sigmoid(u)
        #u, ldj_s = self.ScalarAffineBijection(u)
        #print('x/u/qu:', x.shape, u.shape, qu.shape, act_ldj.shape, ldj_s.shape)
        ldj = act_ldj + qu  #+ ldj_s
        return u, ldj

    def reverse(self, z, context=None):
        return z.long()

    def logdet(self, x, context=None):
        raise NotImplementedError


######################## ArgMax ########################
class ArgmaxCatDequantization(PreprocessingFlowLayer):
    '''
    A generative argmax surjection using a Cartesian product of binary spaces. Argmax is performed over the final dimension.
    Args:
        encoder: ConditionalDistribution, a distribution q(z|x) with support over z s.t. x=argmax z.
    Example:
        Input tensor x of shape (B, D, L) with discrete values {0,1,...,C-1}:
        encoder should be a distribution of shape (B, D, L, D), where D=ceil(log2(C)).
        When e.g. C=27, we have D=5, such that 2**5=32 classes are represented.
    '''

    def __init__(self, encoder, num_cats=[1]):
        super(ArgmaxCatDequantization, self).__init__()
        self.encoder = encoder
        self.num_bits = self.cats2bits(num_cats)
        #print('num_cats/num_bits:', num_cats, self.num_bits)
        self.sigmoid = Sigmoid()
        self.softplus = Softplus()
    
    @staticmethod
    def cats2bits(num_cats):
        if isinstance(num_cats, (list, tuple)):
            return [int(np.ceil(np.log2(cat))) for cat in num_cats]
        else:
            return  int(np.ceil(np.log2(num_cats)))

    def integer_to_base(self, idx_tensor, base, dims):
        '''
        Encodes index tensor to a Cartesian product representation.
        Args:
            idx_tensor (LongTensor): An index tensor, shape (...), to be encoded.
            base (int): The base to use for encoding.
            dims (int): The number of dimensions to use for encoding.
        Returns:
            LongTensor: The encoded tensor, shape (..., dims).
        '''
        powers = base ** torch.arange(dims - 1, -1, -1, device=idx_tensor.device)
        floored = idx_tensor[..., None] // powers
        remainder = floored % base

        base_tensor = remainder
        return base_tensor

    def idx2base(self, idx_tensor):
        if isinstance(self.num_bits, (list, tuple)): 
            for i,dims in enumerate(self.num_bits):
                binary = self.integer_to_base(context[:,i].unsqueeze(1), base=2, dims=dims)
                if i==0: all_binary = binary
                else:    all_binary = torch.cat((all_binary, binary), -1)
        else:
            all_binary = self.integer_to_base(idx_tensor, base=2, dims=self.num_bits)
        return all_binary

    def base2idx(base_tensor, base=2):
        '''
        Decodes Cartesian product representation to an index tensor.
        Args:
            base_tensor (LongTensor): The encoded tensor, shape (..., dims).
            base (int): The base used in the encoding.
        Returns:
            LongTensor: The index tensor, shape (...).
        '''
        dims = base_tensor.shape[-1]
        powers = base ** torch.arange(dims - 1, -1, -1, device=base_tensor.device)
        powers = powers[(None,) * (base_tensor.dim()-1)]

        idx_tensor = (base_tensor * powers).sum(-1)
        return idx_tensor

    def forward(self, input):
        x, context = input
        #print('x:', x.shape, context.shape)

        # Example: context.shape = (B, D) with values in {0,1,...,K-1}
        # Sample z.shape = (B, D, K)
        #print('enc:', context.shape)
        if isinstance(self.num_bits, (list, tuple)):
            for i, bits in enumerate(self.num_bits):
                binary = self.integer_to_base(context[:,i], base=2, dims=bits)
                #print(i, bits, binary.shape)
                if i==0: all_binary = binary
                else:    all_binary = torch.cat((all_binary, binary), -1)
        else:
            all_binary = self.integer_to_base(context, base=2, dims=self.num_bits)
        
        # BxK
        if all_binary.size(-1) % 2: all_binary = torch.cat([all_binary, torch.zeros(all_binary.size(0), 1, device=all_binary.device)], dim=-1)
        sign = all_binary * 2 - 1  # {-1,1}
        #print('sign:', sign.shape)
        u, qu = self.encoder.sample(x, context)
        #print('u/qu: ', u.shape, qu.shape)
        #u_positive, act_ldj = self.softplus(u)
        u_positive, act_ldj = self.sigmoid(u)
        #print('qu/act_ldj:', qu.shape, act_ldj.shape)

        ldj = act_ldj - qu
        z = u_positive * sign
        #print('z/shape:', z.shape, ldj.shape)
        return z, ldj

    def reverse(self, z, context=None):
        binary = torch.gt(z, 0.0).long()
        idx = self.base2idx(binary)
        return idx

    def logdet(self, x, context=None):
        raise NotImplementedError
