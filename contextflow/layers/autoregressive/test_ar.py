import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

B=2
K=5
T=16
D=3  # H*W
#H=1
#W=1
c = torch.randn(B,T,D).transpose(1,-1)  # BxTxD -> BxDxT
c = c[..., :T-K]
P = repeat(c.unsqueeze(0), ' () b d t -> k b d t', k=K)  # KxDxT-K
W = nn.ModuleList([nn.Conv1d(D, D, 1) for i in range(K)])
x = torch.randn(B,T,D).transpose(1,-1)  # BxTxD -> BxDxT
#Z = torch.zeros_like(P)
log_softmax = nn.Softmax2d()
l = 0.0
for k in range(K):
    p = W[k](c)  # BxDx(T-K)  # W_k c_t
    z = x[..., t+1:t+1+T-K]  # BxDx(T-K)
    p = rearrange(p, 'b d tk -> (b tk) d')
    z = rearrange(z, 'b d tk -> (b tk) d')
    f = z @ p.T  # B*(T-K)xD DxB*(T-K) -> B*(T-K)xB*(T-K)
    f = rearrange(f, '(b1 tk1) (b2 tk2) -> b1 b2 tk1 tk2', b1=B, b2=B, tk1=T-K, tk2=T-K)
    l+= torch.sum(torch.diag( torch.sum(log_softmax(f), (2,3)) ))
loss = -1.0 * l / c.numel()
