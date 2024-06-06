import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .flowlayer import FlowLayer
from einops import rearrange, repeat


class Conv1x1(FlowLayer):
    def __init__(self, data_size, context_net=None, contextflow=False):
        super().__init__()
        D, H, W = data_size if len(data_size)==3 else (data_size[0], 1, 1)
        self.D = D
        self.H = H
        self.W = W
        self.NN = nn.Parameter(torch.Tensor(D, D))
        nn.init.orthogonal_(self.NN)
        self.context_net = context_net
        self.contextflow = contextflow
        if self.context_net:  # discrete context
            self.C = C = self.context_net[0].C if isinstance(self.context_net, list) else self.context_net.C
            self.CN = nn.Linear(C, D*D)
            nn.init.zeros_(self.CN.weight)
            nn.init.zeros_(self.CN.bias)
            if self.contextflow:
                self.NN.requires_grad_(False)

    def forward(self, x, context=None):
        B, _, H, W = x.shape
        
        if self.context_net:
            c, logp_c = self.context_net(context)
            logp_c = logp_c * H * W
            c = self.CN(c)  # BxC -> BxD*D
            c = rearrange(c, 'b (d1 d2) -> b d1 d2', d1=self.D, d2=self.D) # BxD*D -> BxDxD
            # triangularize for speed and stability:
            c_diag = torch.diagonal(torch.tril(c), dim1=-2, dim2=-1)  # log at the diagonal
            c_ldj = torch.sum(c_diag, dim=-1)  # B

            if self.contextflow:
                c = torch.tril(c, diagonal=-1) + torch.diag_embed(torch.exp(c_diag)) - torch.eye(self.D).to(x.device)
                c = repeat(c, 'b d1 d2 -> (b h w) d1 d2', h=H, w=W)
                ldj = H * W * (torch.slogdet(self.NN)[1] + c_ldj)
                z = torch.bmm(c + self.NN, rearrange(x, 'b d2 h w -> (b h w) d2 1'))
            else:
                c = torch.tril(c, diagonal=-1) + torch.diag_embed(torch.exp(c_diag), dim1=-2, dim2=-1)
                c = repeat(c, 'b d1 d2 -> (b h w) d1 d2', h=H, w=W)
                ldj = H * W * c_ldj
                z = torch.bmm(c, rearrange(x, 'b d2 h w -> (b h w) d2 1'))
            z = rearrange(z, '(b h w) d 1 -> b d h w', h=H, w=W)

        else:
            ldj = torch.slogdet(self.NN)[1] * H * W
            z = F.conv2d(x, rearrange(self.NN, 'k2 k1 -> k2 k1 () ()'))
            logp_c = torch.zeros(B).to(device=x.device)

        return z, ldj + logp_c
    
    # to update:
    def reverse(self, z, context=None):
        if self.context_net:
            c = self.context_net(context)
            w_cinv = torch.matmul(w_ginv, self.CN)
            if self.contextflow:
                w_ginv = torch.inverse(self.NN)
                x = F.conv2d(z, rearrange(w_ginv, 'k2 k1 -> k2 k1 () ()'))
                w_cinv = torch.matmul(w_ginv, self.CN)
                x-= F.conv2d(c, rearrange(w_cinv, 'k2 k1 -> k2 k1 () ()'))  # x = W_inv z - W_inv Bc
            else:
                x = F.conv2d(z, rearrange(torch.inverse(self.CN), 'k2 k1 -> k2 k1 () ()'))
        else:
            x = F.conv2d(z, rearrange(torch.inverse(self.NN), 'k2 k1 -> k2 k1 () ()'))
        return x

    def logdet(self, input, context=None):
        _, ldj = self.forward(input, context)
        return ldj


class FC(Conv1x1):
    def __init__(self, data_size, context_net=None, contextflow=False):
        super().__init__(data_size, context_net=None, contextflow=False)

    def forward(self, x, context=None):
        x = x.view(-1, self.D, 1, 1)
        output, ldj = super().forward(x, context)
        return output.view(-1, self.D), ldj

    def reverse(self, z, context=None):
        z = z.view(-1, self.D, 1, 1)
        output = super().reverse(z, context)
        return output.view(-1, self.D)

    def logdet(self, x, context=None):
        x = x.view(-1, self.D, 1, 1)
        return super().logdet(x)


'''class Conv1x1Householder(FlowLayer):
    def __init__(self, data_channels, reflections, context_net=None, contextflow=False):
        super().__init__()
        self.D = D = data_channels
        self.K = K = reflections
        assert K <= D, 'reflections <= data_channels'
        self.NN_V = torch.nn.Parameter(torch.randn(K, D))
        self.context_net = context_net
        self.contextflow = contextflow
        if self.context_net:
            self.C = C = self.context_net.C
            self.CV = torch.nn.Linear(C, K*D)
            if self.contextflow:
                nn.init.zeros_(self.CN.weight)
                nn.init.zeros_(self.CN.bias)

    def contruct_Q(self, V):
        I = torch.eye(self.D, device=V.device)
        Q = I
        for k in range(self.K):
            v   = rearrange(V[k], 'd -> d ()')
            vvT = torch.matmul(v, v.T)
            vTv = torch.matmul(v.T, v)
            Q   = torch.matmul(Q, I - 2 * vvT / vTv)
        return Q

    def forward(self, x, context=None, reverse=False):
        Q = self.contruct_Q(self.NN_V)
        if self.context_net:
            c, logp_c  = self.context_net(context)
            #print('x/c:', x.shape, c.shape, logp_c.shape)
            c = self.CN(c)  # BxC -> BxD*D
            #print('c2:', c.shape)
            C = rearrange(c, 'b (d1 d2) -> b d1 d2', d1=self.K, d2=self.D)
            C = torch.mean(c, dim=0)
            QC= self.contruct_Q(C)
            #print('c3:', c.shape)
            logp_c = logp_c * H * W
            if self.contextflow:
                if not reverse:
                    z = F.conv2d(x, rearrange(Q,   'k2 k1 -> k2 k1 () ()'))
                    z+= F.conv2d(x, rearrange(QC,  'k2 k1 -> k2 k1 () ()'))
                else:
                    z = F.conv2d(x, rearrange(Q.T, 'k2 k1 -> k2 k1 () ()'))
                    z+= F.conv2d(x, rearrange(QC.T,'k2 k1 -> k2 k1 () ()'))
            else:
                if not reverse: z = F.conv2d(x, rearrange(QC,   'k2 k1 -> k2 k1 () ()'))
                else:           z = F.conv2d(x, rearrange(QC.T, 'k2 k1 -> k2 k1 () ()'))    
        else:
            if not reverse: z = F.conv2d(x, rearrange(Q,   'k2 k1 -> k2 k1 () ()'))
            else:           z = F.conv2d(x, rearrange(Q.T, 'k2 k1 -> k2 k1 () ()'))
        
        return z, torch.zeros(x.size(0)).to(x.device)

    def reverse(self, z, context=None):
        return self.forward(z, context, reverse=True)[0]

    def logdet(self, x, context=None):
        return torch.zeros(x.size(0)).to(x.device)
'''