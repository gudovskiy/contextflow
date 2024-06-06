import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

''' example:
import torch
from layers.simple_vit import SimpleViT
v = SimpleViT(image_size = (14,28), patch_size = (2,2), dim = 1024, depth = 6, heads = 16, mlp_dim = 2048, channels=1)
img = torch.randn(1, 1, 14, 28)
preds = v(img) # (1, 1, 14, 28)
'''

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            #print(x.shape)
            x = attn(x) + x
            #print('attn:', x.shape)
            x = ff(x) + x
            #print('ff:', x.shape)
        return self.norm(x)


class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        #print(image_height, image_width, patch_height, patch_width)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width
        #print('patch_dim:', patch_dim, channels, patch_height, patch_width)
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        h = image_height // patch_height
        w = image_width // patch_width
        self.pos_embedding = posemb_sincos_2d(h, w, dim)
        #print('pos:', self.pos_embedding.shape, h, w)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        #self.to_patch = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width)
        self.to_image = Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1 = patch_height, p2 = patch_width, h=h, w=w)

    def forward(self, img):
        device = img.device
        #print('img:', img.shape, self.to_patch_embedding)
        x = self.to_patch_embedding(img)
        #print('x:', x.shape, self.pos_embedding.shape)
        x+= self.pos_embedding.to(device, dtype=x.dtype)
        #print('x+pos:', x.shape)
        x = self.transformer(x)
        x = self.to_image(x)
        #print('t:', x.shape)
        return x
