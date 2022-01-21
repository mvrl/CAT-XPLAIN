#This has models options for the paper idea "Inductive Bias Must Match"
from config import *

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
import torch
#Option 1: Self-Attention: ViT
# VIT
#Taken from:
#https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class modifiedViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,explain=False,device='cpu'):
        super().__init__()
        self.device = device
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.explain = explain
        if self.explain:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1 + 1, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.exp_token = nn.Parameter(torch.randn(1, 1, dim)) #Explaination token
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.cls_mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.exp_mlp_head = nn.Sequential(                   #MLP for Explaination
            nn.LayerNorm(dim),
            nn.Linear(dim, num_patches)
        )
        self.num_patches = num_patches
        self.lat_dim = dim
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        if self.explain:
            exp_tokens = repeat(self.exp_token, '() n d -> b n d', b = b) #Explaination token
            x = torch.cat((cls_tokens,exp_tokens, x), dim=1) #Add explaination token
            x += self.pos_embedding[:, :(n + 2)]
        else:
            x = torch.cat((cls_tokens, x), dim=1) 
            x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        out = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else out[:, 0]
        if self.explain:
            mask = out[:,1]
        else:
            mask = torch.ones(b,self.lat_dim).to(self.device)  #Just a placeholder
        mask = self.to_latent(mask) #to_latent is just a placeholder for now,i.e no extra MLP layers except MLP head   
        x = self.to_latent(x) #to_latent is just a placeholder for now,i.e no extra MLP layers except MLP head
        return self.cls_mlp_head(x), self.exp_mlp_head(mask)

# Option 2:
# ConvNet

class ConvNet(nn.Module):
    def __init__(self,num_logits,k=5):
        super(ConvNet, self).__init__()
        self.num_logits = num_logits
        self.c1 = nn.Sequential(nn.Conv2d(1, 8, k, bias=True), nn.MaxPool2d(2), nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv2d(8, 16, k, bias=True), nn.MaxPool2d(2), nn.ReLU())
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(4*4*16, self.num_logits) #conv without padding
    def forward(self, x):
        bs = x.size(0)
        o1 = self.c1(x)
        o2 = self.c2(o1)
        # x = self.dropout(x)
        out = self.fc(o2.view(bs,-1))
        return out


# Option 3:
#MLP
class MLPNet(nn.Module):
    def __init__(self,input_shape,num_logits):
        super(MLPNet, self).__init__()
        self.input_shape = input_shape
        self.pixels = self.input_shape[0]*self.input_shape[1]*self.input_shape[2]
        self.num_logits = num_logits
        self.fc1 = nn.Sequential(nn.Linear(self.pixels, self.pixels//2, bias=True), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(self.pixels//2, self.pixels//4, bias=True), nn.ReLU())
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.pixels//4, self.num_logits) 
    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs,-1)
        o1 = self.fc1(x)
        o2 = self.fc2(o1)
        # x = self.dropout(x)
        out = self.fc(o2.view(bs,-1))
        return out

class ConvNet_selector(nn.Module): # this class is obviously specific to the problem statement that we have
    #takes input 1x28x28 (single channel)
    def __init__(self, k=5):
        super(ConvNet_selector, self).__init__()
        self.c1 = nn.Sequential(nn.Conv2d(1,8,k, padding=2), nn.MaxPool2d(2), nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv2d(8,16,k, padding=2), nn.MaxPool2d(2), nn.ReLU())        
        self.c3 = nn.Conv2d(16,1,1)
    def forward(self, x):
        bs = x.size(0)
        o1 = self.c1(x)
        o2 = self.c2(o1)
        logits = self.c3(o2)
        
        return logits.view(bs,-1) #shape(bs, 49)