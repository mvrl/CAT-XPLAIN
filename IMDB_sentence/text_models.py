
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
import torch

# Modified from the code for VIT
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

class TextTransformer(nn.Module):
    def __init__(self, *, num_classes, max_length, emb_dim, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,train_emb=True):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.train_emb = train_emb
        if self.train_emb:
          self.to_embedding = nn.Linear(emb_dim, dim)
        else:
          dim = emb_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, max_length + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.max_length = max_length
        self.emb_dim = dim

    def forward(self, x,mask=[]):
        b = x.shape[0]
        n = self.max_length
        if self.train_emb:
            x = self.to_embedding(x)
        if len(mask) != 0:
            mask = torch.tensor(mask).unsqueeze(2)
            mask = mask.expand(b,self.max_length,self.emb_dim)
            x = torch.mul(x, mask)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class modifiedTextTransformer(nn.Module):
    def __init__(self, *, num_classes, max_length, emb_dim, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,explain=False,device='cpu',train_emb=True):
        super().__init__()
        self.device = device
        self.train_emb = train_emb
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        if self.train_emb:
          self.to_embedding = nn.Linear(emb_dim, dim)
        else:
          dim = emb_dim
        self.explain = explain
        if self.explain:
            self.pos_embedding = nn.Parameter(torch.randn(1, max_length + 1 + 1, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, max_length + 1, dim))
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
            nn.Linear(dim, max_length)
        )
        self.max_length = max_length
        self.emb_dim = dim
        self.lat_dim = dim
        
    def forward(self, x,mask=[]):
        b = x.shape[0]
        n = self.max_length
        if self.train_emb:
            x = self.to_embedding(x)
        if len(mask) != 0:
            mask = torch.tensor(mask).unsqueeze(2)
            mask = mask.expand(b,self.max_length,self.emb_dim)
            x = torch.mul(x, mask)
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


if __name__ == "__main__":

    #Sanity check
    batch_size = 64
    max_sents = 50
    one_batch = a = torch.rand(64,50,384)
    model1 = TextTransformer(num_classes =2, max_length=max_sents, emb_dim = 384, dim=128, depth=3, heads=8, mlp_dim=256, pool = 'cls', channels =1, dim_head = 64, dropout = 0., emb_dropout = 0.,train_emb = True).to('cpu')
    model2 = modifiedTextTransformer(num_classes =2, max_length=max_sents, emb_dim=384, dim=128, depth=3, heads=8, mlp_dim=256, pool = 'cls', channels =1, dim_head = 64, dropout = 0., emb_dropout = 0.,explain=True,train_emb = True).to('cpu')
    
    print(model1.forward(one_batch).shape)
    print(model2.forward(one_batch)[0].shape, model2.forward(one_batch)[1].shape)
