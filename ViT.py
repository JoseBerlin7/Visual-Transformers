# Importing libs & packages
import torch
from torch import nn

# Step 1: patch embedding
class PatchEmbedding(nn.Module):
    def __init__(self, n_channels, img_size, patch_size, embed_dim):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


# Step 2: Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias, mlp_ratio, p=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MHSA(embed_dim, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(num_heads, eps=1e-6)

        hidden_layer = int(embed_dim * mlp_ratio)

        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, hidden_layer),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden_layer, embed_dim),
            nn.Dropout(p)
        )


    def forward(self, x, return_attention=False):
        if return_attention:
            out, attn = self.attn(self.norm1(x), return_attention)
            x = x + out
            x = x + self.MLP(self.norm2(x))
            return x, attn
        else:
            out = self.attn(self.norm1(x), return_attention)
            x = x + out
            x = x + self.MLP(x)
            return x


# Step 2.1: MultiHead Self ATTENTION
class MHSA(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim*3, bias=qkv_bias)
        
        self.scale = (embed_dim // self.num_heads) ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x , return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        attn = (Q @ K.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V)
        out = out.transpose(2, 1).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        if return_attention:
            return out, attn
        return out


# ViT
class ViT(nn.Module):
    def __init__(self, n_channels=1, num_classses=10, img_size=28, patch_size=4, embed_dim=64, num_heads=8, depth=6, mlp_ratio=4.0, qkv_bias=False, p=0.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(n_channels, img_size, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p)


        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, qkv_bias, mlp_ratio, p) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.classify_head = nn.Linear(embed_dim, num_classses)

        self._init_weights()    # Better to initialize weights orelse random weights will be assigned as zeros

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.2)
        nn.init.normal_(self.pos_embed, std=0.2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    
    def forward(self, x, return_all_attention=False):

        B = x.shape[0]     # getting batch size for defining cls token
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        attentions = []
        if return_all_attention:
            for blk in self.blocks:
                x, attn = blk(x, return_all_attention)
                attentions.append(attn)
        else:
            for blk in self.blocks:
                x = blk(x)
        x = self.norm(x)
        final_cls = x[:, 0]
        logits =  self.classify_head(final_cls)
        if return_all_attention:
            return logits, attentions
        return logits