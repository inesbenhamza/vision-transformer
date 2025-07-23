import torch 
import torch.nn as nn 
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):

        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # size is batch_size, num_patches, embed_dim)
        self.position_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

    def forward(self, x):
        B = x.shape[0] 

        x = self.projection(x)  
        x = x.flatten(2).transpose(1, 2) 
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)  

        # Adding positional embeddings
        x = x + self.position_embed

        return x


class TransformerEncoder (nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = x + self.dropout1(attn_output)  
        x = self.norm1(x)  

        ff_output = F.gelu(self.fc1(x))
        ff_output = self.fc2(ff_output)
        x = x + self.dropout2(ff_output)  
        x = self.norm2(x)  

        return x
    


class VisionTransformer(nn.Module):
    def __init__( self, img_size=224, patch_size=16, in_channels=3, num_classes=10, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout_rate=0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        self.encoder_blocks = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_ratio, dropout_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)  

    def forward(self, x):
        x = self.patch_embed(x)  

        for block in self.encoder_blocks:
            x = block(x)

        x = self.norm(x)

        cls_token = x[:, 0]  
        out = self.head(cls_token)  

        return out