import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Patch_embed(nn.Module):
    # 输入：（B, 3, 224, 224）
    # 输出：(B, 768, 196)转置为(B, 196, 768)
    def __init__(self, img_size=224, patch_size=16, patch_embed=768, input_dim=3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_embed = patch_embed

        self. patch_num= (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels = input_dim, out_channels = patch_embed, kernel_size = patch_size, stride = patch_size)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x) #（B, 768, 14, 14), (224-16)/16 +1 = 14
        x = x.flatten(2) # (B, 768, 196)
        x = x.transpose(1, 2) # (B, 196, 768)

        return x
     
class Mult_Head_Inten(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.QKV = nn.Linear(embed_dim, 3*embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape[0], x.shape[1], x.shape[2]
        QKV = self.QKV(x)
        QKV = QKV.reshape(B, N, 3, self.embed_dim)
        Q, K, V = QKV[:, :, 0], QKV[:, :, 1], QKV[:, :, 2]
        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, N, head_dim)
        K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) # Q*K: (B, num_heads, N, N)
        attn_weights = torch.softmax(attn_score, dim=-1)
        attn_output = torch.matmul(attn_weights, V).transpose(1, 2) # (B, N, num_heads, head_dim)
        attn_output = attn_output.reshape(B, N, self.embed_dim)
        attn_output = self.proj(attn_output)
        return attn_output
    
class MLP(nn.Module):
    def __init__(self, embed_dim = 768, mlp_ratio = 4, drop=0.1):
        super().__init__()
        self.proj_1 = nn.Linear(embed_dim, embed_dim*mlp_ratio)
        self.proj_2 = nn.Linear(embed_dim*mlp_ratio, embed_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = F.gelu(self.proj_1(x))
        x = self.dropout(x)
        x = self.proj_2(x)
        x = self.dropout(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Mult_Head_Inten(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=10, embed_dim=768, depth=6, mlp_ratio=4, num_heads=12, drop=0.1):  # 修复1: 加冒号
        super().__init__()  # 修复2: super() 要加括号
        self.embed_dim = embed_dim

        # Patch Embedding
        self.patch_embed = Patch_embed(img_size=img_size, patch_size=patch_size, patch_embed=embed_dim, input_dim=3)  # 修复3: input_dim

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Position Embedding - 修复4: 正确计算大小
        num_patches = (img_size // patch_size) ** 2  # 196
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))  # 197

        # Dropout
        self.pos_drop = nn.Dropout(drop)

        # 修复6: 创建多个 TransformerBlock
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop)
            for _ in range(depth)
        ])

        # LayerNorm
        self.norm = nn.LayerNorm(embed_dim)

        # 修复8: 添加分类头
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)

        # Patch Embedding
        x = self.patch_embed(x)  # (B, 196, 768)

        # 扩展 CLS token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, 768)

        # 修复5: CLS token 放在最前面
        x = torch.cat([cls_token, x], dim=1)  # (B, 197, 768)

        # 加上位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 修复7: 正确遍历所有 Transformer Block
        for block in self.blocks:
            x = block(x)

        # LayerNorm
        x = self.norm(x)

        # 取出 CLS token 并分类
        cls_token_final = x[:, 0]  # (B, 768)
        output = self.head(cls_token_final)  # (B, num_classes)

        return output




