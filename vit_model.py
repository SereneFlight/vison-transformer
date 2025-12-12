"""
Vision Transformer (ViT) 完整实现
参考论文: An Image is Worth 16x16 Words (ICLR 2021)

作者: 你的名字
日期: 2025-12-07

代码结构:
1. PatchEmbed - 图像切分成 patches
2. Attention - 多头自注意力机制
3. MLP - 前馈神经网络
4. Block - 一个完整的 Transformer Block
5. VisionTransformer - 完整模型
"""

import torch
import torch.nn as nn
import math


# ========================================
# 模块 1: Patch Embedding
# ========================================

class PatchEmbed(nn.Module):
    """
    把图像切分成 patches 并投影到 embedding 维度

    输入: (B, 3, 224, 224)
    输出: (B, 196, 768)

    参数:
        img_size: 输入图像大小 (默认 224)
        patch_size: 每个 patch 的大小 (默认 16)
        in_chans: 输入通道数 (RGB=3)
        embed_dim: 嵌入维度 (默认 768)
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # 计算有多少个 patches.
        # 例如: 224 / 16 = 14, 所以有 14×14 = 196 个
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 196

        # 用卷积实现 patch embedding
        # kernel_size=16, stride=16 → 不重叠地扫描整个图像
        # 等价于: 先切成 16×16 的块，再对每块做线性变换
        self.proj = nn.Conv2d(
            in_channels=in_chans,      # 3
            out_channels=embed_dim,    # 768
            kernel_size=patch_size,    # 16
            stride=patch_size          # 16
        )

    def forward(self, x):
        """
        x: (B, 3, 224, 224) - 输入图像

        返回: (B, 196, 768) - patch embeddings
        """
        B, C, H, W = x.shape

        # 步骤1: 卷积投影
        # (B, 3, 224, 224) → (B, 768, 14, 14)
        x = self.proj(x)

        # 步骤2: 展平空间维度
        # (B, 768, 14, 14) → (B, 768, 196)
        # flatten(2) 表示从第2维开始展平 (维度编号: 0=B, 1=C, 2=H, 3=W)
        x = x.flatten(2)

        # 步骤3: 转置，把序列长度放到第1维
        # (B, 768, 196) → (B, 196, 768)
        x = x.transpose(1, 2)

        return x


# ========================================
# 模块 2: Multi-Head Self-Attention
# ========================================

class Attention(nn.Module):
    """
    多头自注意力机制

    核心公式: Attention(Q,K,V) = softmax(QK^T / √d) V

    参数:
        dim: 输入维度 (768)
        num_heads: 注意力头数 (12)
        qkv_bias: 是否在 QKV 投影中使用 bias
        attn_drop: 注意力权重的 dropout 概率
        proj_drop: 输出投影的 dropout 概率
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768 / 12 = 64 (每个头的维度)
        self.scale = self.head_dim ** -0.5  # 1 / √64 ≈ 0.125

        # 一次性生成 Q, K, V 三个矩阵 (更高效)
        # 输入: (B, 197, 768)
        # 输出: (B, 197, 2304)  # 2304 = 768 * 3
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # 注意力权重的 dropout
        self.attn_drop = nn.Dropout(attn_drop)

        # 输出投影
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        x: (B, N, C) - 输入 tokens
           B = batch size
           N = 197 (1 CLS + 196 patches)
           C = 768 (embedding dim)

        返回: (B, N, C) - 注意力后的输出
        """
        B, N, C = x.shape

        # ===== 步骤1: 生成 Q, K, V =====
        # (B, 197, 768) → (B, 197, 2304)
        qkv = self.qkv(x)

        # 重塑成 (B, N, 3, num_heads, head_dim)
        # (B, 197, 2304) → (B, 197, 3, 12, 64)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)

        # 转置成 (3, B, num_heads, N, head_dim)
        # 这样方便后面分离 Q, K, V
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # 分离 Q, K, V
        # 每个形状: (B, 12, 197, 64)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # ===== 步骤2: 计算注意力分数 =====
        # Q @ K^T: (B, 12, 197, 64) × (B, 12, 64, 197) = (B, 12, 197, 197)
        # 每个元素 attn[b, h, i, j] 表示:
        # "在第 b 个样本的第 h 个头中，token i 对 token j 的关注程度"
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # ===== 步骤3: Softmax 归一化 =====
        # 对最后一维做 softmax，使每一行和为 1
        # (B, 12, 197, 197)
        attn = attn.softmax(dim=-1)

        # Dropout (随机丢弃一些注意力连接)
        attn = self.attn_drop(attn)

        # ===== 步骤4: 加权求和 =====
        # attn @ V: (B, 12, 197, 197) × (B, 12, 197, 64) = (B, 12, 197, 64)
        # 对每个 token，按注意力权重加权所有 value
        x = attn @ v

        # ===== 步骤5: 合并多头 =====
        # 转置: (B, 12, 197, 64) → (B, 197, 12, 64)
        x = x.transpose(1, 2)

        # 重塑: (B, 197, 12, 64) → (B, 197, 768)
        x = x.reshape(B, N, C)

        # ===== 步骤6: 输出投影 =====
        # (B, 197, 768) → (B, 197, 768)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# ========================================
# 模块 3: MLP (Feed-Forward Network)
# ========================================

class MLP(nn.Module):
    """
    两层全连接网络，用于对每个 token 做非线性变换

    结构:
        Linear(768 → 3072)
        → GELU
        → Dropout
        → Linear(3072 → 768)
        → Dropout

    参数:
        in_features: 输入维度 (768)
        hidden_features: 隐藏层维度 (3072 = 768*4)
        out_features: 输出维度 (768)
        act_layer: 激活函数 (GELU)
        drop: Dropout 概率
    """
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        # 默认值设置
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # 第一层: 扩大维度
        self.fc1 = nn.Linear(in_features, hidden_features)  # 768 → 3072

        # 激活函数: GELU (Gaussian Error Linear Unit)
        # 比 ReLU 更平滑，Transformer 标配
        self.act = act_layer()

        # Dropout
        self.drop1 = nn.Dropout(drop)

        # 第二层: 压缩回原维度
        self.fc2 = nn.Linear(hidden_features, out_features)  # 3072 → 768

        # Dropout
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        """
        x: (B, N, C) - 输入

        返回: (B, N, C) - 输出 (形状不变)
        """
        x = self.fc1(x)      # (B, 197, 768) → (B, 197, 3072)
        x = self.act(x)      # GELU 激活
        x = self.drop1(x)    # Dropout
        x = self.fc2(x)      # (B, 197, 3072) → (B, 197, 768)
        x = self.drop2(x)    # Dropout
        return x


# ========================================
# 模块 4: Transformer Block
# ========================================

class Block(nn.Module):
    """
    一个完整的 Transformer Block

    结构 (Pre-Norm):
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    参数:
        dim: 嵌入维度 (768)
        num_heads: 注意力头数 (12)
        mlp_ratio: MLP 隐藏层的扩展倍数 (4)
        qkv_bias: 是否在 QKV 中使用 bias
        drop: Dropout 概率
        attn_drop: 注意力 Dropout 概率
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0.):
        super().__init__()

        # 第一个 LayerNorm (在 Attention 之前)
        self.norm1 = nn.LayerNorm(dim)

        # Multi-Head Self-Attention
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        # 第二个 LayerNorm (在 MLP 之前)
        self.norm2 = nn.LayerNorm(dim)

        # MLP (Feed-Forward Network)
        mlp_hidden_dim = int(dim * mlp_ratio)  # 768 * 4 = 3072
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=drop
        )

    def forward(self, x):
        """
        x: (B, N, C) - 输入

        返回: (B, N, C) - 输出 (形状不变)
        """
        # 第一个子层: Self-Attention + 残差连接
        # Pre-Norm: 先 LayerNorm，再 Attention，最后加残差
        x = x + self.attn(self.norm1(x))

        # 第二个子层: MLP + 残差连接
        # Pre-Norm: 先 LayerNorm，再 MLP，最后加残差
        x = x + self.mlp(self.norm2(x))

        return x


# ========================================
# 模块 5: Vision Transformer 完整模型
# ========================================

class VisionTransformer(nn.Module):
    """
    Vision Transformer 完整模型

    默认配置 (ViT-Base/16):
        - 图像大小: 224×224
        - Patch 大小: 16×16
        - Embedding 维度: 768
        - 深度: 12 层
        - 注意力头数: 12
        - MLP ratio: 4
        - 参数量: ~86M

    参数:
        img_size: 输入图像大小
        patch_size: Patch 大小
        in_chans: 输入通道数
        num_classes: 分类类别数
        embed_dim: 嵌入维度
        depth: Transformer 层数
        num_heads: 注意力头数
        mlp_ratio: MLP 隐藏层扩展倍数
        qkv_bias: 是否使用 QKV bias
        drop_rate: Dropout 概率
        attn_drop_rate: 注意力 Dropout 概率
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        # ===== 1. Patch Embedding =====
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches  # 196

        # ===== 2. CLS Token (可学习参数) =====
        # 形状: (1, 1, 768)
        # 会在 forward 时扩展成 (B, 1, 768)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ===== 3. Position Embedding (可学习参数) =====
        # 需要 197 个位置: 1 个 CLS + 196 个 patches
        # 形状: (1, 197, 768)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Position Embedding 后的 Dropout
        self.pos_drop = nn.Dropout(p=drop_rate)

        # ===== 4. Transformer Blocks (12 层) =====
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for i in range(depth)
        ])

        # ===== 5. 最后的 Layer Norm =====
        self.norm = nn.LayerNorm(embed_dim)

        # ===== 6. 分类头 =====
        # 只用 CLS token 的输出做分类
        self.head = nn.Linear(embed_dim, num_classes)

        # ===== 7. 初始化权重 =====
        # 使用截断正态分布初始化 (论文附录 B)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        权重初始化策略 (遵循论文)

        - Linear: 截断正态分布 (std=0.02)，bias 置 0
        - LayerNorm: weight 置 1，bias 置 0
        """
        if isinstance(m, nn.Linear):
            # 截断正态分布: 超过 2*std 的值会被重新采样
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        前向传播

        x: (B, 3, 224, 224) - 输入图像

        返回: (B, num_classes) - 类别 logits
        """
        B = x.shape[0]

        # ===== 步骤1: Patch Embedding =====
        # (B, 3, 224, 224) → (B, 196, 768)
        x = self.patch_embed(x)

        # ===== 步骤2: 加上 CLS token =====
        # CLS token: (1, 1, 768) → (B, 1, 768)
        cls_token = self.cls_token.expand(B, -1, -1)

        # 拼接: (B, 1, 768) + (B, 196, 768) → (B, 197, 768)
        x = torch.cat([cls_token, x], dim=1)

        # ===== 步骤3: 加上位置编码 =====
        # (B, 197, 768) + (1, 197, 768) → (B, 197, 768)
        # 广播加法: pos_embed 会自动扩展到 batch 维度
        x = x + self.pos_embed

        # Dropout
        x = self.pos_drop(x)

        # ===== 步骤4: 通过 12 层 Transformer =====
        for block in self.blocks:
            x = block(x)  # (B, 197, 768) → (B, 197, 768)

        # ===== 步骤5: 最后的 Layer Norm =====
        x = self.norm(x)  # (B, 197, 768)

        # ===== 步骤6: 只取 CLS token 的输出 =====
        # x[:, 0] 表示取每个样本的第 0 个 token (即 CLS)
        cls_output = x[:, 0]  # (B, 768)

        # ===== 步骤7: 分类头 =====
        logits = self.head(cls_output)  # (B, 768) → (B, num_classes)

        return logits


# ========================================
# 辅助函数: 创建不同规模的模型
# ========================================

def vit_base_patch16_224(num_classes=1000, **kwargs):
    """
    ViT-Base/16

    参数量: ~86M
    配置:
        - embed_dim: 768
        - depth: 12
        - num_heads: 12
        - patch_size: 16
    """
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        num_classes=num_classes,
        **kwargs
    )
    return model


def vit_large_patch16_224(num_classes=1000, **kwargs):
    """
    ViT-Large/16

    参数量: ~307M
    配置:
        - embed_dim: 1024
        - depth: 24
        - num_heads: 16
    """
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        num_classes=num_classes,
        **kwargs
    )
    return model


def vit_huge_patch14_224(num_classes=1000, **kwargs):
    """
    ViT-Huge/14

    参数量: ~632M
    配置:
        - embed_dim: 1280
        - depth: 32
        - num_heads: 16
        - patch_size: 14
    """
    model = VisionTransformer(
        img_size=224,
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        num_classes=num_classes,
        **kwargs
    )
    return model


# ========================================
# 测试代码
# ========================================

if __name__ == "__main__":
    print("=" * 60)
    print("Vision Transformer 测试")
    print("=" * 60)

    # 创建模型
    model = vit_base_patch16_224(num_classes=10)
    model.eval()  # 设置为评估模式

    # 随机输入 (batch_size=2)
    x = torch.randn(2, 3, 224, 224)

    # 前向传播
    with torch.no_grad():
        out = model(x)

    # 打印形状
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"预期输出: (2, 10) ← batch_size=2, num_classes=10")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")

    # 打印模型结构摘要
    print("\n" + "=" * 60)
    print("模型结构摘要")
    print("=" * 60)

    print(f"\nPatch Embedding:")
    print(f"  - Num patches: {model.patch_embed.num_patches}")
    print(f"  - Embed dim: {model.embed_dim}")

    print(f"\nTransformer:")
    print(f"  - Depth: {len(model.blocks)} 层")
    print(f"  - Num heads: {model.blocks[0].attn.num_heads}")
    print(f"  - Head dim: {model.blocks[0].attn.head_dim}")
    print(f"  - MLP hidden dim: {model.blocks[0].mlp.fc1.out_features}")

    print(f"\nClassification Head:")
    print(f"  - Input dim: {model.head.in_features}")
    print(f"  - Output dim (num_classes): {model.head.out_features}")

    print("\n" + "=" * 60)
    print("测试完成！✅")
    print("=" * 60)
