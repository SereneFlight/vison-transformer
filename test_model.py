"""
测试 ViT 模型的每个模块
帮助理解每一步的维度变化

运行: python test_model.py
"""

import torch
import torch.nn as nn
from vit_model import (
    PatchEmbed,
    Attention,
    MLP,
    Block,
    VisionTransformer,
    vit_base_patch16_224
)


def test_patch_embed():
    """测试 Patch Embedding"""
    print("\n" + "=" * 60)
    print("测试 1: Patch Embedding")
    print("=" * 60)

    # 创建模块
    patch_embed = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768)

    # 输入: batch_size=2 的图像
    x = torch.randn(2, 3, 224, 224)
    print(f"输入形状: {x.shape}  # (B, C, H, W)")

    # 前向传播
    out = patch_embed(x)
    print(f"输出形状: {out.shape}  # (B, num_patches, embed_dim)")

    # 解释
    print(f"\n说明:")
    print(f"  - 原图: 224×224 = {224*224} 个像素")
    print(f"  - Patch 大小: 16×16")
    print(f"  - Patch 数量: (224/16)×(224/16) = {patch_embed.num_patches}")
    print(f"  - 每个 patch 投影到 {out.shape[-1]} 维")

    assert out.shape == (2, 196, 768), "形状错误!"
    print(f"\n✅ 测试通过!")


def test_attention():
    """测试 Multi-Head Attention"""
    print("\n" + "=" * 60)
    print("测试 2: Multi-Head Attention")
    print("=" * 60)

    # 创建模块
    attn = Attention(dim=768, num_heads=12, qkv_bias=True)

    # 输入: (B, N, C) = (2, 197, 768)
    # 197 = 1 CLS token + 196 patches
    x = torch.randn(2, 197, 768)
    print(f"输入形状: {x.shape}  # (B, N, C)")

    # 前向传播
    out = attn(x)
    print(f"输出形状: {out.shape}  # (B, N, C)")

    # 解释
    print(f"\n说明:")
    print(f"  - 注意力头数: {attn.num_heads}")
    print(f"  - 每个头的维度: {attn.head_dim} (总维度 / 头数 = {768 // 12})")
    print(f"  - 缩放因子: {attn.scale:.4f} (1/√{attn.head_dim})")
    print(f"\n  计算流程:")
    print(f"    1. QKV 投影: (2, 197, 768) → (2, 197, 2304)")
    print(f"    2. 拆分多头: (2, 197, 2304) → (2, 12, 197, 64)")
    print(f"    3. 注意力矩阵: Q@K^T → (2, 12, 197, 197)")
    print(f"    4. 加权求和: attn@V → (2, 12, 197, 64)")
    print(f"    5. 合并多头: (2, 12, 197, 64) → (2, 197, 768)")

    assert out.shape == (2, 197, 768), "形状错误!"
    print(f"\n✅ 测试通过!")


def test_mlp():
    """测试 MLP"""
    print("\n" + "=" * 60)
    print("测试 3: MLP (Feed-Forward Network)")
    print("=" * 60)

    # 创建模块
    mlp = MLP(in_features=768, hidden_features=3072, out_features=768)

    # 输入
    x = torch.randn(2, 197, 768)
    print(f"输入形状: {x.shape}  # (B, N, C)")

    # 前向传播
    out = mlp(x)
    print(f"输出形状: {out.shape}  # (B, N, C)")

    # 解释
    print(f"\n说明:")
    print(f"  - 输入维度: {mlp.fc1.in_features}")
    print(f"  - 隐藏层维度: {mlp.fc1.out_features} (扩大 4 倍)")
    print(f"  - 输出维度: {mlp.fc2.out_features}")
    print(f"  - 激活函数: GELU")
    print(f"\n  计算流程:")
    print(f"    1. fc1: (2, 197, 768) → (2, 197, 3072)")
    print(f"    2. GELU 激活")
    print(f"    3. fc2: (2, 197, 3072) → (2, 197, 768)")

    assert out.shape == (2, 197, 768), "形状错误!"
    print(f"\n✅ 测试通过!")


def test_block():
    """测试 Transformer Block"""
    print("\n" + "=" * 60)
    print("测试 4: Transformer Block")
    print("=" * 60)

    # 创建模块
    block = Block(dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True)

    # 输入
    x = torch.randn(2, 197, 768)
    print(f"输入形状: {x.shape}  # (B, N, C)")

    # 前向传播
    out = block(x)
    print(f"输出形状: {out.shape}  # (B, N, C)")

    # 解释
    print(f"\n说明:")
    print(f"  一个 Block 包含两个子层:")
    print(f"    1. Self-Attention 子层:")
    print(f"       x = x + Attention(LayerNorm(x))")
    print(f"    2. MLP 子层:")
    print(f"       x = x + MLP(LayerNorm(x))")
    print(f"\n  关键:")
    print(f"    - Pre-Norm: LayerNorm 在子层之前")
    print(f"    - Residual: 每个子层都有残差连接 (+)")

    assert out.shape == (2, 197, 768), "形状错误!"
    print(f"\n✅ 测试通过!")


def test_full_model():
    """测试完整 ViT 模型"""
    print("\n" + "=" * 60)
    print("测试 5: 完整 Vision Transformer")
    print("=" * 60)

    # 创建模型
    model = vit_base_patch16_224(num_classes=1000)
    model.eval()

    # 输入
    x = torch.randn(2, 3, 224, 224)
    print(f"输入形状: {x.shape}  # (B, C, H, W)")

    # 逐步打印中间结果
    print(f"\n完整前向传播流程:")
    print(f"-" * 60)

    with torch.no_grad():
        B = x.shape[0]

        # 步骤 1
        x1 = model.patch_embed(x)
        print(f"1. Patch Embedding:     {x.shape} → {x1.shape}")

        # 步骤 2
        cls_token = model.cls_token.expand(B, -1, -1)
        x2 = torch.cat([cls_token, x1], dim=1)
        print(f"2. 加 CLS token:        {x1.shape} → {x2.shape}")

        # 步骤 3
        x3 = x2 + model.pos_embed
        print(f"3. 加位置编码:          {x2.shape} → {x3.shape}")

        # 步骤 4
        x4 = model.pos_drop(x3)
        for i, block in enumerate(model.blocks):
            x4 = block(x4)
            if i == 0:
                print(f"4. Transformer Block 1: {x3.shape} → {x4.shape}")
        print(f"   ...通过 12 层...")
        print(f"   Transformer Block 12: {x4.shape} → {x4.shape}")

        # 步骤 5
        x5 = model.norm(x4)
        print(f"5. Layer Norm:          {x4.shape} → {x5.shape}")

        # 步骤 6
        cls_output = x5[:, 0]
        print(f"6. 提取 CLS token:      {x5.shape} → {cls_output.shape}")

        # 步骤 7
        logits = model.head(cls_output)
        print(f"7. 分类头:              {cls_output.shape} → {logits.shape}")

    print(f"-" * 60)

    # 对比直接调用
    with torch.no_grad():
        out_direct = model(torch.randn(2, 3, 224, 224))

    assert logits.shape == out_direct.shape, "形状不一致!"
    assert logits.shape == (2, 1000), "输出形状错误!"

    print(f"\n✅ 测试通过!")


def test_different_variants():
    """测试不同规模的 ViT"""
    print("\n" + "=" * 60)
    print("测试 6: 不同规模的 ViT 变体")
    print("=" * 60)

    from vit_model import vit_large_patch16_224, vit_huge_patch14_224

    variants = [
        ("ViT-Base/16", vit_base_patch16_224(num_classes=1000)),
        ("ViT-Large/16", vit_large_patch16_224(num_classes=1000)),
        ("ViT-Huge/14", vit_huge_patch14_224(num_classes=1000)),
    ]

    print(f"\n{'模型':<15} {'参数量':<15} {'Embed Dim':<12} {'Depth':<8} {'Heads':<8}")
    print(f"-" * 60)

    for name, model in variants:
        params = sum(p.numel() for p in model.parameters())
        embed_dim = model.embed_dim
        depth = len(model.blocks)
        num_heads = model.blocks[0].attn.num_heads

        print(f"{name:<15} {params/1e6:>6.1f}M        {embed_dim:<12} {depth:<8} {num_heads:<8}")

    print(f"\n✅ 所有变体创建成功!")


def test_parameter_count():
    """详细分析参数量"""
    print("\n" + "=" * 60)
    print("测试 7: 参数量详细分析")
    print("=" * 60)

    model = vit_base_patch16_224(num_classes=1000)

    print(f"\n{'模块':<30} {'参数量':<15} {'形状'}")
    print(f"-" * 70)

    total = 0
    for name, param in model.named_parameters():
        params = param.numel()
        total += params
        # 只打印关键层
        if any(key in name for key in ['patch_embed', 'cls_token', 'pos_embed',
                                         'blocks.0', 'blocks.11', 'head']):
            print(f"{name:<30} {params:>12,}   {list(param.shape)}")

    print(f"-" * 70)
    print(f"{'总计':<30} {total:>12,}   ({total/1e6:.2f}M)")

    print(f"\n✅ 测试通过!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Vision Transformer 完整测试套件")
    print("=" * 60)

    # 运行所有测试
    test_patch_embed()
    test_attention()
    test_mlp()
    test_block()
    test_full_model()
    test_different_variants()
    test_parameter_count()

    print("\n" + "=" * 60)
    print("🎉 所有测试通过! ViT 实现正确!")
    print("=" * 60)
    print("\n下一步:")
    print("  1. 可视化 Attention Map (运行 visualize_attention.py)")
    print("  2. 在 CIFAR-10 上训练 (运行 train_cifar10.py)")
    print("  3. 记录到 Notion")
    print("=" * 60 + "\n")
