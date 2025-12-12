# Vision Transformer (ViT) 从零实现

> 参考论文：**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** (ICLR 2021)
>
> 论文地址：~/桌面/vison_transformer.pdf

---

## 一、项目结构

```
vit_from_scratch/
├── README.md              # 本文件：项目大纲和学习指南
├── vit_architecture.md    # ViT 架构详解（数学原理 + 图解）
├── vit_model.py           # 完整实现代码
├── test_model.py          # 测试代码：验证每层输出形状
├── visualize_attention.py # 可视化 Attention Map（复现论文 Figure 7）
├── train_cifar10.py       # 在 CIFAR-10 上训练
└── checkpoints/           # 保存训练好的模型
```

---

## 二、整体架构流程图

```
【输入】图像 224×224×3
    ↓
┌─────────────────────────────────────────┐
│ 1. Patch Embedding                      │
│    - 切分成 196 个 16×16 patches         │
│    - 线性投影到 768 维                   │
└─────────────────────────────────────────┘
    ↓ (B, 196, 768)
┌─────────────────────────────────────────┐
│ 2. 添加 [CLS] Token                     │
│    - 在序列最前面加一个可学习的 token     │
└─────────────────────────────────────────┘
    ↓ (B, 197, 768)  # 1 CLS + 196 patches
┌─────────────────────────────────────────┐
│ 3. 位置编码 (Position Embedding)        │
│    - 加上可学习的位置信息                │
└─────────────────────────────────────────┘
    ↓ (B, 197, 768)
┌─────────────────────────────────────────┐
│ 4. Transformer Encoder × 12 层          │
│                                         │
│  每一层包含：                            │
│  ┌───────────────────────────────────┐  │
│  │ (1) Layer Norm                    │  │
│  │ (2) Multi-Head Self-Attention     │  │
│  │     - 12 个注意力头                │  │
│  │     - 每个头 64 维                 │  │
│  │ (3) Residual Connection           │  │
│  └───────────────────────────────────┘  │
│            ↓                            │
│  ┌───────────────────────────────────┐  │
│  │ (4) Layer Norm                    │  │
│  │ (5) MLP (FFN)                     │  │
│  │     - 768 → 3072 → 768            │  │
│  │     - GELU 激活函数               │  │
│  │ (6) Residual Connection           │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
    ↓ (B, 197, 768)
┌─────────────────────────────────────────┐
│ 5. Layer Norm                           │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 6. 提取 CLS Token                       │
│    - 只取第一个 token 的输出             │
└─────────────────────────────────────────┘
    ↓ (B, 768)
┌─────────────────────────────────────────┐
│ 7. 分类头 (MLP Head)                    │
│    - Linear: 768 → num_classes          │
└─────────────────────────────────────────┘
    ↓ (B, num_classes)
【输出】类别概率
```

---

## 三、核心模块说明

### 模块 1: PatchEmbed
**作用**：把图像切成小块并投影

```
输入: (B, 3, 224, 224)
                ↓
【用卷积实现】Conv2d(3, 768, kernel_size=16, stride=16)
                ↓
        (B, 768, 14, 14)
                ↓
【展平 + 转置】flatten + transpose
                ↓
输出: (B, 196, 768)
```

**为什么用卷积？**
- 论文说"linear projection of flattened patches"
- 等价于：先切 16×16 块，再对每块做线性变换
- 卷积更高效！

---

### 模块 2: Multi-Head Self-Attention
**核心公式**：
```
Attention(Q, K, V) = softmax(QK^T / √d) V
```

**计算流程**：
```python
# 输入 X: (B, 197, 768)

# 1. 生成 Q, K, V
Q = X @ W_q  # (B, 197, 768)
K = X @ W_k  # (B, 197, 768)
V = X @ W_v  # (B, 197, 768)

# 2. 拆分成 12 个头
Q → (B, 12, 197, 64)  # 768/12 = 64
K → (B, 12, 197, 64)
V → (B, 12, 197, 64)

# 3. 计算注意力分数
scores = Q @ K^T / √64  # (B, 12, 197, 197)
attn = softmax(scores)

# 4. 加权求和
out = attn @ V  # (B, 12, 197, 64)

# 5. 合并多头
out → (B, 197, 768)

# 6. 输出投影
out = out @ W_o  # (B, 197, 768)
```

**为什么要多头？**
- 每个头学习不同的模式
- 头1可能关注边缘，头2关注纹理，头3关注全局结构
- 类似 CNN 的多通道卷积核

---

### 模块 3: MLP (Feed-Forward Network)
```
输入: (B, 197, 768)
    ↓
Linear(768 → 3072)    # 扩大 4 倍
    ↓
GELU 激活
    ↓
Dropout
    ↓
Linear(3072 → 768)    # 压缩回原维度
    ↓
Dropout
    ↓
输出: (B, 197, 768)
```

**GELU vs ReLU**：
- ReLU: `max(0, x)` - 硬截断
- GELU: `x * Φ(x)` - 平滑版本，Transformer 标配

---

### 模块 4: Transformer Block
```python
# 一个完整的 Block

x = x + Attention(LayerNorm(x))  # 注意力子层
x = x + MLP(LayerNorm(x))        # 前馈子层
```

**重点**：
- **Pre-Norm**：LayerNorm 在子层之前（ViT 用这个）
- **Post-Norm**：LayerNorm 在残差之后（原始 Transformer）
- Pre-Norm 训练更稳定！

---

## 四、关键参数对照表

### ViT-Base/16（默认配置）
| 参数名 | 值 | 说明 |
|--------|-----|------|
| `img_size` | 224 | 输入图像大小 |
| `patch_size` | 16 | Patch 大小 |
| `num_patches` | 196 | (224/16)² = 196 |
| `embed_dim` | 768 | Token 嵌入维度 |
| `depth` | 12 | Transformer 层数 |
| `num_heads` | 12 | 注意力头数 |
| `head_dim` | 64 | 每个头的维度 (768/12) |
| `mlp_ratio` | 4 | MLP 隐藏层扩展倍数 |
| `mlp_hidden` | 3072 | MLP 隐藏层维度 (768×4) |
| **总参数量** | **86M** | |

### 其他变体
| 模型 | embed_dim | depth | num_heads | 参数量 |
|------|-----------|-------|-----------|--------|
| ViT-Base | 768 | 12 | 12 | 86M |
| ViT-Large | 1024 | 24 | 16 | 307M |
| ViT-Huge | 1280 | 32 | 16 | 632M |

---

## 五、与 CNN 的对比

| 特性 | CNN | ViT |
|------|-----|-----|
| **归纳偏置** | 强（局部性、平移不变性） | 弱 |
| **数据需求** | 小数据集也能训练 | 需要大规模预训练 |
| **计算复杂度** | O(HWC²k²) | O(N²d) - 注意力是平方 |
| **全局视野** | 需要堆叠多层 | 第一层就能看到全局 |
| **可解释性** | 看卷积核 | 看注意力图 |

**论文核心发现**：
> 在大规模数据上预训练时，学习到的表征胜过归纳偏置！

---

## 六、学习路线

### 第一步：理解架构（今天）
- [x] 阅读论文前 7 页（已完成）
- [x] 理解整体架构（已完成）
- [ ] 看懂每个模块的数学原理
- [ ] 运行代码，看每层的输出形状

### 第二步：手写代码（明天）
- [ ] 手写 PatchEmbed
- [ ] 手写 Attention
- [ ] 手写 MLP
- [ ] 手写 Block
- [ ] 手写 VisionTransformer

### 第三步：实验验证（后天）
- [ ] 可视化 Attention Map
- [ ] 在 CIFAR-10 上训练
- [ ] 对比不同配置的效果

### 第四步：记录到 Notion
- [ ] 添加 ViT 论文到 Papers Database
- [ ] 创建论文笔记（用模板）
- [ ] 记录实验结果到 Experiments Database

---

## 七、常见问题

### Q1: 为什么要加 [CLS] token？
**A**: 借鉴 BERT 的设计。因为：
- 所有 patch tokens 会通过 self-attention 交互信息
- CLS token 能聚合全局信息
- 最后只用 CLS token 做分类

### Q2: Position Embedding 为什么是可学习的？
**A**:
- ViT 用的是**绝对位置编码**（和 BERT 一样）
- Transformer 原文用的是**正弦位置编码**（sin/cos 函数）
- 论文实验发现两者效果差不多，可学习的更简单

### Q3: 为什么小数据集上 ViT 不如 CNN？
**A**:
- CNN 有**归纳偏置**：局部性、平移不变性
- ViT 几乎没有归纳偏置，需要从数据中学习
- 小数据集信息不够，学不到好的表征
- 解决方案：在 ImageNet-21K (14M 图像) 上预训练

### Q4: 注意力的平方复杂度怎么办？
**A**:
- ViT 的序列长度固定 (196)，比 NLP 的几千 tokens 小得多
- 论文后续工作提出了很多优化：
  - Swin Transformer: 局部窗口注意力
  - PVT: 金字塔结构 + 空间缩减注意力
  - Twins: 局部 + 全局注意力混合

---

## 八、代码文件说明

### 1. `vit_architecture.md`
- 详细的数学推导
- 每一步的维度变化
- 图解说明

### 2. `vit_model.py`
- 完整实现代码（带详细注释）
- 5 个核心类：PatchEmbed、Attention、MLP、Block、VisionTransformer
- 3 个便捷函数：vit_base、vit_large、vit_huge

### 3. `test_model.py`
- 测试每个模块的输出形状
- 打印模型结构
- 计算参数量

### 4. `visualize_attention.py`
- 复现论文 Figure 7
- 可视化不同层、不同头的注意力图
- 分析模型在看什么

### 5. `train_cifar10.py`
- 在 CIFAR-10 (32×32 图像) 上训练
- 使用 Wandb 离线模式记录训练曲线
- 验证实现的正确性

---

## 九、重要公式速查

### Patch Embedding
```
x ∈ ℝ^(H×W×C)  →  x_p ∈ ℝ^(N×D)
N = HW/P²  (patch 数量)
D = 768    (嵌入维度)
```

### Self-Attention
```
Q = XW_q,  K = XW_k,  V = XW_v
Attention(Q,K,V) = softmax(QK^T / √d_k) V
```

### Multi-Head Attention
```
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
```

### Transformer Block
```
x' = x + MSA(LN(x))
x'' = x' + MLP(LN(x'))
```

### 完整前向传播
```
z_0 = [x_class; x_p^1 E; x_p^2 E; ...; x_p^N E] + E_pos
z_l = MSA(LN(z_{l-1})) + z_{l-1},  l=1...L
z_l = MLP(LN(z_l)) + z_l,          l=1...L
y = LN(z_L^0)
```

---

## 十、参考资料

### 论文
- 原文：An Image is Worth 16x16 Words (ICLR 2021)
- Attention Is All You Need (NeurIPS 2017) - Transformer 原文
- BERT: Pre-training of Deep Bidirectional Transformers (NAACL 2019)

### 代码
- 官方 JAX 实现：https://github.com/google-research/vision_transformer
- PyTorch 实现 (timm)：https://github.com/huggingface/pytorch-image-models

### 博客
- Jay Alammar 的可视化教程
- The Illustrated Transformer

---

**开始时间**：2025-12-07
**预计完成**：3 天
**当前进度**：理解架构阶段 ✅

让我们开始吧！🚀
