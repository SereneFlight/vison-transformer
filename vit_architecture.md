# ViT 架构详解：数学原理 + 维度变化

> 这个文档会**非常详细**地解释每一步的数学原理和维度变化
> 适合代码能力还在提升的同学仔细阅读！

---

## 目录
1. [整体流程](#1-整体流程)
2. [Patch Embedding 详解](#2-patch-embedding-详解)
3. [Multi-Head Attention 详解](#3-multi-head-attention-详解)
4. [MLP 详解](#4-mlp-详解)
5. [Transformer Block 详解](#5-transformer-block-详解)
6. [完整模型详解](#6-完整模型详解)
7. [训练细节](#7-训练细节)

---

## 1. 整体流程

### 1.1 输入输出
```
输入：RGB 图像
- 形状：(Batch, 3, 224, 224)
- 数值范围：[0, 1] 或 [-1, 1]（归一化后）

输出：类别概率
- 形状：(Batch, num_classes)
- 例如 ImageNet-1K：(Batch, 1000)
```

### 1.2 数据流

```
(B, 3, 224, 224)                    输入图像
    ↓
(B, 196, 768)                       Patch Embedding
    ↓
(B, 197, 768)                       加 CLS token
    ↓
(B, 197, 768)                       加位置编码
    ↓
(B, 197, 768) → Block 1 → ...       Transformer Encoder
    ↓
(B, 197, 768)                       12层后
    ↓
(B, 768)                            取 CLS token
    ↓
(B, num_classes)                    分类头
```

---

## 2. Patch Embedding 详解

### 2.1 原理

**核心思想**：把图像当作一个"句子"，每个 patch 是一个"单词"

```
原始图像: 224×224×3
    ↓
切分成 patches: (224/16) × (224/16) = 14×14 = 196 个
每个 patch: 16×16×3 = 768 个像素值
    ↓
展平每个 patch: [768]
    ↓
线性投影: [768] → [768]  (可学习的权重矩阵)
```

### 2.2 为什么用卷积实现？

**数学等价性证明**：

方法1（原始描述）：
```python
# 1. 切分 patch
patches = []
for i in range(14):
    for j in range(14):
        patch = img[:, :, i*16:(i+1)*16, j*16:(j+1)*16]  # (B, 3, 16, 16)
        patch = patch.reshape(B, -1)  # (B, 768)
        patches.append(patch)
patches = torch.stack(patches, dim=1)  # (B, 196, 768)

# 2. 线性投影
output = patches @ W  # W: (768, 768)
```

方法2（卷积实现）：
```python
# 一步到位！
output = Conv2d(3, 768, kernel_size=16, stride=16)(img)
output = output.flatten(2).transpose(1, 2)
```

**为什么等价？**
- 卷积的 kernel_size=16, stride=16 → 正好不重叠地扫过整个图像
- 每个卷积核的输出 → 对应一个 patch 的线性投影
- 计算效率更高，GPU 友好！

### 2.3 详细维度变化

```python
输入: x
形状: (B, 3, 224, 224)
    ↓
【卷积】self.proj = Conv2d(3, 768, kernel_size=16, stride=16)
    参数量: 3 × 768 × 16 × 16 = 589,824
    ↓
输出: (B, 768, 14, 14)
    解释: 768 个通道，14×14 的空间位置
    ↓
【展平】x.flatten(2)
    flatten 从第2维开始（0=B, 1=C, 2=H）
    ↓
输出: (B, 768, 196)
    解释: 196 = 14×14
    ↓
【转置】x.transpose(1, 2)
    交换维度 1 和 2
    ↓
最终输出: (B, 196, 768)
    解释: 196 个 tokens，每个 768 维
```

### 2.4 代码实现

```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2  # 196

        # 用卷积实现 patch embedding
        self.proj = nn.Conv2d(
            in_chans,      # 3
            embed_dim,     # 768
            kernel_size=patch_size,  # 16
            stride=patch_size        # 16
        )

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.proj(x)            # (B, 768, 14, 14)
        x = x.flatten(2)            # (B, 768, 196)
        x = x.transpose(1, 2)       # (B, 196, 768)
        return x
```

---

## 3. Multi-Head Attention 详解

### 3.1 Self-Attention 原理

**目标**：让每个 token 关注其他所有 token

**三个关键矩阵**：
- **Q (Query)**：我要查询什么信息？
- **K (Key)**：我有什么信息？
- **V (Value)**：我的信息内容是什么？

**类比搜索引擎**：
```
你输入 "Vision Transformer 论文"  ← Query
搜索引擎的索引库                ← Key
返回的论文内容                  ← Value

匹配度 = Query · Key
结果 = Σ (匹配度 × Value)
```

### 3.2 公式推导

**Step 1: 计算 Q, K, V**
```
输入: X ∈ ℝ^(N×D)  # N=197 tokens, D=768 维度

Q = XW_q,  W_q ∈ ℝ^(D×D)  →  Q ∈ ℝ^(N×D)
K = XW_k,  W_k ∈ ℝ^(D×D)  →  K ∈ ℝ^(N×D)
V = XW_v,  W_v ∈ ℝ^(D×D)  →  V ∈ ℝ^(N×D)
```

**Step 2: 计算注意力分数**
```
Scores = QK^T / √d_k

维度分析:
Q: (N, D) × K^T: (D, N) = (N, N)

每个元素 scores[i, j] 表示:
"token i 对 token j 的关注程度"
```

**为什么除以 √d_k？**
```
QK^T 的值会随着维度增大而增大
例如: d=768 时，点积可能达到几百
softmax(大数) → 梯度消失！

除以 √768 ≈ 27.7 → 缩放到合理范围
```

**Step 3: Softmax 归一化**
```
Attention = softmax(Scores, dim=-1)

每一行是一个概率分布:
Σ_j attention[i, j] = 1

attention[i, j] = "token i 给 token j 的权重"
```

**Step 4: 加权求和**
```
Output = Attention × V

维度: (N, N) × (N, D) = (N, D)

output[i] = Σ_j attention[i, j] * v[j]
         = "把所有 token 的 value 按注意力权重加权平均"
```

### 3.3 Multi-Head 的意义

**单头的局限**：
- 只能学习一种模式
- 比如只关注空间位置，忽略语义信息

**多头的优势**：
- 12 个头 = 12 种不同的关注模式
- 头1: 关注边缘
- 头2: 关注纹理
- 头3: 关注颜色
- ...

**实现方式**：
```
原始维度: D = 768
头数: h = 12
每个头的维度: d_h = D/h = 64

对于每个头 i:
Q_i = X W_q^i,  W_q^i ∈ ℝ^(768×64)  →  Q_i ∈ ℝ^(N×64)
K_i = X W_k^i,  W_k^i ∈ ℝ^(768×64)  →  K_i ∈ ℝ^(N×64)
V_i = X W_v^i,  W_v^i ∈ ℝ^(768×64)  →  V_i ∈ ℝ^(N×64)

head_i = Attention(Q_i, K_i, V_i)  →  ℝ^(N×64)

最后拼接:
Output = Concat(head_1, ..., head_12)  →  ℝ^(N×768)
```

### 3.4 详细维度变化

```python
输入: x
形状: (B, 197, 768)
    ↓
【生成 QKV】self.qkv = Linear(768, 768*3)
    ↓
qkv: (B, 197, 2304)  # 2304 = 768*3
    ↓
【重塑】qkv.reshape(B, N, 3, num_heads, head_dim)
    ↓
qkv: (B, 197, 3, 12, 64)
    ↓
【转置】qkv.permute(2, 0, 3, 1, 4)
    ↓
qkv: (3, B, 12, 197, 64)
    ↓
【分离】q, k, v = qkv[0], qkv[1], qkv[2]
    ↓
q: (B, 12, 197, 64)
k: (B, 12, 197, 64)
v: (B, 12, 197, 64)
    ↓
【计算注意力】attn = (q @ k.T) * scale
    q @ k.T: (B, 12, 197, 64) × (B, 12, 64, 197)
    ↓
attn: (B, 12, 197, 197)  # 注意力矩阵
    ↓
【softmax】attn = softmax(attn, dim=-1)
    ↓
【加权求和】x = attn @ v
    (B, 12, 197, 197) × (B, 12, 197, 64)
    ↓
x: (B, 12, 197, 64)
    ↓
【转置】x.transpose(1, 2)
    ↓
x: (B, 197, 12, 64)
    ↓
【合并多头】x.reshape(B, N, -1)
    ↓
x: (B, 197, 768)
    ↓
【输出投影】self.proj(x)
    ↓
最终输出: (B, 197, 768)
```

### 3.5 代码实现

```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 768 / 12 = 64
        self.scale = self.head_dim ** -0.5  # 1/√64

        # 一次性生成 Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # 输出投影
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # (2, 197, 768)

        # 生成 QKV
        qkv = self.qkv(x)  # (B, 197, 2304)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, 12, 197, 64)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, 12, 197, 197)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 加权求和
        x = attn @ v  # (B, 12, 197, 64)

        # 合并多头
        x = x.transpose(1, 2)  # (B, 197, 12, 64)
        x = x.reshape(B, N, C)  # (B, 197, 768)

        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
```

---

## 4. MLP 详解

### 4.1 原理

**作用**：对每个 token 独立地做非线性变换

**为什么需要？**
- Attention 是线性操作（加权平均）
- 需要 MLP 引入非线性，增强表达能力

### 4.2 结构

```
输入: (B, 197, 768)
    ↓
Linear(768 → 3072)      # 扩大 4 倍
    ↓
GELU 激活
    ↓
Dropout(0.1)
    ↓
Linear(3072 → 768)      # 压缩回来
    ↓
Dropout(0.1)
    ↓
输出: (B, 197, 768)
```

### 4.3 GELU 激活函数

**公式**：
```
GELU(x) = x · Φ(x)
Φ(x) = 标准正态分布的累积分布函数

近似:
GELU(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
```

**对比 ReLU**：
```
ReLU(x) = max(0, x)
    ↑ 硬截断，x<0 时梯度为 0

GELU(x) = x · Φ(x)
    ↑ 平滑版本，所有地方可导
    ↑ Transformer 标配！
```

### 4.4 代码实现

```python
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)  # 768 → 3072
        self.act = act_layer()  # GELU
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)  # 3072 → 768
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
```

---

## 5. Transformer Block 详解

### 5.1 Pre-Norm vs Post-Norm

**Post-Norm（原始 Transformer）**：
```
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + MLP(x))
```

**Pre-Norm（ViT 使用）**：
```
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

**区别**：
- Pre-Norm: 先归一化再做子层操作
- Post-Norm: 先做子层操作再归一化

**为什么 ViT 用 Pre-Norm？**
- 训练更稳定（梯度流更顺畅）
- 可以不用学习率 warmup
- 深层网络（如 ViT-Huge 32层）也能训练

### 5.2 残差连接的作用

```
x_out = x_in + F(x_in)
```

**好处**：
1. **梯度流畅**：反向传播时梯度可以直接回传
2. **恒等映射**：至少可以学到 F(x)=0，不会退化
3. **深层网络**：可以堆叠更多层

### 5.3 完整流程

```python
输入: x (B, 197, 768)

# 第一个子层：Self-Attention
norm_x = LayerNorm(x)           # (B, 197, 768)
attn_out = Attention(norm_x)    # (B, 197, 768)
x = x + attn_out                # 残差连接

# 第二个子层：MLP
norm_x = LayerNorm(x)           # (B, 197, 768)
mlp_out = MLP(norm_x)           # (B, 197, 768)
x = x + mlp_out                 # 残差连接

输出: x (B, 197, 768)
```

### 5.4 代码实现

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # 768 * 4 = 3072
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        # 注意力子层
        x = x + self.attn(self.norm1(x))

        # MLP 子层
        x = x + self.mlp(self.norm2(x))

        return x
```

---

## 6. 完整模型详解

### 6.1 CLS Token

**为什么需要？**
- 借鉴 BERT 的 [CLS] token
- 所有 patch tokens 通过 self-attention 交互
- CLS token 聚合全局信息
- 最后只用 CLS token 做分类

**实现**：
```python
# 可学习参数
self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))

# forward 时：
cls_token = self.cls_token.expand(B, -1, -1)  # (1,1,768) → (B,1,768)
x = torch.cat([cls_token, x], dim=1)  # (B, 196, 768) → (B, 197, 768)
```

### 6.2 Position Embedding

**为什么需要？**
- Self-Attention 是**置换不变**的
- 打乱 token 顺序，输出不变！
- 需要位置编码告诉模型"你是第几个 patch"

**ViT 的选择**：可学习的绝对位置编码
```python
self.pos_embed = nn.Parameter(torch.zeros(1, 197, 768))

# forward 时：
x = x + self.pos_embed  # 广播加法
```

**其他选择**：
- 正弦位置编码（原始 Transformer）
- 相对位置编码（T5, DeBERTa）
- 旋转位置编码（RoFormer）

### 6.3 完整前向传播

```python
def forward(self, x):
    B = x.shape[0]

    # 1. Patch Embedding
    x = self.patch_embed(x)  # (B,3,224,224) → (B,196,768)

    # 2. 加 CLS token
    cls_token = self.cls_token.expand(B, -1, -1)  # (B,1,768)
    x = torch.cat([cls_token, x], dim=1)  # (B,197,768)

    # 3. 加位置编码
    x = x + self.pos_embed  # (B,197,768)
    x = self.pos_drop(x)

    # 4. 通过 12 层 Transformer
    for block in self.blocks:
        x = block(x)  # (B,197,768) → (B,197,768)

    # 5. 最后的 Layer Norm
    x = self.norm(x)  # (B,197,768)

    # 6. 只取 CLS token
    cls_output = x[:, 0]  # (B,768)

    # 7. 分类头
    logits = self.head(cls_output)  # (B,768) → (B,num_classes)

    return logits
```

### 6.4 参数量计算

**ViT-Base/16**：
```
1. Patch Embedding:
   - Conv2d: 3×768×16×16 = 589,824

2. CLS token:
   - 1×1×768 = 768

3. Position Embedding:
   - 1×197×768 = 151,296

4. Transformer Block × 12:
   每个 Block:
   - Attention:
     * QKV: 768×(768×3) = 1,769,472
     * Proj: 768×768 = 589,824
   - MLP:
     * fc1: 768×3072 = 2,359,296
     * fc2: 3072×768 = 2,359,296
   - LayerNorm × 2: 忽略不计

   单个 Block: ~7M 参数
   12 个 Block: ~84M 参数

5. 最后的 LayerNorm: 768

6. 分类头:
   - Linear: 768×1000 = 768,000

总计: ~86M 参数
```

---

## 7. 训练细节

### 7.1 预训练策略

**数据集**：
- 小数据：ImageNet-1K (1.2M 图像)
- 中数据：ImageNet-21K (14M 图像)
- 大数据：JFT-300M (300M 图像)

**论文发现**：
```
ImageNet-1K 预训练:
  ViT-Base < ResNet-50  ❌

ImageNet-21K 预训练:
  ViT-Base ≈ ResNet-101  ✓

JFT-300M 预训练:
  ViT-Base > ResNet-152  ✓✓
```

### 7.2 超参数

```python
# 优化器
optimizer = Adam(lr=0.001, betas=(0.9, 0.999), weight_decay=0.1)

# 学习率调度
# 1. Warmup: 0 → 0.001 (10k steps)
# 2. Cosine decay: 0.001 → 0 (剩余 steps)

# 正则化
dropout = 0.1
stochastic_depth = 0.1  # 随机丢弃整个 Block

# 数据增强
- RandAugment
- Mixup
- Cutmix
- Random Erasing
```

### 7.3 Fine-tuning 细节

```
预训练模型: 在 ImageNet-21K (14M 图像) 上
    ↓
Fine-tune: 在下游任务上
    - 分辨率: 224 → 384 (可选)
    - 学习率: 0.001 → 0.003
    - Batch size: 512
    - Epochs: ~20
```

**位置编码的处理**：
- 预训练: 224×224 → 14×14 = 196 patches
- Fine-tune: 384×384 → 24×24 = 576 patches
- 解决方案: 2D 插值 (bicubic)

---

## 8. 重要图表

### 8.1 Attention Map 示例

```
论文 Figure 7:

第1层: 关注局部邻域（类似 CNN）
第6层: 开始关注全局结构
第12层: 聚焦到目标物体

不同的头学到不同的模式:
- 头1: 边缘检测
- 头2: 纹理模式
- 头3: 全局形状
```

### 8.2 性能对比

```
ImageNet-1K (从头训练):
  ResNet-50:     76.5%
  ViT-Base:      77.9%  ← 略好

ImageNet-21K 预训练 → ImageNet-1K:
  ResNet-152:    78.3%
  ViT-Base:      81.8%  ← 显著提升

JFT-300M 预训练 → ImageNet-1K:
  ResNet-152:    79.8%
  ViT-Huge:      88.5%  ← 碾压式领先
```

---

## 9. 常见错误和调试技巧

### 9.1 维度不匹配

```python
# 错误示例
x = x.reshape(B, N, C)  # 忘记考虑 num_heads

# 正确做法
x = x.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
```

### 9.2 注意力分数溢出

```python
# 问题: attn = softmax(q @ k.T) 梯度消失

# 解决: 缩放
attn = softmax((q @ k.T) / sqrt(d_k))
```

### 9.3 调试建议

```python
# 在每个模块后打印形状
print(f"After patch_embed: {x.shape}")
print(f"After add cls: {x.shape}")
print(f"After block 0: {x.shape}")
...

# 检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

---

**这份文档应该能帮你完全理解 ViT 的每一个细节！**

有任何不懂的地方，随时问我 😊
