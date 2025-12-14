# ViT 可视化完整指南

## 📚 可视化类型总览

我为你准备了 **7 种不同的可视化**，涵盖了 ViT 论文中最经典的图像！

### 🎨 可视化列表

| 类型 | 文件 | 说明 | 论文中常见度 |
|-----|------|------|------------|
| **1. Patch Grid** | `visualize_attention.py` | 图像如何被切成 16×16 的 patches | ⭐⭐⭐ |
| **2. Attention Map** | `visualize_attention.py` | 模型关注图像的哪些区域（最经典！） | ⭐⭐⭐⭐⭐ |
| **3. All Attention Heads** | `visualize_attention.py` | 不同 head 学到的不同模式 | ⭐⭐⭐⭐ |
| **4. Position Embedding** | `visualize_advanced.py` | 位置编码的相似度 | ⭐⭐⭐ |
| **5. Attention Distance** | `visualize_advanced.py` | 浅层 vs 深层（Local vs Global） | ⭐⭐⭐⭐ |
| **6. CLS Token Evolution** | `visualize_advanced.py` | CLS token 在每层的变化 | ⭐⭐⭐ |
| **7. Attention Rollout** | `visualize_advanced.py` | 累积所有层的 attention | ⭐⭐⭐⭐ |

---

## 🚀 快速开始

### 方法1：运行完整示例（推荐）

```bash
# 激活环境
conda activate vla_learning

# 运行完整示例（会生成所有 7 种可视化）
python run_visualization_demo.py
```

**这个脚本会：**
1. 自动从 CIFAR-10 下载一张示例图像
2. 生成所有 7 种可视化
3. 保存为 `vis_1_*.png` 到 `vis_7_*.png`

**运行时间：** 约 2-3 分钟

---

### 方法2：自定义可视化

如果你想用**自己的图像**，可以这样：

```python
from visualize_attention import ViTWithAttention, visualize_attention_map

# 1. 创建模型
model = ViTWithAttention(
    img_size=224,
    patch_size=16,
    num_classes=10,
    embed_dim=768,
    depth=12,
    num_heads=12
)

# 2. 可视化你的图像
visualize_attention_map(
    model,
    'your_image.jpg',  # 替换成你的图像路径
    save_path='my_attention.png',
    layer_indices=[0, 3, 6, 11],  # 可视化哪几层
    head_index=0  # 可视化第几个 head
)
```

---

## 📊 各种可视化详解

### 1. **Patch Grid** - 图像切分

**效果：** 显示图像如何被切成 14×14 = 196 个 patches

```python
from visualize_attention import visualize_patch_grid

visualize_patch_grid('your_image.jpg', save_path='patch_grid.png')
```

**说明：**
- 每个 patch 是 16×16 像素
- 红色网格显示切分位置
- 数字表示 patch 的索引（0-195）

---

### 2. **Attention Map** ⭐ 最经典！

**效果：** 显示模型"看"图像的哪些部分（论文中最常见的图）

```python
from visualize_attention import ViTWithAttention, visualize_attention_map

model = ViTWithAttention(img_size=224, patch_size=16, num_classes=10,
                         embed_dim=768, depth=12, num_heads=12)

visualize_attention_map(
    model,
    'image.jpg',
    save_path='attention_map.png',
    layer_indices=[0, 2, 5, 11],  # 第0, 2, 5, 11层
    head_index=0  # 第0个 attention head
)
```

**参数说明：**
- `layer_indices`: 选择哪几层进行可视化
  - 浅层（0-2）：通常关注局部细节
  - 深层（9-11）：通常关注全局语义
- `head_index`: 选择哪个 attention head（0-11）

**解读：**
- 红色/黄色区域 = 高 attention（模型重点关注）
- 蓝色/紫色区域 = 低 attention（模型较少关注）

---

### 3. **All Attention Heads** - 不同头的模式

**效果：** 显示某一层所有 12 个 heads 的 attention

```python
from visualize_attention import visualize_all_heads

visualize_all_heads(
    model,
    'image.jpg',
    save_path='all_heads.png',
    layer_idx=5  # 可视化第5层
)
```

**说明：**
- 不同 head 学到不同的模式
- 有的 head 关注边缘，有的关注纹理，有的关注整体

---

### 4. **Position Embedding** - 位置编码相似度

**效果：** 显示哪些位置在模型眼中是"相近"的

```python
from visualize_advanced import visualize_position_embedding

visualize_position_embedding(model, save_path='pos_embed.png')
```

**3 个子图：**
1. **相似度矩阵** - 所有 patches 之间的相似度
2. **中心 patch** - 中心 patch 与其他位置的相似度
3. **PCA 2D** - 降维后的空间分布

**解读：**
- 相邻 patches 通常相似度高（颜色亮）
- 距离远的 patches 相似度低（颜色暗）

---

### 5. **Attention Distance** ⭐ Local vs Global

**效果：** 显示每一层平均关注多远的 patches

```python
from visualize_advanced import visualize_attention_distance

# 需要先运行模型获取 attention_maps
# ... (运行模型代码)

visualize_attention_distance(
    model.attention_maps,
    save_path='attention_distance.png'
)
```

**2 个子图：**
1. **每层的平均距离** - 曲线图
2. **浅层 vs 深层对比** - 柱状图

**论文中的发现：**
- **浅层（Layer 0-2）**：关注邻近 patches（Local attention）
- **深层（Layer 10-12）**：关注全局（Global attention）

---

### 6. **CLS Token Evolution** - CLS token 的演变

**效果：** CLS token 在每一层的变化轨迹

```python
from visualize_advanced import visualize_cls_token_evolution

visualize_cls_token_evolution(
    model,
    img_tensor,
    save_path='cls_evolution.png'
)
```

**2 个子图：**
1. **PCA 轨迹** - CLS token 在特征空间的移动
2. **范数变化** - CLS token 向量的大小变化

**解读：**
- 绿色方块 = 输入层的 CLS token
- 红色星星 = 输出层的 CLS token（用于分类）
- 轨迹显示 CLS token 如何逐渐"聚合"信息

---

### 7. **Attention Rollout** ⭐ 累积 Attention

**效果：** 将所有层的 attention 累积起来

```python
from visualize_advanced import visualize_attention_rollout

visualize_attention_rollout(
    model.attention_maps,
    save_path='attention_rollout.png'
)
```

**说明：**
- 不是单独看某一层，而是看"从输入到输出"的完整路径
- 更能反映信息的流动

**论文中的用途：**
- 理解整个模型的关注模式
- 调试模型是否关注正确的区域

---

## 🎯 论文中最常见的图

如果你要复现论文中的图，推荐这几个：

### 图1: Attention Map（多层对比）

```python
visualize_attention_map(
    model, 'image.jpg',
    layer_indices=[0, 3, 6, 9, 11],
    head_index=0
)
```

### 图2: 所有 Attention Heads

```python
visualize_all_heads(model, 'image.jpg', layer_idx=5)
```

### 图3: Attention Distance（Local vs Global）

```python
visualize_attention_distance(attention_maps)
```

### 图4: Attention Rollout

```python
visualize_attention_rollout(attention_maps)
```

---

## 💡 使用技巧

### 技巧1: 选择合适的层

- **浅层（0-2）**: 看局部细节（纹理、边缘）
- **中层（4-7）**: 看中级特征（形状、部件）
- **深层（9-11）**: 看全局语义（整体对象）

### 技巧2: 选择合适的 Head

不同 head 学到不同模式，多试几个：
```python
# 可视化不同的 heads
for head_idx in [0, 3, 6, 9]:
    visualize_attention_map(model, 'image.jpg', head_index=head_idx)
```

### 技巧3: 使用有意义的图像

建议使用：
- **单一对象** - 容易看出关注区域
- **高分辨率** - 细节更清晰
- **清晰背景** - 减少干扰

---

## 🔍 高级用法

### 对比训练前后的 Attention

```python
# 训练前
model_before = ViTWithAttention(...)
visualize_attention_map(model_before, 'image.jpg', save_path='before.png')

# 训练后
model_after = ViTWithAttention(...)
model_after.load_state_dict(torch.load('trained_model.pth'))
visualize_attention_map(model_after, 'image.jpg', save_path='after.png')
```

### 批量可视化多张图像

```python
images = ['cat.jpg', 'dog.jpg', 'bird.jpg']
for img in images:
    visualize_attention_map(model, img, save_path=f'attn_{img}')
```

---

## 📝 常见问题

### Q1: 为什么我的 attention map 很模糊？

**答：** 这是正常的！因为：
1. Attention 是在 14×14 的 patch 网格上计算的
2. 我们插值到 224×224 显示，会有模糊
3. 可以增加 `interpolation='nearest'` 显示块状效果

### Q2: 不同 head 的 attention 为什么差别很大？

**答：** 这正是 Multi-Head Attention 的优势！
- 不同 head 学习不同的模式
- 有的 head 关注边缘，有的关注纹理，有的关注整体
- 这样模型可以从多个角度理解图像

### Q3: 浅层和深层的 attention 为什么不同？

**答：** 这反映了特征的层次性：
- **浅层**：关注低级特征（边缘、纹理）→ Local attention
- **深层**：关注高级语义（整体对象）→ Global attention

### Q4: 如何理解 Attention Rollout？

**答：** Rollout 累积了所有层的 attention：
- 单层 attention：只看一层的关注
- Rollout：看信息从输入到输出的完整路径
- 更能反映整个模型的行为

---

## 🎨 可视化示例效果

运行 `python run_visualization_demo.py` 后，你会得到：

```
vis_1_patch_grid.png          # Patch 切分网格
vis_2_position_embedding.png  # 位置编码相似度
vis_3_attention_layers.png    # 多层 Attention 对比
vis_4_all_heads.png           # 所有 heads 的 attention
vis_5_attention_distance.png  # Local vs Global 距离分析
vis_6_cls_evolution.png       # CLS token 演变轨迹
vis_7_attention_rollout.png   # 累积 Attention
```

---

## 📚 参考资料

- **ViT 论文**: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- **Attention Rollout**: "Quantifying Attention Flow in Transformers"

---

## 🆘 遇到问题？

1. 确保激活了环境：`conda activate vla_learning`
2. 确保安装了所有依赖：`pip install scipy scikit-learn seaborn`
3. 检查图像路径是否正确
4. 确保图像是 RGB 格式（不是 RGBA 或灰度图）

---

好好享受可视化的乐趣吧！🎉
