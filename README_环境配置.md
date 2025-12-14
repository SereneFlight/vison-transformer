# ViT 项目环境配置说明

## 📦 虚拟环境信息

- **环境名称**： `vla_learning`
- **Python 版本**: `3.11.14`
- **PyTorch 版本**: `2.9.1+cu128`
- **CUDA 版本**: `12.8`
- **GPU**: NVIDIA GeForce RTX 5070 Laptop GPU

## 🚀 快速启动

### 1. 激活虚拟环境

```bash
conda activate vla_learning
```

### 2. 在 VSCode 中使用

1. 打开项目文件夹： `/home/yj/桌面/vla_learning/vit_from_scratch`
2. 按 `Ctrl + Shift + P`
3. 输入 `Python: Select Interpreter`
4. 选择 `/home/yj/anaconda3/envs/vla_learning/bin/python`

### 3. 重新加载 VSCode

按 `Ctrl + Shift + P` → 输入 `Reload Window` → 回车

## 📚 已安装的包

### 核心包
- `torch==2.9.1+cu128` - PyTorch 深度学习框架
- `torchvision==0.24.1+cu128` - 计算机视觉工具
- `torchaudio==2.9.1+cu128` - 音频处理工具
- `numpy==2.3.5` - 数值计算
- `pillow==12.0.0` - 图像处理

### 可视化和开发工具
- `matplotlib` - 绘图库
- `tqdm` - 进度条
- `jupyter` - Jupyter Notebook
- `ipython` - 交互式 Python

### NVIDIA CUDA 库
- 所有必要的 CUDA 库（cudnn, cublas, cusparse 等）

## 🧪 测试环境

运行测试脚本验证环境：

```bash
cd /home/yj/桌面/vla_learning/vit_from_scratch
conda activate vla_learning
python test_environment.py
```

## 📋 VSCode 配置

项目已配置以下文件：

### `.vscode/settings.json`
- Python 解释器自动指向 `vla_learning` 环境
- 启用 Pylance 语言服务器
- 启用代码提示和类型检查
- 自动保存设置

### `.vscode/launch.json`
- 配置了调试器
- 可以直接按 F5 调试 Python 文件

### `.vscode/extensions.json`
- 推荐安装的 VSCode 扩展
- 包括 Python, Pylance, Jupyter 等

## 🔧 常见问题

### Q1: torch 标红？
**解决方法：**
1. 确保 VSCode 选择了正确的解释器（右下角）
2. 重启 Pylance: `Ctrl+Shift+P` → `Pylance: Restart Server`
3. 重新加载窗口: `Ctrl+Shift+P` → `Reload Window`

### Q2: 没有代码提示？
**解决方法：**
1. 检查 Pylance 是否安装: `Ctrl+Shift+P` → `Extensions: Show Installed Extensions`
2. 确认 Python 解释器路径正确
3. 尝试关闭并重新打开项目文件夹

### Q3: 想添加新的包？
**方法：**
```bash
conda activate vla_learning
pip install <package-name>
```

## 📂 项目结构

```
vit_from_scratch/
├── my_vit_model.py          # 手写的 ViT 模型
├── test_environment.py      # 环境测试脚本
├── README_环境配置.md        # 本文件
└── .vscode/                 # VSCode 配置
    ├── settings.json
    ├── launch.json
    └── extensions.json
```

## 🎯 下一步

环境配置完成后，你可以：

1. ✅ **测试模型** - 运行 `test_environment.py`
2. **训练 ViT** - 在 CIFAR-10 上训练
3. **可视化 Attention Map** - 查看模型关注哪些区域
4. **学习 VLA** - 为研究做准备

## 💡 提示

- 每次打开新终端都需要 `conda activate vla_learning`
- VSCode 会自动使用配置的 Python 解释器
- 确保 GPU 正常工作：`python -c "import torch; print(torch.cuda.is_available())"`

---

**配置完成时间**: 2025-12-14
**配置者**: Claude Code
