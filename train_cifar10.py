"""
在 CIFAR-10 上训练 ViT

CIFAR-10 数据集:
- 10 个类别: 飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车
- 训练集: 50,000 张 32×32 的彩色图像
- 测试集: 10,000 张

注意: CIFAR-10 的图像是 32×32，但 ViT 需要 224×224
      所以我们会 resize 到 224×224
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
import os
import wandb
from vit_model import vit_base_patch16_224


# ========================================
# 配置
# ========================================

class Config:
    # 数据
    data_dir = './data'           # 数据保存目录
    num_classes = 10              # CIFAR-10 有 10 个类别

    # 模型（使用小一点的 ViT，训练更快）
    img_size = 224
    patch_size = 16
    embed_dim = 384               # 比 ViT-Base (768) 小一半
    depth = 6                     # 比 ViT-Base (12) 少一半
    num_heads = 6
    mlp_ratio = 4

    # 训练
    batch_size = 64               # 根据显存调整
    num_epochs = 20               # 训练轮数
    learning_rate = 3e-4          # 学习率
    weight_decay = 0.1            # 权重衰减

    # 其他
    num_workers = 4               # 数据加载线程数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = './checkpoints'    # 模型保存目录

    # WandB
    wandb_project = 'vit-cifar10'  # WandB 项目名称
    wandb_name = None              # 运行名称（None 则自动生成）

config = Config()


# ========================================
# 数据准备
# ========================================

def get_dataloaders():
    """准备 CIFAR-10 数据加载器"""

    print("=" * 70)
    print("准备数据集")
    print("=" * 70)

    # 数据增强和预处理
    transform_train = transforms.Compose([
        transforms.Resize(config.img_size),           # 32×32 → 224×224
        transforms.RandomHorizontalFlip(),            # 随机水平翻转
        transforms.RandomCrop(config.img_size, padding=4),  # 随机裁剪
        transforms.ToTensor(),                        # 转成 tensor
        transforms.Normalize(                         # 归一化
            mean=[0.4914, 0.4822, 0.4465],           # CIFAR-10 的均值
            std=[0.2023, 0.1994, 0.2010]             # CIFAR-10 的标准差
        ),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        ),
    ])

    # 下载并加载数据集
    print(f"\n下载 CIFAR-10 数据集到: {config.data_dir}")
    print("(第一次运行会自动下载，大约 170MB)")

    train_dataset = torchvision.datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=config.data_dir,
        train=False,
        download=True,
        transform=transform_test
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )

    print(f"\n数据集信息:")
    print(f"  训练集: {len(train_dataset)} 张图像")
    print(f"  测试集: {len(test_dataset)} 张图像")
    print(f"  类别数: {config.num_classes}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  训练 batches: {len(train_loader)}")
    print(f"  测试 batches: {len(test_loader)}")

    # 类别名称
    classes = ['飞机', '汽车', '鸟', '猫', '鹿',
               '狗', '青蛙', '马', '船', '卡车']
    print(f"\n类别: {classes}")

    return train_loader, test_loader, classes


# ========================================
# 模型创建
# ========================================

def create_model():
    """创建 ViT 模型（小版本，适合快速训练）"""

    print("\n" + "=" * 70)
    print("创建模型")
    print("=" * 70)

    from vit_model import VisionTransformer

    model = VisionTransformer(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_chans=3,
        num_classes=config.num_classes,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1
    )

    model = model.to(config.device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n模型配置:")
    print(f"  图像大小: {config.img_size}×{config.img_size}")
    print(f"  Patch 大小: {config.patch_size}×{config.patch_size}")
    print(f"  Embedding 维度: {config.embed_dim}")
    print(f"  Transformer 层数: {config.depth}")
    print(f"  注意力头数: {config.num_heads}")
    print(f"\n参数量:")
    print(f"  总参数: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    return model


# ========================================
# 训练函数
# ========================================

def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """训练一个 epoch"""

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(config.device), targets.to(config.device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 打印进度
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            avg_loss = running_loss / (batch_idx + 1)
            acc = 100. * correct / total
            print(f'  Batch [{batch_idx+1}/{len(train_loader)}] '
                  f'Loss: {avg_loss:.4f} | Acc: {acc:.2f}%')

    epoch_time = time.time() - start_time
    avg_loss = running_loss / len(train_loader)
    acc = 100. * correct / total

    return avg_loss, acc, epoch_time


# ========================================
# 测试函数
# ========================================

def test(model, test_loader, criterion):
    """在测试集上评估"""

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = test_loss / len(test_loader)
    acc = 100. * correct / total

    return avg_loss, acc


# ========================================
# 主训练循环
# ========================================

def main():
    print("\n" + "=" * 70)
    print("ViT 在 CIFAR-10 上的训练")
    print("=" * 70)

    print(f"\n设备: {config.device}")
    if config.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 初始化 WandB
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_name,
        config={
            "learning_rate": config.learning_rate,
            "epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "weight_decay": config.weight_decay,
            "img_size": config.img_size,
            "patch_size": config.patch_size,
            "embed_dim": config.embed_dim,
            "depth": config.depth,
            "num_heads": config.num_heads,
            "mlp_ratio": config.mlp_ratio,
        }
    )
    print(f"\n✓ WandB 初始化完成: {wandb.run.name}")
    print(f"  项目: {config.wandb_project}")
    print(f"  链接: {wandb.run.url}")

    # 1. 准备数据
    train_loader, test_loader, classes = get_dataloaders()

    # 2. 创建模型
    model = create_model()

    # 3. 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs
    )

    # 4. 训练
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)

    best_acc = 0.0

    for epoch in range(config.num_epochs):
        print(f"\nEpoch [{epoch+1}/{config.num_epochs}]")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")

        # 训练
        train_loss, train_acc, epoch_time = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch
        )

        # 测试
        test_loss, test_acc = test(model, test_loader, criterion)

        # 更新学习率
        scheduler.step()

        # 记录到 WandB
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "test/loss": test_loss,
            "test/acc": test_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch_time": epoch_time
        })

        # 打印结果
        print(f"\n  训练: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  测试: Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
        print(f"  用时: {epoch_time:.2f}秒")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc

            # 创建保存目录
            os.makedirs(config.save_dir, exist_ok=True)

            # 保存模型
            save_path = os.path.join(config.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'config': config.__dict__
            }, save_path)

            print(f"  ✓ 保存最佳模型 (Acc={test_acc:.2f}%) 到: {save_path}")

            # 保存到 WandB
            wandb.save(save_path)

    # 5. 训练完成
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    print(f"\n最佳测试准确率: {best_acc:.2f}%")
    print(f"模型保存在: {config.save_dir}/best_model.pth")

    # 6. 测试最佳模型
    print("\n加载最佳模型进行最终测试...")
    checkpoint = torch.load(os.path.join(config.save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    final_loss, final_acc = test(model, test_loader, criterion)
    print(f"最终测试准确率: {final_acc:.2f}%")

    # 记录最终结果到 WandB
    wandb.log({
        "final/test_loss": final_loss,
        "final/test_acc": final_acc,
        "final/best_acc": best_acc
    })
    wandb.finish()
    print("\n✓ WandB 记录已完成")
    print(f"查看结果: {wandb.run.url}")


if __name__ == '__main__':
    main()
