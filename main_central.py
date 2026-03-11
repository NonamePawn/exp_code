import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path  # 专门用于处理路径的库，比 os.path 好用


# dataset.py: 负责把硬盘里的图片读进内存
from dataSet.dataset import get_dataloader
# partitioner.py: 负责划分数据（这里我们用它获取全量文件列表）
from dataSet.partitioner import FederatedPartitioner
# model_factory.py: 负责根据名字生产模型
from model.model_factory import get_model
# logger.py: 负责记录日志以及保存结果
from utils.logger import ExperimentLogger, save_analysis_data, print_box


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    [核心功能 1] 训练一个 Epoch (一轮)
    -------------------------------------------
    逻辑：
    1. 拿一批数据 (Batch)
    2. 正向传播算结果 (Forward)
    3. 算误差 (Loss)
    4. 反向传播算梯度 (Backward)
    5. 更新权重 (Step)
    """
    # 1. 开启训练模式
    # 这很重要！它会启用 Dropout 和 Batch Normalization 的参数更新。
    model.to(device)
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # tqdm 是进度条工具，让控制台输出好看点
    pbar = tqdm(loader, desc="[Train]", leave=False)

    for images, labels in pbar:
        # 把数据搬运到 GPU 上 (如果 device 是 cuda)
        images, labels = images.to(device), labels.to(device)

        # 2. 梯度清零
        # PyTorch 默认会累加梯度。如果不清零，这轮的梯度会和上轮混在一起，导致训练失败。
        optimizer.zero_grad()

        # 3. 正向传播 (Forward Pass)
        # model() 会调用模型的 forward 函数
        # 注意：这里我们只关心 logits (预测分数)，不关心 features
        output = model(images)

        # 兼容性处理：有的模型返回 (logits, features)，有的只返回 logits
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        # 4. 计算损失 (Loss)
        loss = criterion(logits, labels)

        # 5. 反向传播 (Backward Pass) -> 计算梯度
        loss.backward()

        # 6. 优化器更新参数 (Optimizer Step) -> 修改权重
        optimizer.step()

        # --- 统计数据 ---
        running_loss += loss.item()
        # torch.max 返回 (最大值, 索引)。索引就是预测的类别 ID
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新进度条上的文字
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # 返回这一轮的平均 Loss 和 准确率
    return running_loss / len(loader), 100 * correct / total


def validate(model, loader, criterion, device, collect_data=False):
    """
    [核心功能 2] 验证/测试 (Validation/Test)
    -------------------------------------------
    逻辑：只看不改。
    只计算 Loss 和 Acc，绝对不能更新模型参数。

    参数:
    - collect_data: 如果为 True，会把所有预测结果存下来（用于画图）。
    """
    # 1. 开启评估模式
    # 这会冻结 BatchNorm 和 Dropout，保证推理结果稳定。
    model.eval()

    running_loss = 0.0

    # 容器：用来装所有的预测结果
    all_preds = []
    all_labels = []
    all_features = []  # 用于画 t-SNE
    all_logits = []  # 用于画混淆矩阵

    # 2. 告诉 PyTorch 不要计算梯度
    # 省显存，速度快，因为我们不需要反向传播。
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="[Eval]", leave=False):
            images, labels = images.to(device), labels.to(device)

            # 正向传播
            output = model(images)

            # 拆解输出
            features = None
            if isinstance(output, tuple):
                logits = output[0]
                features = output[1]
            else:
                logits = output
                # 如果模型没返回特征，就拿 logits 充当特征（权宜之计）
                features = logits

                # 算 Loss 用于监控
            loss = criterion(logits, labels)
            running_loss += loss.item()

            _, predicted = torch.max(logits, 1)

            # --- 收集数据 ---
            # 必须先 .cpu() 搬回内存，再转 .numpy()，因为 GPU Tensor 不能直接转 numpy
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if collect_data:
                all_logits.extend(logits.cpu().numpy())
                if features is not None:
                    # 如果是 4D 张量 (B, C, H, W)，平均成 (B, C) 向量
                    if len(features.shape) > 2:
                        features = features.mean(dim=[2, 3])
                    all_features.extend(features.cpu().numpy())

    # 计算整体指标
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    acc = 100 * np.mean(all_preds == all_labels)
    avg_loss = running_loss / len(loader)

    metrics = {"loss": avg_loss, "acc": acc}

    # 如果是测试阶段，返回所有详细数据；如果是验证阶段，只返回指标
    if collect_data:
        return metrics, np.array(all_features), all_labels, all_preds, np.array(all_logits)
    else:
        return metrics


def main():
    # =================================================
    # [Step 1] 参数配置 (Configuration)
    # =================================================
    print(f"[1/5] 🏆 Reading Configuration...")
    parser = argparse.ArgumentParser(description="Central")
    # --model: 你在 model_factory 里注册的名字，或者 timm 的模型名
    parser.add_argument('--structure_name', type=str, default='central', help='架构名称')
    parser.add_argument('--data_root', type=str, default='/assets_processed', help='数据根目录')
    parser.add_argument('--model', type=str, default='resnet50', help='模型名称')
    parser.add_argument('--device', type=str, default='cuda', help='运算设备')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='学习率')
    parser.add_argument('--epochs', type=int, default=30)

    args = parser.parse_args()

    if torch.cuda.is_available() and args.device == 'cuda':
        DEVICE = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        # 获取显存 (GB)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        hw_info = f"🚀 GPU: {gpu_name} ({total_mem:.1f} GB)"
    else:
        DEVICE = torch.device("cpu")
        hw_info = "🐢 Device: CPU"

    # 初始化日志记录器 (它会自动创建上面的文件夹)
    logger = ExperimentLogger(save_dir=f"result/{args.structure_name}")

    # 打印配置表
    print_box(" Centralized Training Start", [
        f"Info: Structure={args.structure_name} | Model={args.model} | Data Root={args.data_root}",
        f"Settings: Batch Size={args.batch_size} | Learning Rate={args.learning_rate} | Epochs={args.epochs}",
        hw_info
    ])
    # =================================================
    # [Step 2] 数据准备 (Data Preparation)
    # =================================================
    print(f"\n[2/5] 📂 Preparing Data...")
    root_path = Path(args.data_root)
    train_path = root_path / 'train'
    val_path = root_path / 'val'
    test_path = root_path / 'test'

    # 利用 Partitioner 获取文件列表（并不读取图片，只读路径，所以很快）
    train_list = FederatedPartitioner(data_root=train_path, num_clients=1).get_all_data()
    val_list = FederatedPartitioner(data_root=val_path, num_clients=1).get_all_data()
    test_list = FederatedPartitioner(data_root=test_path, num_clients=1).get_all_data()

    # 自动获取类别数
    num_classes = len(FederatedPartitioner(data_root=train_path, num_clients=1).class_map)


    # 创建 DataLoader (真正负责在训练时多线程读图)
    train_loader = get_dataloader(train_list, batch_size=args.batch_size, is_train=True)
    val_loader = get_dataloader(val_list, batch_size=args.batch_size, is_train=False)
    test_loader = get_dataloader(test_list, batch_size=args.batch_size, is_train=False)

    # 打印汇总表
    print_box("DATASET SUMMARY", [
        f"Classes:       {num_classes}",
        f"Train Data: {len(train_list)} samples",
        f"Val Data:    {len(val_list)} samples",
        f"Test Data:     {len(test_list)} samples"
    ])
    # =================================================
    # [Step 3] 模型初始化 (Init)
    # =================================================
    print(f"\n[3/5] 🤖 Initializing Models...")
    # 调用工厂获取模型
    model = get_model(args.model, num_classes)
    # 定义损失函数 (分类任务通常用交叉熵)
    criterion = nn.CrossEntropyLoss()
    # 定义优化器 (Adam 通常比 SGD 收敛快)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_acc = 0.0

    # =================================================
    # [Step 4] 中心化训练框架 (Training Framework)
    # =================================================
    print(f"\n[4/5] 🚀 Start Training Loop...")
    start_total_time = time.time()

    for epoch in range(args.epochs):
        ep_start = time.time()

        # A. 训练阶段
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # B. 验证阶段
        val_metrics = validate(model, val_loader, criterion, DEVICE)

        ep_time = time.time() - ep_start

        # C. 打印日志
        print(f"Epoch {epoch + 1:02d}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_metrics['acc']:.2f}% | "
              f"Time: {ep_time:.1f}s")

        # D. 写入 CSV
        logger.log_metrics(epoch, train_loss, train_acc, val_metrics['loss'], val_metrics['acc'], 0, ep_time)

        # E. 保存最佳模型 (Checkpointing)
        # 只有当验证集准确率创新高时，才保存模型权重
        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            torch.save(model.state_dict(), logger.save_dir / "best_model.pth")
            print(f"  New Best Model Saved!")

    total_time = (time.time() - start_total_time) / 60
    print(f"\nTraining Finished. Total Time: {total_time:.1f} mins. Best Val Acc: {best_acc:.2f}%")

    # =================================================
    # [Step 5] 最终测试 & 数据保存 (Final Test & Save)
    # =================================================
    print(f"\n[5/5] 🧪 Final Evaluation on Test Set...")

    # 重新加载训练过程中保存的最好的那个模型，而不是最后一个 Epoch 的模型
    model.load_state_dict(torch.load(logger.save_dir / "best_model.pth"))

    # 在测试集上跑一遍，并且 collect_data=True (收集画图数据)
    test_res, feats, labs, preds, logits = validate(model, test_loader, criterion, DEVICE, collect_data=True)

    # 保存 .npz 数据
    save_analysis_data(logger.save_dir, feats, labs, preds, logits, phase='test')
    print(f"\n[Done] All results saved to: {logger.save_dir}")

    print_box("EXPERIMENT FINISHED", [
        f"Total Time:   {total_time:.1f} mins",
        f"Best Val Acc: {test_res['acc']:.2f}%",
        f"Status:       Success"
    ])


if __name__ == "__main__":
    main()