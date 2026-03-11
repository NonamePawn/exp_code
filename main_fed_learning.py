import argparse
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- 导入自定义模块 ---
from dataSet.dataset import get_dataloader
from dataSet.partitioner import FederatedPartitioner
# 引入 save_analysis_data 用于保存 .npz
from utils import ExperimentLogger, save_analysis_data
# 导入模型工厂
from model.model_factory import get_model

from endPoints.client import Client
from endPoints.server import Server


# --- 辅助打印函数 ---
def print_box(title, content_lines):
    width = 72
    print(f"\n╔{'═' * width}╗")
    print(f"║ {title.center(width - 2)} ║")
    print(f"╠{'═' * width}╣")
    for line in content_lines:
        print(f"║ {line.ljust(width - 2)} ║")
    print(f"╚{'═' * width}╝")


def get_args():
    parser = argparse.ArgumentParser(description="Federated Learning Framework")
    parser.add_argument('--exp_name', type=str, default='fed_resnet50_final', help='Experiment Name')
    parser.add_argument('--data_root', type=str, default='/assets_processed', help='Data Root')
    parser.add_argument('--model', type=str, default='resnet50', help='Model Name')

    parser.add_argument('--num_clients', type=int, default=10, help='Number of Clients')
    parser.add_argument('--rounds', type=int, default=50, help='Total Rounds')
    parser.add_argument('--local_epochs', type=int, default=3, help='Local Epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')

    return parser.parse_args()


def main():
    args = get_args()

    # 1. 硬件配置
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        # 获取显存 (GB)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        hw_info = f"🚀 GPU: {gpu_name} ({total_mem:.1f} GB) | Device: {DEVICE}"
    else:
        DEVICE = torch.device("cpu")
        hw_info = "🐢 Device: CPU"

    config_lines = [
        f"Experiment: {args.exp_name}",
        f"Data Root:  {args.data_root}",
        f"Model:      {args.model}",
        f"Settings:   Clients={args.num_clients} | Rounds={args.rounds}",
        f"Params:     Ep={args.local_epochs} | BS={args.batch_size} | LR={args.lr}",
        hw_info
    ]
    print_box("EXPERIMENT CONFIGURATION", config_lines)

    logger = ExperimentLogger(save_dir=f"result/{args.exp_name}")

    # =================================================
    # [Phase 1] 数据准备
    # =================================================
    print(f"\n[1/4] 📂 Partitioning Data...")
    train_path = Path(args.data_root) / 'train'
    val_path = Path(args.data_root) / 'val'
    test_path = Path(args.data_root) / 'test'

    # 1. 训练集切分 (Train Partitioning)
    train_partitioner = FederatedPartitioner(data_root=train_path, num_clients=args.num_clients)
    client_data_map = train_partitioner.split_iid()

    # 2. 验证集加载 (Val Scan - 用于每轮监控)
    val_scanner = FederatedPartitioner(data_root=val_path, num_clients=1)
    # 验证时 Batch Size 可以大一点，因为不反向传播
    val_loader = get_dataloader(val_scanner.data_indices, batch_size=args.batch_size * 2, is_train=False)

    # 3. 测试集加载 (Test Scan - 用于最终测试)
    print(f"   -> Scanning Test Data from: {test_path}")
    if test_path.exists():
        test_scanner = FederatedPartitioner(data_root=test_path, num_clients=1)
        test_loader = get_dataloader(test_scanner.data_indices, batch_size=args.batch_size * 2, is_train=False)
        print(f"      Found {len(test_scanner.data_indices)} test images.")
    else:
        print("⚠️ Warning: Test path not found! Using Val set as fallback.")
        test_loader = val_loader

    num_classes = len(train_partitioner.class_map)
    print(f"   -> Classes: {num_classes} | Train Samples: {sum([len(v) for v in client_data_map.values()])}")

    # =================================================
    # [Phase 2] 模型初始化
    # =================================================
    print(f"\n[2/4] 🤖 Initializing Nodes...")

    global_model = get_model(args.model, num_classes=num_classes)
    global_model.to(DEVICE)

    server = Server(global_model, val_loader, DEVICE)
    clients = [Client(i, client_data_map[i], DEVICE, args.batch_size) for i in range(args.num_clients)]

    # =================================================
    # [Phase 3] 联邦训练循环
    # =================================================
    best_acc = 0.0
    start_time_total = time.time()
    best_model_path = logger.save_dir / "best_fed_model.pth"

    print(f"\n[3/4] 🔥 Starting Training Loop ({args.rounds} Rounds)...")

    for round_idx in range(1, args.rounds + 1):
        round_start = time.time()
        print(f"\n🔰 Global Round {round_idx} / {args.rounds}")

        local_weights, local_losses = [], []

        # A. Client Training (使用绿色进度条)
        with tqdm(total=args.num_clients, desc="   🔄 Training", unit="client", ncols=100, colour='green') as pbar:
            for client in clients:
                w, loss = client.train(server.global_model, args.local_epochs, args.lr, verbose=False)
                local_weights.append(w)
                local_losses.append(loss)
                pbar.set_postfix({"Loss": f"{loss:.3f}"})
                pbar.update(1)

        # B. Aggregation
        server.aggregate(local_weights)

        # C. Evaluation (Fast validation using Val set)
        val_acc, val_loss = server.evaluate()

        # D. Logging
        round_time = time.time() - round_start
        avg_train_loss = sum(local_losses) / len(local_losses)

        print(
            f"   📊 Round {round_idx}: Val Acc \033[92m{val_acc:.2f}%\033[0m | Val Loss {val_loss:.4f} | Time {round_time:.1f}s")

        logger.log_metrics(round_idx, avg_train_loss, 0.0, val_loss, val_acc, 0.0, round_time)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(server.global_model.state_dict(), best_model_path)
            print(f"   🏆 New Best Saved! ({best_acc:.2f}%)")

    # =================================================
    # [Phase 4] 最终测试 & 数据保存
    # =================================================
    print(f"\n[4/4] 🧪 Final Testing & Data Saving...")

    if not best_model_path.exists():
        print("⚠️ Warning: No best model found. Using current model.")
    else:
        # [修改] 调用 run_final_test 时传入 test_loader
        features, labels, preds, logits = server.run_final_test(
            model_path=best_model_path,
            test_loader=test_loader
        )

        print(f"   -> Saving analysis data to {logger.save_dir}")
        save_analysis_data(
            save_dir=logger.save_dir,
            features=features,
            labels=labels,
            preds=preds,
            logits=logits,
            phase='fed_test'
        )

    # Final Report
    total_time = (time.time() - start_time_total) / 60
    final_report = [
        f"Total Time:    {total_time:.1f} mins",
        f"Best Val Acc:  {best_acc:.2f}%",
        f"Log Saved:     {logger.save_dir}",
        f"Status:        Success"
    ]
    print_box("EXPERIMENT COMPLETED", final_report)


if __name__ == '__main__':
    main()