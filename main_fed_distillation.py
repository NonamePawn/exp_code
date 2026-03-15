import argparse
import time
import torch
import copy
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- 导入自定义模块 ---
from dataSet.dataset import get_dataloader
from dataSet.partitioner import FederatedPartitioner
from utils.logger import ExperimentLogger, save_analysis_data
from model.model_factory import get_model
from endPoints.server import Server
from endPoints.client_distill import DistillClient


# --- 辅助打印工具 ---
def print_box(title, content_lines):
    width = 76
    print(f"\n╔{'═' * width}╗")
    print(f"║ {title.center(width - 2)} ║")
    print(f"╠{'═' * width}╣")
    for line in content_lines:
        print(f"║ {line.ljust(width - 2)} ║")
    print(f"╚{'═' * width}╝")

def main():
    # =================================================
    # [Step 1] 参数配置 (Configuration)
    # =================================================
    parser = argparse.ArgumentParser(description="Federated Distillation")

    # 基础实验设置
    parser.add_argument('--structure_name', type=str, default='fed_distill', help='架构名称')
    parser.add_argument('--data_root', type=str, default='/assets_processed', help='数据根目录')
    parser.add_argument('--model', type=str, default='fdmff', help='模型名称')
    parser.add_argument('--device', type=str, default='cuda', help='运算设备')

    # 联邦学习设置
    parser.add_argument('--num_clients', type=int, default=10, help='客户端数量')
    parser.add_argument('--rounds', type=int, default=50, help='通信轮次')
    parser.add_argument('--local_epochs', type=int, default=3, help='本地训练轮次')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='学习率')

    # [蒸馏核心参数]
    parser.add_argument('--kd_alpha', type=float, default=0.3, help='蒸馏权重 (0.3表示30%看老师, 70%看自己)')
    parser.add_argument('--kd_temp', type=float, default=4.0, help='蒸馏温度')

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

    # 打印配置表
    print_box("EXPERIMENTAL INFORMATION", [
        f"Info: Structure={args.structure_name} | Model={args.model} | Data Root={args.data_root}",
        f"Settings: Batch Size={args.batch_size} | Learning Rate={args.learning_rate} | Device={DEVICE}",
        f"Federated Params: Clients={args.num_clients} | Rounds={args.rounds} | Local Epochs={args.local_epochs}",
        f"Distill Params:  Alpha={args.kd_alpha} | Temp={args.kd_temp}",
        hw_info
    ])

    # 初始日志记录器
    logger = ExperimentLogger(save_dir=f"result/{args.structure_name}")

    # =================================================
    # [Step 2] 数据准备 (Data Preparation)
    # =================================================
    print(f"\n[1/5] 📂 Preparing Data...")

    # 使用 Path 对象
    root_path = Path(args.data_root)
    train_path = root_path / 'train'
    val_path = root_path / 'val'
    test_path = root_path / 'test'

    # 1. 训练数据 (Client)
    train_partitioner = FederatedPartitioner(data_root=train_path, num_clients=args.num_clients)
    private_data_map = train_partitioner.split_iid()
    total_private_samples = len(train_partitioner.data_indices)

    # 2. 验证与测试数据 (Server)
    val_scanner = FederatedPartitioner(data_root=val_path, num_clients=1)
    val_loader = get_dataloader(val_scanner.data_indices, batch_size=args.batch_size, is_train=False)

    test_scanner = FederatedPartitioner(data_root=test_path, num_clients=1)
    test_loader = get_dataloader(test_scanner.data_indices, batch_size=args.batch_size, is_train=False)

    num_classes = len(train_partitioner.class_map)

    # 打印汇总表
    print_box("DATASET SUMMARY", [
        f"Classes:       {num_classes}",
        f"Private Train: {total_private_samples} samples (Distributed to {args.num_clients} Clients)",
        f"               -> Avg {total_private_samples // args.num_clients} samples per Client",
        f"Validation:    {len(val_scanner.data_indices)} samples",
        f"Test Data:     {len(test_scanner.data_indices)} samples"
    ])

    # =================================================
    # [Step 3] 模型初始化 (无预训练)
    # =================================================
    print(f"\n[2/5] 🤖 Initializing Models...")
    global_model = get_model(args.model, num_classes=num_classes)

    # 【修复点 1】：先在 CPU 上复制 10 份，绝对不要在 GPU 上复制！
    local_models = [copy.deepcopy(global_model) for _ in range(args.num_clients)]

    # 复制完之后，再把服务端模型放上 GPU
    global_model.to(DEVICE)

    server = Server(global_model, val_loader, DEVICE)

    clients = []
    print(f"   -> Creating {args.num_clients} Clients with ONLY Private Data...")
    for i in range(args.num_clients):
        c = DistillClient(
            client_id=i,
            private_dataset=get_dataloader(private_data_map[i], args.batch_size, is_train=True).dataset,
            device=DEVICE,
            batch_size=args.batch_size
        )
        clients.append(c)

    # =================================================
    # [Step 4] 在线类原型蒸馏协同框架 (Class-Proto ODCM Loop)
    # =================================================
    print(f"\n[3/5] 🚀 Starting Federated Distillation Loop")
    best_acc = 0.0
    best_model_path = logger.save_dir / "best_fed_distill.pth"
    start_time_total = time.time()
    for round_idx in range(1, args.rounds + 1):
        round_start = time.time()
        print(f"\n🔰 Round {round_idx} / {args.rounds}")

        # --- 阶段 1: 各中心在私有数据上提取类原型 ---
        print("   📡 Extracting Class-wise Logits from clients' private data...")
        all_class_logits = {}
        all_class_counts = {}
        # 【修复点 2】：加上最外层的 TQDM 进度条
        with tqdm(total=args.num_clients, desc="   🔍 Extracting", ncols=100, colour='yellow') as pbar:
            for i, client in enumerate(clients):
                # 此时提取特征，客户端内部会把自己的 local_models[i] 放上 GPU，算完再拿下来
                c_logits, c_counts = client.get_local_class_logits(local_models[i])
                all_class_logits[i] = c_logits
                all_class_counts[i] = c_counts
                pbar.update(1)

        # --- 阶段 2: 服务端排他性聚合 ---
        # 计算每个客户端专属的虚拟教师类别知识 z_{-k}
        exclusive_class_logits_dict = server.compute_exclusive_class_logits(all_class_logits, all_class_counts)

        # --- 阶段 3: 客户端在本地私有数据上协同演进 ---
        local_weights = []
        local_losses = []
        with tqdm(total=args.num_clients, desc="   ⚗️Distilling", ncols=100, colour='cyan') as pbar:
            for i, client in enumerate(clients):
                w, loss = client.train_odcm_no_public(
                    local_model=local_models[i],
                    exclusive_class_logits=exclusive_class_logits_dict[i],
                    epochs=args.local_epochs,
                    lr=args.learning_rate,
                    alpha=args.kd_alpha,
                    temperature=args.kd_temp
                )
                local_weights.append(w)
                local_losses.append(loss)
                local_models[i].load_state_dict(w)  # 本地模型自我保留

                pbar.set_postfix({"Loss": f"{loss:.3f}"})
                pbar.update(1)

        # --- 阶段 4: 服务端验证聚合 (纯联邦蒸馏评估，坚决不用 FedAvg) ---
        val_acc_list = []
        val_loss_list = []

        # 依次评估每个客户端自己训练的个性化模型
        for i in range(args.num_clients):
            # 临时将客户端的权重装载到服务端的验证模型中进行测试
            server.global_model.load_state_dict(local_weights[i])
            c_acc, c_loss = server.evaluate()
            val_acc_list.append(c_acc)
            val_loss_list.append(c_loss)

        # 计算所有客户端模型的平均表现，作为整个联邦系统的指标
        val_acc = sum(val_acc_list) / len(val_acc_list)
        val_loss = sum(val_loss_list) / len(val_loss_list)

        # ... 后续打印日志的代码保持不变 ...
        round_time = time.time() - round_start
        logger.log_metrics(round_idx, sum(local_losses) / len(local_losses), 0, val_loss, val_acc, 0,
                           round_time)

        # ==========================================
        # 🌟 核心修改：全能王保存策略
        # ==========================================
        # 1. 找出本轮所有客户端里的“单科最高分”及其索引
        current_max_acc = max(val_acc_list)
        best_client_idx = val_acc_list.index(current_max_acc)

        # 2. 打印一下本轮的极值情况，方便你监控
        print(
            f"   📈 Round {round_idx} Max Acc: {current_max_acc:.2f}% (from Client {best_client_idx}) | Avg Acc: {val_acc:.2f}%")

        # 3. 只有当“单科最高分”打破了历史记录，才保存为全能王！
        if current_max_acc > best_acc:
            best_acc = current_max_acc
            torch.save(local_weights[best_client_idx], best_model_path)
            print(
                f"   🏆 [全能王诞生！] New Best Model Saved! Client {best_client_idx} hit the historical peak: {best_acc:.2f}%")

        # 清理显存
        torch.cuda.empty_cache()

    # =================================================
    # [Step 5] 最终测试 & 数据保存 (Final Test & Save)
    # =================================================
    print(f"\n[4/5] 🧪 Final Evaluation on Test Set...")

    if best_model_path.exists():
        features, labels, preds, logits = server.run_final_test(
            model_path=best_model_path,
            test_loader=test_loader
        )

        print(f"\n[5/5] 💾 Saving Analysis Data...")
        save_analysis_data(
            save_dir=logger.save_dir,
            features=features,
            labels=labels,
            preds=preds,
            logits=logits,
            phase='fed_distill_test'
        )
        print(f"   -> Data saved to: {logger.save_dir}")
    else:
        print("❌ Error: No best model found.")

    total_time = (time.time() - start_time_total) / 60
    print_box("EXPERIMENT FINISHED", [
        f"Total Time:   {total_time:.1f} mins",
        f"Best Val Acc: {best_acc:.2f}%",
        f"Status:       Success"
    ])


if __name__ == '__main__':
    main()