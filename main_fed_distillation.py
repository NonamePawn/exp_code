import argparse
import time
import torch
import copy
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from collections import Counter
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
    parser.add_argument('--num_clients', type=int, default=4, help='客户端数量')
    parser.add_argument('--rounds', type=int, default=50, help='通信轮次')
    parser.add_argument('--local_epochs', type=int, default=2, help='本地训练轮次')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='学习率')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.1, help='狄利克雷参数')

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

    # ==========================================
    # 🛡️ 实验室生存法则：显存占位护盾
    # ==========================================
    if DEVICE.type == 'cuda':
        print("\n[0/5] 🛡️ Activating VRAM Guard to prevent resource hijacking...")
        # 我们预先强行申请 23GB 的显存 (23 * 1024 * 1024 * 1024 bytes)
        # 因为 float32 占 4 个字节，所以除以 4
        guard_size_gb = 23
        tensor_size = int(guard_size_gb * (1024 ** 3) / 4)

        try:
            # 生成占位张量，瞬间吃掉 23GB
            guard_tensor = torch.empty(tensor_size, dtype=torch.float32, device=DEVICE)

            # 立刻删除该张量变量
            del guard_tensor
            # 加入这一行：清空历史峰值记录，重新开始统计真正的模型开销
            torch.cuda.reset_peak_memory_stats()
            # 绝对不要写 torch.cuda.empty_cache()！
            # 此时这 23GB 已经成为你当前 Python 进程的专属私有缓存，
            # 等到训练阶段需要 25GB 时，PyTorch 会直接从这里面拿，而不会 OOM。
            print(f"      ✅ Successfully reserved {guard_size_gb}GB of VRAM!")
        except Exception as e:
            print(f"      ⚠️ VRAM Guard failed: Not enough free memory right now. {e}")

    # 打印配置表
    print_box("EXPERIMENTAL INFORMATION", [
        f"Info: Structure={args.structure_name} | Model={args.model} | Data Root={args.data_root}",
        f"Settings: Batch Size={args.batch_size} | Learning Rate={args.learning_rate} | Device={DEVICE}",
        f"Federated Params: Clients={args.num_clients} | Rounds={args.rounds} | Local Epochs={args.local_epochs}",
        f"Distill Params: Distill Alpha={args.kd_alpha} | Temp={args.kd_temp} | Dirichlet Alpha = {args.dirichlet_alpha}",
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
    print(f"   📊 [Non-IID] Applying Dirichlet distribution (alpha={args.dirichlet_alpha}) for data partitioning...")

    # 1. 训练数据 (Client)
    train_partitioner = FederatedPartitioner(data_root=train_path, num_clients=args.num_clients)
    # 🌟 把原本写死的 0.1 替换为 args.dirichlet_alpha
    private_data_map = train_partitioner.split_non_iid_dirichlet(alpha=args.dirichlet_alpha)
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

    # 1. 建立类别 ID 到设备名称的反向映射
    id_to_class = {v: k for k, v in train_partitioner.class_map.items()}

    dist_lines = []
    # 2. 遍历每个客户端统计样本分布
    for i in range(args.num_clients):
        # 统计当前客户端所有样本的标签数量
        c_labels = [item['label'] for item in private_data_map[i]]
        counts = Counter(c_labels)

        # 格式化输出：只显示该客户端拥有的类别，并按名称排序
        # 结果类似于: "Client 00 ( 450 samples) -> [Canon: 120, Siemens: 330]"
        stat_str = ", ".join([f"{id_to_class[lbl]}: {count}" for lbl, count in sorted(counts.items())])
        dist_lines.append(f"Client {i:02d} ({len(private_data_map[i]):4d} samples) -> [{stat_str}]")

    # 3. 调用你现有的 print_box 工具进行可视化展示
    print_box("DETAILED CLIENT-DATA DISTRIBUTION (NON-IID)", dist_lines)

    # =================================================
    # [Step 3] 模型初始化 (无预训练)
    # =================================================
    print(f"\n[2/5] 🤖 Initializing Models...")
    global_model = get_model(args.model, num_classes=num_classes)

    # 🌟 核心：只在 CPU 里维护轻量级的参数字典，而不是整个模型实例
    local_weights = [{k: v.cpu().clone() for k, v in global_model.state_dict().items()} for _ in
                     range(args.num_clients)]

    # 全局模型长期驻留 GPU，作为服务端的验证模型
    global_model.to(DEVICE)
    server = Server(global_model, val_loader, DEVICE)

    # 创建一个同样长期驻留 GPU 的“客户端打工模型”
    worker_model = get_model(args.model, num_classes=num_classes)
    worker_model.to(DEVICE)

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
    global_best_acc = 0.0
    # --- 2各个中心专属追踪变量 (新增) ---
    clients_best_acc = [0.0] * args.num_clients
    best_model_path = logger.save_dir / "best_model.pth"
    start_time_total = time.time()
    for round_idx in range(1, args.rounds + 1):
        round_start = time.time()
        print(f"\n🔰 Round {round_idx} / {args.rounds}")

        # --- 阶段 1: 各中心在私有数据上提取类原型 ---
        print("   📡 Extracting Class-wise Logits from clients' private data...")
        all_class_logits = {}
        all_class_counts = {}
        with tqdm(total=args.num_clients, desc="   🔍 Extracting", ncols=100, colour='yellow') as pbar:
            for i, client in enumerate(clients):
                # 🌟 让 GPU 上的打工模型加载当前客户端的权重
                worker_model.load_state_dict(local_weights[i])

                c_logits, c_counts = client.get_local_class_logits(worker_model)
                all_class_logits[i] = c_logits
                all_class_counts[i] = c_counts
                pbar.update(1)

        # --- 阶段 2: 服务端排他性聚合 ---
        # 计算每个客户端专属的虚拟教师类别知识 z_{-k}
        exclusive_class_logits_dict = server.compute_exclusive_class_logits(all_class_logits, all_class_counts)

        # --- 阶段 3: 客户端在本地私有数据上协同演进 ---
        local_losses = []
        with tqdm(total=args.num_clients, desc="   ⚗️Distilling", ncols=100, colour='cyan') as pbar:
            for i, client in enumerate(clients):
                # 🌟 让 GPU 上的打工模型加载当前客户端的权重
                worker_model.load_state_dict(local_weights[i])

                w_cpu, loss = client.train_odcm_no_public(
                    local_model=worker_model,
                    exclusive_class_logits=exclusive_class_logits_dict[i],
                    epochs=args.local_epochs,
                    lr=args.learning_rate,
                    alpha=args.kd_alpha,
                    temperature=args.kd_temp
                )

                # 🌟 把更新后的 CPU 权重存回列表
                local_weights[i] = w_cpu
                local_losses.append(loss)
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
        round_time_min = round_time / 60
        print(f" Round {round_idx} Total Time: {round_time_min:.2f}min")
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
        if current_max_acc > global_best_acc:
            global_best_acc = current_max_acc
            torch.save(local_weights[best_client_idx], best_model_path)
            print(
                f"   🏆 [全能王诞生！] New Best Model Saved! Client {best_client_idx} hit the historical peak: {global_best_acc:.2f}%")
        # ==========================================
        # 🌟 核心修改 2：各中心个性化最佳模型保存 (新增)
        # ==========================================
        for i in range(args.num_clients):
            # 如果某个客户端打破了它自己的历史记录
            if val_acc_list[i] > clients_best_acc[i]:
                clients_best_acc[i] = val_acc_list[i]
                client_model_path = logger.save_dir / f"best_client_{i}.pth"
                torch.save(local_weights[i], client_model_path)
                # 静默保存，不 print 以免刷屏

        # 查看当前瞬间的真实显存占用 (GB)
        current_vram = torch.cuda.memory_allocated() / (1024 ** 3)

        # 查看从程序启动到现在的历史显存峰值 (GB) —— 这个最重要，它决定了你到底会不会 OOM！
        max_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

        print(f"📊 真实显存: 当前 {current_vram:.2f} GB | 峰值 {max_vram:.2f} GB")
        # 清理显存
        # torch.cuda.empty_cache()

    # =================================================
    # [Step 5] 最终测试 & 数据保存 (Final Test & Save)
    # =================================================
    print(f"\n[4/5] 🧪 Final Evaluation on Test Set...")

    if best_model_path.exists():
        features, labels, preds, logits = server.run_final_test(
            model_path=best_model_path,
            test_loader=test_loader
        )

        print(f"\n 💾 Saving Analysis Data...")
        save_analysis_data(
            save_dir=logger.save_dir,
            features=features,
            labels=labels,
            preds=preds,
            logits=logits,
            phase=args.structure_name + '_' + args.model
        )
        print(f"   -> Data saved to: {logger.save_dir}")
    else:
        print("❌ Error: No best model found.")

    # --- 阶段 B: 测试各中心模型 (用于消融实验的均值和方差) ---
    print(f"\n[5/5] 🧪 Evaluating Individual Clients for Ablation Metrics...")

    client_final_accs = []
    client_final_f1s = []

    for i in range(args.num_clients):
        model_path = logger.save_dir / f"best_client_{i}.pth"
        if model_path.exists():
            # 这里我们只需要 labels 和 preds 算指标，不需要保存特征
            _, c_labels, c_preds, _ = server.run_final_test(
                model_path=model_path,
                test_loader=test_loader
            )

            acc = accuracy_score(c_labels, c_preds) * 100
            f1 = f1_score(c_labels, c_preds, average='macro') * 100

            client_final_accs.append(acc)
            client_final_f1s.append(f1)
            print(f"   -> Client {i} | Acc: {acc:.2f}% | F1: {f1:.2f}%")

    # 计算均值和方差
    acc_mean, acc_var = np.mean(client_final_accs), np.var(client_final_accs)
    f1_mean, f1_var = np.mean(client_final_f1s), np.var(client_final_f1s)

    # 保存各中心详细指标用于画消融实验表格/柱状图
    metrics_df = pd.DataFrame({
        'Client_ID': range(args.num_clients),
        'Accuracy': client_final_accs,
        'F1_Score': client_final_f1s
    })
    metrics_df.loc['Mean'] = ['Mean', acc_mean, f1_mean]
    metrics_df.loc['Variance'] = ['Var', acc_var, f1_var]

    csv_path = logger.save_dir / "clients_final_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)

    total_time = (time.time() - start_time_total) / 60
    print_box("EXPERIMENT FINISHED", [
        f"Global Best Acc: {global_best_acc:.2f}%",
        f"Client Acc -> Mean: {acc_mean:.2f}% | Variance: {acc_var:.2f}",
        f"Client F1  -> Mean: {f1_mean:.2f}% | Variance: {f1_var:.2f}",
        f"Total Time:   {total_time:.1f} mins",
        f"Status:       Success"
    ])

if __name__ == '__main__':
    main()