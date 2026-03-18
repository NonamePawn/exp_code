# 文件名: main_fed_distillation_openset.py
import argparse  # 导入命令行参数解析库
import time  # 导入时间模块，用于计算每轮训练耗时
import torch  # 导入 PyTorch
import copy  # 导入拷贝模块，主要用于深度拷贝(deepcopy)模型权重参数
import numpy as np  # 导入 NumPy
from pathlib import Path  # 导入 Path 库，用于优雅且跨平台地处理文件路径
from tqdm import tqdm  # 导入进度条
from sklearn.metrics import accuracy_score, f1_score  # 导入 sklearn 的评价指标
import pandas as pd  # 导入 pandas，用于最后输出和保存实验数据的 CSV 表格
from collections import Counter  # 导入计数器，用于统计本地数据集的类别分布

# --- 导入自定义模块 (确保这些文件在你项目中真实存在) ---
from dataSet.dataset import get_dataloader
from dataSet.partitioner import FederatedPartitioner
from utils.logger import ExperimentLogger, save_analysis_data
from endPoints.server import Server

# 导入我们在第四章专属构建的：K+1 维解耦模型 & 双轨开集客户端
from model.FDMFF_openset import FDMFF_OpenSet
from endPoints.client_distill_openset import OpenSetDistillClient


def print_box(title, content_lines):
    """
    辅助打印工具：用于在终端打印一个美观的带边框的配置信息面板。
    """
    width = 76  # 预设面板宽度
    print(f"\n╔{'═' * width}╗")
    print(f"║ {title.center(width - 2)} ║")  # 标题居中
    print(f"╠{'═' * width}╣")
    for line in content_lines:
        print(f"║ {line.ljust(width - 2)} ║")  # 内容左对齐并填充空格
    print(f"╚{'═' * width}╝")


# =========================================================
# 🌟 [4.2.4 新增] 联邦投票开集协同策略：全局投票共识生成
# =========================================================
def compute_fv_ocs_consensus(all_class_logits, all_class_counts, current_tau):
    """
    根据论文公式 (4-10) 和 (4-11) 计算加权全局共识 P_vote
    """
    global_consensus_probs = {}
    class_client_weights = {}

    # 1. 收集各局部模型在对应类别的特征响应 (Logits)
    for client_id, c_logits in all_class_logits.items():
        for c_id, logits in c_logits.items():
            if c_id not in global_consensus_probs:
                global_consensus_probs[c_id] = []
                class_client_weights[c_id] = []

            # 公式 4-10：应用动态温度 τ(t) 进行概率软化
            soft_prob = torch.nn.functional.softmax(logits / current_tau, dim=0)
            global_consensus_probs[c_id].append(soft_prob)

            # 记录该中心的历史协同贡献度 (以样本量表征可靠性权重 γ_m)
            class_client_weights[c_id].append(all_class_counts[client_id][c_id])

    # 2. 加权投票生成全局共识 P_vote
    final_consensus = {}
    for c_id in global_consensus_probs.keys():
        probs_stack = torch.stack(global_consensus_probs[c_id])  # [参与该类别的中心数, K+1]
        weights = torch.tensor(class_client_weights[c_id], dtype=torch.float32)
        weights = weights / weights.sum()  # 归一化 γ_m 满足 \sum γ_m = 1

        # 公式 4-11：加权求和 P_vote = \sum γ_m * p_m
        voted_prob = torch.sum(probs_stack * weights.unsqueeze(1), dim=0)
        final_consensus[c_id] = voted_prob

    return final_consensus

def main():
    # =================================================
    # [Step 1] 参数配置 (Configuration)
    # =================================================
    # 实例化参数解析器，并在描述中标注这是第四章的双轨开集 (DNOSM) 实验
    parser = argparse.ArgumentParser(description="Federated Distillation for Open-Set (DNOSM)")

    # 基础实验设置：模型名称、数据根目录、使用的设备类型
    parser.add_argument('--structure_name', type=str, default='fed_openset_dnosm', help='实验架构与保存目录的名称')
    parser.add_argument('--data_root', type=str, default='/assets_processed', help='预处理后数据的绝对或相对路径')
    parser.add_argument('--model', type=str, default='fdmff_openset', help='指定网络模型')
    parser.add_argument('--device', type=str, default='cuda', help='指定优先运行的硬件设备 (cuda/cpu)')

    # 联邦学习通用设置：通信轮数、客户端总数、本地迭代次数等
    parser.add_argument('--num_clients', type=int, default=4, help='参与联邦训练的总客户端数量')
    parser.add_argument('--rounds', type=int, default=50, help='服务端与客户端的全局通信轮次')
    parser.add_argument('--local_epochs', type=int, default=2, help='客户端每次拿到全局模型后，在本地训练的 Epoch 数')
    parser.add_argument('--batch_size', type=int, default=32, help='数据加载器的 Batch Size')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='SGD 优化器的学习率')
    # 狄利克雷分布参数，alpha 越小，Non-IID 程度越极端（数据越集中在单一类别）
    parser.add_argument('--dirichlet_alpha', type=float, default=0.1, help='控制数据异构程度的分布参数')

    # 第三章保留的蒸馏核心参数：温度与权重
    parser.add_argument('--kd_alpha', type=float, default=0.3,
                        help='知识蒸馏所占的 Loss 权重 (0.3代表蒸馏占30%，CE占70%)')
    parser.add_argument('--kd_temp', type=float, default=4.0, help='蒸馏的软化温度(Temperature)，建议值 4.0')

    # 第四章新增的双轨核心参数：极化阈值与扰动步长
    parser.add_argument('--adv_threshold', type=float, default=0.7,
                        help='触发对抗生成轨道的极化浓度阈值 (70%以上走对抗)')
    # 使用 nargs=2，意味着在命令行可以用 --adv_epsilon_range 0.01 0.05 来传入两个值
    parser.add_argument('--adv_epsilon_range', type=float, nargs=2, default=[0.01, 0.05],
                        help='FGSM 实例级动态对抗扰动的步长范围 (min, max)')

    args = parser.parse_args()  # 解析命令行参数字典

    # 硬件环境自检：检测 cuda 是否真实可用
    if torch.cuda.is_available() and args.device == 'cuda':
        DEVICE = torch.device("cuda")  # 设置全局默认设备为 GPU
        gpu_name = torch.cuda.get_device_name(0)  # 获取显卡型号，如 "RTX A6000"
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # 获取物理显存大小 (GB)
        hw_info = f"🚀 GPU: {gpu_name} ({total_mem:.1f} GB)"
    else:
        DEVICE = torch.device("cpu")  # 退化为 CPU 运行
        hw_info = "🐢 Device: CPU"

    # ==========================================
    # 🛡️ 实验室生存法则：VRAM Guard 显存占位护盾
    # ==========================================
    if DEVICE.type == 'cuda':
        print("\n[0/5] 🛡️ Activating VRAM Guard to prevent resource hijacking...")
        guard_size_gb = 23  # 预先向系统强行申请 23GB 的连续显存
        tensor_size = int(guard_size_gb * (1024 ** 3) / 4)  # 计算所需张量元素的个数 (float32占4字节)
        try:
            # 在 GPU 上创建一个巨大的空张量，瞬间“霸占”这 23GB
            guard_tensor = torch.empty(tensor_size, dtype=torch.float32, device=DEVICE)
            # 立刻删除该 Python 变量的引用，但底层显存并未退还给操作系统，而是成为了 PyTorch 的内部缓存池
            del guard_tensor
            # 清空历史峰值统计，重新记录真实模型的显存消耗
            torch.cuda.reset_peak_memory_stats()
            print(f"      ✅ Successfully reserved {guard_size_gb}GB of VRAM!")
        except Exception as e:
            # 如果显卡本身空闲不足 23GB，则打印警告并继续执行
            print(f"      ⚠️ VRAM Guard failed: Not enough free memory right now. {e}")

    # 调用格式化打印函数，输出参数表
    print_box("EXPERIMENTAL INFORMATION (OPEN-SET)", [
        f"Info: Structure={args.structure_name} | Model={args.model} | Data Root={args.data_root}",
        f"Settings: Batch Size={args.batch_size} | Learning Rate={args.learning_rate} | Device={DEVICE}",
        f"Federated: Clients={args.num_clients} | Rounds={args.rounds} | Alpha = {args.dirichlet_alpha}",
        f"Open-Set Params: Adv Threshold={args.adv_threshold * 100}% | Eps Range={args.adv_epsilon_range}",  # <--- 修改这行
        hw_info
    ])

    # 实例化日志记录器，用于在 result 文件夹下生成日志文本
    logger = ExperimentLogger(save_dir=f"result/{args.structure_name}")

    # =================================================
    # [Step 2] 数据准备 (Data Preparation)
    # =================================================
    print(f"\n[1/5] 📂 Preparing Data...")

    root_path = Path(args.data_root)  # 包装根目录路径
    train_path = root_path / 'train'  # 自动拼接出 train 文件夹路径
    val_path = root_path / 'val'  # 验证集路径
    test_path = root_path / 'test'  # 测试集路径

    print(f"   📊 [Non-IID] Applying Dirichlet distribution (alpha={args.dirichlet_alpha})...")

    # 1. 实例化客户端的训练数据划分器 (Partitioner)
    train_partitioner = FederatedPartitioner(data_root=train_path, num_clients=args.num_clients)
    # 基于指定的狄利克雷 alpha 参数进行 Non-IID 切分，返回每个客户端拥有的样本路径/信息字典
    private_data_map = train_partitioner.split_non_iid_dirichlet(alpha=args.dirichlet_alpha)
    total_private_samples = len(train_partitioner.data_indices)  # 训练集总样本数

    # 2. 实例化服务端的验证集和测试集 (不对它们切分，num_clients=1 意为服务端独占整体数据)
    val_scanner = FederatedPartitioner(data_root=val_path, num_clients=1)
    val_loader = get_dataloader(val_scanner.data_indices, batch_size=args.batch_size, is_train=False)

    test_scanner = FederatedPartitioner(data_root=test_path, num_clients=1)
    test_loader = get_dataloader(test_scanner.data_indices, batch_size=args.batch_size, is_train=False)

    num_classes = len(train_partitioner.class_map)  # 获取系统已知的 CT 设备类别总数 K

    # 打印数据分布概要信息，明确标出 K+1 维的信息
    print_box("DATASET SUMMARY", [
        f"Known Classes (K): {num_classes} | Open-Set Head: {num_classes + 1} dims",
        f"Private Train: {total_private_samples} samples (Distributed to {args.num_clients} Clients)",
        f"Validation:    {len(val_scanner.data_indices)} samples",
        f"Test Data:     {len(test_scanner.data_indices)} samples"
    ])

    # 反向字典推导式：建立从整数类别 ID 到字符串真实名称的映射，方便打印可读日志
    id_to_class = {v: k for k, v in train_partitioner.class_map.items()}
    dist_lines = []

    # 遍历所有客户端，统计并打印它们本地到底分到了哪些类别的 CT 数据
    for i in range(args.num_clients):
        # 提取第 i 个客户端拥有的所有标签
        c_labels = [item['label'] for item in private_data_map[i]]
        counts = Counter(c_labels)  # 使用 Counter 工具快速统计频次
        # 拼装字符串：格式如 "Siemens: 100, Canon: 10"
        stat_str = ", ".join([f"{id_to_class[lbl]}: {count}" for lbl, count in sorted(counts.items())])
        # 将结果存入准备打印的列表中
        dist_lines.append(f"Client {i:02d} ({len(private_data_map[i]):4d} samples) -> [{stat_str}]")

    print_box("DETAILED CLIENT-DATA DISTRIBUTION (NON-IID)", dist_lines)

    # =================================================
    # [Step 3] 模型初始化 (K+1 维双轨架构)
    # =================================================
    print(f"\n[2/5] 🤖 Initializing Models...")

    # 实例化服务端的全局模型，注意此时网络内部的 classifier 已经变成了 K+1 维
    global_model = FDMFF_OpenSet(num_classes=num_classes)

    # 核心显存节省策略：建立一个列表，仅在 CPU 的普通内存中保存各客户端最新一轮的模型权重（状态字典）
    # .cpu().clone() 确保这些权重与当前模型彻底切断引用关联
    local_weights = [{k: v.cpu().clone() for k, v in global_model.state_dict().items()} for _ in
                     range(args.num_clients)]

    global_model.to(DEVICE)  # 服务端全局模型长期驻扎在 GPU 显存，用于性能验证
    server = Server(global_model, val_loader, DEVICE)  # 初始化服务端评价引擎

    # 实例化一个打工仔模型 (worker_model)，同样长期驻扎在 GPU
    # 它负责在每次客户端训练时“套用”对应客户端的权重去干活，避免了反复把模型搬进搬出 GPU 导致的极高 IO 延迟
    worker_model = FDMFF_OpenSet(num_classes=num_classes)
    worker_model.to(DEVICE)

    clients = []
    print(f"   -> Creating {args.num_clients} Open-Set Clients with Dual-Track Routing...")
    # 循环实例化联邦的参与方（第四章专用的 OpenSetDistillClient）
    for i in range(args.num_clients):
        c = OpenSetDistillClient(
            client_id=i,
            # 将对应的原始图片路径转化为真实的 PyTorch Dataset 实例并传入
            private_dataset=get_dataloader(private_data_map[i], args.batch_size, is_train=True).dataset,
            device=DEVICE,
            batch_size=args.batch_size,
            adv_threshold=args.adv_threshold  # 传入极化阈值
        )
        clients.append(c)

    # =================================================
    # [Step 4] 开集联合蒸馏演进 (DNOSM + FV-OCS Loop)
    # =================================================
    print(f"\n[3/5] 🚀 Starting Open-Set Federated Distillation Loop")
    global_best_acc = 0.0  # 全局最高精度记录变量
    clients_best_acc = [0.0] * args.num_clients  # 记录每个客户端自己跑出的历史最佳精度
    best_model_path = logger.save_dir / "best_model.pth"
    start_time_total = time.time()

    # 🌟 [4.2.4 新增] 动态温度参数配置
    initial_tau = args.kd_temp  # 初始温度 (如 4.0)
    min_tau = 1.0  # 最低温度阈值
    decay_rate = 0.95  # 温度衰减率

    for round_idx in range(1, args.rounds + 1):
        round_start = time.time()

        # 🌟 公式 4-10：计算随通信轮次衰减的动态温度 τ(t)
        current_tau = max(min_tau, initial_tau * (decay_rate ** (round_idx - 1)))
        print(f"\n🔰 Round {round_idx} / {args.rounds} [🌡️ Dynamic Temp τ(t): {current_tau:.3f}]")

        # --- 阶段 1: 提取各中心类原型 ---
        print("   📡 Extracting Class-wise Logits from clients' private data...")
        all_class_logits = {}
        all_class_counts = {}
        with tqdm(total=args.num_clients, desc="   🔍 Extracting", ncols=100, colour='yellow') as pbar:
            for i, client in enumerate(clients):
                worker_model.load_state_dict(local_weights[i])
                c_logits, c_counts = client.get_local_class_logits(worker_model)
                all_class_logits[i] = c_logits
                all_class_counts[i] = c_counts
                pbar.update(1)

        # --- 阶段 2: 服务端 FV-OCS 全局投票共识生成 ---
        # 传入当前动态温度，生成加权的全局共识 P_vote
        global_consensus_dict = compute_fv_ocs_consensus(all_class_logits, all_class_counts, current_tau)

        # --- 阶段 3: 客户端在本地私有数据上进行两阶段解耦训练 ---
        local_losses = []
        with tqdm(total=args.num_clients, desc="   ⚗️Distilling & Fine-Tuning", ncols=110, colour='cyan') as pbar:
            for i, client in enumerate(clients):
                worker_model.load_state_dict(local_weights[i])

                # 传入 FV-OCS 共识字典和动态参数，执行双轨生成与两阶段微调
                w_cpu, loss = client.train_openset_model(
                    local_model=worker_model,
                    global_consensus=global_consensus_dict,  # 🌟 传入 P_vote 投票共识
                    epochs=args.local_epochs,
                    lr=args.learning_rate,
                    kd_alpha=args.kd_alpha,
                    temperature=current_tau,  # 🌟 传入当前的动态温度
                    num_known_classes=num_classes,
                    epsilon_range=tuple(args.adv_epsilon_range)  # 🌟 传入实例级动态扰动步长范围
                )

                local_weights[i] = w_cpu
                local_losses.append(loss)
                pbar.set_postfix({"Loss": f"{loss:.3f}"})
                pbar.update(1)

        # --- 阶段 4: 服务端验证聚合 (纯联邦验证) ---
        val_acc_list = []
        val_loss_list = []

        # 依次评估每个客户端进化后的个性化模型
        for i in range(args.num_clients):
            server.global_model.load_state_dict(local_weights[i])
            c_acc, c_loss = server.evaluate()
            val_acc_list.append(c_acc)
            val_loss_list.append(c_loss)

        val_acc = sum(val_acc_list) / len(val_acc_list)
        val_loss = sum(val_loss_list) / len(val_loss_list)

        round_time = time.time() - round_start
        round_time_min = round_time / 60
        print(f" Round {round_idx} Total Time: {round_time_min:.2f}min")

        # 记录指标
        logger.log_metrics(round_idx, sum(local_losses) / len(local_losses), 0, val_loss, val_acc, 0, round_time)

        # ==========================================
        # 🌟 全能王模型保存策略
        # ==========================================
        current_max_acc = max(val_acc_list)
        best_client_idx = val_acc_list.index(current_max_acc)

        print(
            f"   📈 Round {round_idx} Max Acc: {current_max_acc:.2f}% (from Client {best_client_idx}) | Avg Acc: {val_acc:.2f}%")

        if current_max_acc > global_best_acc:
            global_best_acc = current_max_acc
            torch.save(local_weights[best_client_idx], best_model_path)
            print(
                f"   🏆 [全能王诞生！] New Best Open-Set Model Saved! Client {best_client_idx} hit peak: {global_best_acc:.2f}%")

        # 监控每个客户端是否打破局部历史记录
        for i in range(args.num_clients):
            if val_acc_list[i] > clients_best_acc[i]:
                clients_best_acc[i] = val_acc_list[i]
                client_model_path = logger.save_dir / f"best_client_{i}.pth"
                torch.save(local_weights[i], client_model_path)

        current_vram = torch.cuda.memory_allocated() / (1024 ** 3)
        max_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"📊 真实显存: 当前 {current_vram:.2f} GB | 峰值 {max_vram:.2f} GB")

    # =================================================
    # [Step 5] 最终测试 & 绘制所需数据的保存
    # =================================================
    print(f"\n[4/5] 🧪 Final Evaluation on Test Set...")

    # 如果存在全能王模型，我们在全新的、未曾见过的测试集 (test_loader) 上进行最终大考
    if best_model_path.exists():
        features, labels, preds, logits = server.run_final_test(
            model_path=best_model_path,
            test_loader=test_loader
        )

        print(f"\n 💾 Saving Analysis Data...")
        # 将最后一层提取出的多维特征向量、预测标签等存为专门的数据包
        # 你写论文时可以用这些包画出精美的 t-SNE 聚类图或混淆矩阵！
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

    # --- 阶段 B：测试并保存各中心个性化模型的差异度量 ---
    print(f"\n[5/5] 🧪 Evaluating Individual Clients for Ablation Metrics...")

    client_final_accs = []
    client_final_f1s = []

    for i in range(args.num_clients):
        model_path = logger.save_dir / f"best_client_{i}.pth"
        if model_path.exists():
            # 运行最终测试，由于我们只要指标，不需要画图特征，因此丢弃第一和第四个返回值
            _, c_labels, c_preds, _ = server.run_final_test(
                model_path=model_path,
                test_loader=test_loader
            )

            # 使用 sklearn 工具计算并转换成百分制的 Acc 和 F1 分数 (宏平均)
            acc = accuracy_score(c_labels, c_preds) * 100
            f1 = f1_score(c_labels, c_preds, average='macro') * 100

            client_final_accs.append(acc)
            client_final_f1s.append(f1)
            print(f"   -> Client {i} | Acc: {acc:.2f}% | F1: {f1:.2f}%")

    # 使用 NumPy 统计这些客户端模型指标的均值和方差，用于衡量联邦的“系统稳定性”
    acc_mean, acc_var = np.mean(client_final_accs), np.var(client_final_accs)
    f1_mean, f1_var = np.mean(client_final_f1s), np.var(client_final_f1s)

    # 借助 pandas 制作一个规整的 DataFrame 表格
    metrics_df = pd.DataFrame({
        'Client_ID': range(args.num_clients),
        'Accuracy': client_final_accs,
        'F1_Score': client_final_f1s
    })
    # 在表格尾部追加汇总统计行
    metrics_df.loc['Mean'] = ['Mean', acc_mean, f1_mean]
    metrics_df.loc['Variance'] = ['Var', acc_var, f1_var]

    # 将上述 DataFrame 保存为 CSV 格式，方便你直接导入到 Excel 做大论文的表格
    csv_path = logger.save_dir / "clients_final_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)

    total_time = (time.time() - start_time_total) / 60
    # 大功告成，打印最终战报面板
    print_box("EXPERIMENT FINISHED", [
        f"Global Best Acc: {global_best_acc:.2f}%",
        f"Client Acc -> Mean: {acc_mean:.2f}% | Variance: {acc_var:.2f}",
        f"Client F1  -> Mean: {f1_mean:.2f}% | Variance: {f1_var:.2f}",
        f"Total Time:   {total_time:.1f} mins",
        f"Status:       Success"
    ])


# 标准的 Python 入口防护，保证脚本作为主程序时才执行 main()
if __name__ == '__main__':
    main()