# 文件名: endPoints/client_distill_openset.py
import torch  # 导入PyTorch核心库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入常用的函数接口，如激活函数、独热编码、损失函数等
from torch.utils.data import DataLoader  # 导入数据加载器，用于批量提取数据
from tqdm import tqdm  # 导入进度条工具，方便在终端实时观察训练进度
import numpy as np  # 导入NumPy，用于生成MixUp所需的Beta分布随机数


class OpenSetDistillClient:
    """
    联邦开集双轨客户端类。
    负责本地数据的极化浓度检测，并在 MixUp 轨道和 FGSM 对抗轨道之间智能切换，
    同时兼容第三章的在线知识蒸馏（ODCM）逻辑。
    """

    def __init__(self, client_id, private_dataset, device, batch_size, adv_threshold=0.75):
        self.client_id = client_id  # 当前客户端的唯一标识符（ID）
        self.device = device  # 训练设备，通常是 'cuda' (如你的 A6000 显卡)
        self.batch_size = batch_size  # 每次送入网络的图像批次大小
        self.adv_threshold = adv_threshold  # 极化阈值：某类数据占比超过此值即触发对抗轨道

        # 初始化私有数据加载器，配置多线程和显存固定以加速训练
        self.private_loader = DataLoader(
            private_dataset,  # 客户端本地分配到的非独立同分布 (Non-IID) 数据集
            batch_size=batch_size,  # 按照预设的 Batch Size 划分
            shuffle=True,  # 开启打乱，保证每个 Epoch 网络看到的样本顺序不同
            num_workers=8,  # 开启8个CPU线程去读取硬盘数据，防止GPU等待
            pin_memory=True  # 将数据直接锁在主机的锁页内存中，加速向GPU显存的拷贝
        )

        # 实例化交叉熵损失函数，用于分类任务（原生支持软标签，无需额外设置）
        self.criterion_ce = nn.CrossEntropyLoss()
        # 实例化KL散度损失函数，用于知识蒸馏任务，'batchmean' 表示在 Batch 维度上求平均
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')

    def get_local_class_logits(self, model):
        """
        提取本地私有数据上各个设备类别的平均特征响应 (Logits)。
        这是第三章联邦蒸馏提取“类原型”的核心函数。
        """
        model.eval()  # 将模型切换到评估模式，关闭 Dropout 和 BatchNorm 的随机性
        # model.to(self.device)  # 将服务端下发的全局模型搬运到当前 GPU 显存上

        class_logits_sum = {}  # 字典：用于累加每个类别的输出 Logits
        class_counts = {}  # 字典：用于记录每个类别的样本数量，方便求平均

        with torch.no_grad():  # 关闭梯度计算引擎，大幅节省显存并加速前向传播
            # 遍历本地 DataLoader，附带 tqdm 进度条
            for imgs, labels in tqdm(self.private_loader, desc=f"Client {self.client_id}", leave=False, ncols=80):
                imgs = imgs.to(self.device)  # 将当前 Batch 的 CT 图像搬运到 GPU (维度: [B, 1, 512, 512])

                with torch.cuda.amp.autocast():  # 开启自动混合精度计算 (FP16/FP32 智能切换)
                    out = model(imgs)  # 获得网络输出 (维度: [B, K+1])
                    if isinstance(out, tuple): out = out[0]  # 如果网络返回多项内容，只取第一项 Logits

                # 遍历当前 Batch 中的每一张图的预测结果和真实标签
                for i, label in enumerate(labels):
                    lbl = label.item()  # 取出当前图片的真实类别索引（标量）
                    if lbl not in class_logits_sum:
                        # 如果该类别是第一次出现，克隆结果并放回 CPU (防止积攒在 GPU 导致显存爆炸)
                        class_logits_sum[lbl] = out[i].clone().cpu()
                        class_counts[lbl] = 1  # 计数初始化为 1
                    else:
                        # 如果类别已存在，累加 Logits 向量，并计数 +1
                        class_logits_sum[lbl] += out[i].cpu()
                        class_counts[lbl] += 1

        # 通过字典推导式，用总和除以总数，算出每个类别的平均特征响应（类原型）
        mean_class_logits = {lbl: class_logits_sum[lbl] / class_counts[lbl] for lbl in class_logits_sum}
        # model.cpu()  # 【极其重要】计算完毕后，立刻将模型踢出 GPU 显存，留出空间给接下来的训练
        return mean_class_logits, class_counts  # 返回局部类原型字典和数量字典

    def _check_track_condition(self):
        """
        动态路由机制：扫描本地数据集，统计类别分布浓度，决定走 MixUp 还是 对抗轨道。
        """
        label_counts = {}  # 字典：记录本地拥有的各类别样本总数
        total_samples = 0  # 变量：记录本地总样本数

        # 快速遍历本地 DataLoader（不涉及模型，速度极快）
        for _, labels in self.private_loader:
            for lbl in labels:  # 遍历每个标签
                l = lbl.item()  # 转换为 Python 标量
                # 在字典中累计该类别的数量，如果没有则默认为 0 再加 1
                label_counts[l] = label_counts.get(l, 0) + 1
                total_samples += 1  # 总数加 1

        # 找出数据集中占比最大的那一个类别的具体比例
        max_ratio = max(label_counts.values()) / total_samples

        # 如果最大类别的占比 >= 设定的阈值 (如 75%)
        if max_ratio >= self.adv_threshold:
            # 说明数据极度倾斜（单一设备孤岛），触发对抗生成轨道
            return "Adversarial_Track", max_ratio
        else:
            # 否则说明有多个类别可以混合，触发特征 MixUp 轨道
            return "MixUp_Track", max_ratio

    def _compute_kd_loss(self, out_clean, private_labels, exclusive_class_logits, temperature):
        """
        知识蒸馏（ODCM）辅助函数：计算当前样本与全局虚拟教师的 KL 散度损失。
        """
        teacher_logits_list = []  # 列表：用于存放拼装好的教师信号
        # 遍历当前 Batch 的每一个真实标签
        for i, lbl in enumerate(private_labels):
            lbl_idx = lbl.item()  # 提取标量索引
            # 如果服务端下发了该类别排他性全局类原型
            if lbl_idx in exclusive_class_logits:
                # 就把全局类原型作为该样本的教师信号加入列表
                teacher_logits_list.append(exclusive_class_logits[lbl_idx])
            else:
                # 如果没有，则使用当前模型自己生成的 Logits 作为平替（并切断梯度，且搬到 CPU 统一格式）
                teacher_logits_list.append(out_clean[i].detach().cpu())

        # 将列表中的 B 个张量堆叠成一个完整的 Batch 张量，并搬运到 GPU (维度: [B, K+1])
        teacher_logits = torch.stack(teacher_logits_list).to(self.device)

        # 按照知识蒸馏标准公式：学生输出除以温度T，并取 log_softmax
        student_log_probs = F.log_softmax(out_clean / temperature, dim=1)
        with torch.no_grad():
            # 教师输出除以温度T，并取 softmax (概率化)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

        # 计算 KL 散度损失，并乘以温度平方以保持梯度幅度一致
        loss_kd = self.criterion_kl(student_log_probs, teacher_probs) * (temperature ** 2)
        return loss_kd  # 返回蒸馏 Loss

    def train_mixup_track(self, local_model, optimizer, scaler, epochs, exclusive_class_logits,
                          num_known_classes, kd_alpha, temperature, mix_alpha=0.2):
        """
        【轨道 A】特征级 MixUp 平滑轨道 (极度省显存版：分离计算 + 梯度累加)
        """
        epoch_loss = []
        for epoch in range(epochs):
            batch_loss_list = []
            inner_loader = tqdm(self.private_loader,
                                desc=f"   [Client {self.client_id} | Ep {epoch + 1}/{epochs} | MixUp]", leave=False,
                                ncols=100, colour='blue')

            for imgs, labels in inner_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                total_loss_val = 0.0  # 用于记录合并后的 Loss 值以便打印

                # -------- 第一波：纯净图像前向传播，仅计算知识蒸馏 Loss --------
                with torch.cuda.amp.autocast():
                    out_clean = local_model(imgs)
                    loss_kd = self._compute_kd_loss(out_clean, labels, exclusive_class_logits, temperature)
                    # 按照 kd_alpha 权重对蒸馏 Loss 进行缩放
                    loss_kd_weighted = kd_alpha * loss_kd

                # 反向传播，累加蒸馏的梯度
                # 【关键动作】：这一步反传后，out_clean 产生的庞大计算图会被立刻释放！显存回落。
                scaler.scale(loss_kd_weighted).backward()
                total_loss_val += loss_kd_weighted.item()

                # -------- 第二波：截断式前向传播，进行 MixUp 并计算分类 Loss --------
                with torch.cuda.amp.autocast():
                    # 1. 提取高频特征
                    gmpm_features = local_model.forward_gmpm(imgs)
                    batch_sz = gmpm_features.size(0)

                    # 2. 生成 MixUp 混叠参数
                    index = torch.randperm(batch_sz).to(self.device)
                    lam = np.random.beta(mix_alpha, mix_alpha)

                    # 3. 特征空间混叠
                    mixed_features = lam * gmpm_features + (1 - lam) * gmpm_features[index]

                    # 4. 标签空间混叠 (K+1维)
                    labels_onehot = F.one_hot(labels, num_classes=num_known_classes + 1).float()
                    labels_onehot_idx = F.one_hot(labels[index], num_classes=num_known_classes + 1).float()
                    mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot_idx

                    # 5. 后续网络传播
                    outputs_mixed = local_model.forward_features(mixed_features)
                    loss_ce = self.criterion_ce(outputs_mixed, mixed_labels)

                    # 按照 (1 - kd_alpha) 权重对分类 Loss 进行缩放
                    loss_ce_weighted = (1 - kd_alpha) * loss_ce

                # 反向传播，累加 MixUp 的分类梯度
                scaler.scale(loss_ce_weighted).backward()
                total_loss_val += loss_ce_weighted.item()

                # -------- 最终：两波梯度累加完毕，更新网络权重 --------
                scaler.step(optimizer)
                scaler.update()

                batch_loss_list.append(total_loss_val)
                inner_loader.set_postfix({'loss': f"{total_loss_val:.3f}"})

            epoch_loss.append(sum(batch_loss_list) / len(batch_loss_list))
        return epoch_loss

    def train_adv_track(self, local_model, optimizer, scaler, epochs, exclusive_class_logits,
                        num_known_classes, kd_alpha, temperature, epsilon_range=(0.01, 0.05)):
        """
        【轨道 B】像素级对抗生成轨道 (最终完美版)
        包含两大核心优化：
        1. 实例级动态步长 (Dynamic Epsilon)：覆盖连续的特征偏移光谱，建立厚重拒识缓冲带。
        2. 分离式梯度累加 (Gradient Accumulation)：防止前向传播显存翻倍，保证 24G/48G 显卡安全运行。
        """
        epoch_loss = []
        for epoch in range(epochs):
            batch_loss_list = []
            # 设置终端彩色进度条
            inner_loader = tqdm(self.private_loader,
                                desc=f"   [Client {self.client_id} | Ep {epoch + 1}/{epochs} | Adv]", leave=False,
                                ncols=100, colour='red')

            for imgs, labels in inner_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                # =========================================================
                # 阶段一：在像素空间生成对抗负样本 (FGSM with Dynamic Epsilon)
                # =========================================================
                # 1. 开启原图梯度追踪，准备求导
                imgs.requires_grad = True

                # 2. 关闭混合精度 (autocast) 进行前向传播，防止半精度导致梯度下溢出
                outputs_clean_for_grad = local_model(imgs)

                # 3. 仅对已知的 K 个类别切片求 Loss，寻找破坏已知类边界的最快方向
                loss_for_adv = F.cross_entropy(outputs_clean_for_grad[:, :num_known_classes], labels)

                local_model.zero_grad()
                loss_for_adv.backward()

                # 🌟【核心突变：实例级动态步长】🌟
                # 为当前 Batch 中的每一张图像 (size(0))，在 epsilon_range [0.01, 0.05] 范围内
                # 独立生成一个随机的扰动步长。利用广播机制扩展到 [B, 1, 1, 1] 以便与图像张量相乘。
                dynamic_epsilons = torch.empty(imgs.size(0), 1, 1, 1, device=self.device).uniform_(*epsilon_range)

                # 4. FGSM 公式生成假图：x_adv = x + ε_dynamic * sign(grad)
                adv_imgs = imgs + dynamic_epsilons * imgs.grad.sign()

                # 5. 像素值截断至合法范围，并用 detach() 切断庞大的计算图释放显存！
                adv_imgs = torch.clamp(adv_imgs, 0, 1).detach()
                imgs.requires_grad = False

                # =========================================================
                # 阶段二：分离计算与梯度累加 (彻底解决拼接导致的 OOM 显存翻倍)
                # =========================================================
                optimizer.zero_grad()  # 正式训练前，清空阶段一残留的梯度
                total_loss_val = 0.0  # 累加器，用于记录标量 Loss 并在进度条打印

                # -------- 第一波：只送入真实的干净图像 (Batch Size = B) --------
                with torch.cuda.amp.autocast():
                    out_clean = local_model(imgs)

                    # 计算在线知识蒸馏 Loss (利用辅助函数)
                    loss_kd = self._compute_kd_loss(out_clean, labels, exclusive_class_logits, temperature)
                    # 计算干净图像的常规分类 Loss
                    loss_ce_clean = self.criterion_ce(out_clean, labels)

                    # 第一波总 Loss (乘以 0.5 是因为我们将数据分成了两波计算，保证总体期望与 concat 一致)
                    loss_real = ((1 - kd_alpha) * loss_ce_clean + kd_alpha * loss_kd) * 0.5

                # 【显存释放点】：反向传播累加真实图像梯度，同时底层释放 out_clean 的中间激活图
                scaler.scale(loss_real).backward()
                total_loss_val += loss_real.item()

                # -------- 第二波：只送入对抗假图像 (Batch Size = B) --------
                # 给假图打上第 K+1 类 (未知类) 的统一标签，索引即 num_known_classes
                adv_labels = torch.full((imgs.size(0),), num_known_classes, dtype=torch.long).to(self.device)

                with torch.cuda.amp.autocast():
                    out_adv = local_model(adv_imgs)

                    # 假图只算交叉熵 Loss，逼迫网络学会拒识（不需要计算知识蒸馏）
                    loss_ce_adv = self.criterion_ce(out_adv, adv_labels)

                    # 第二波总 Loss (同样乘以 0.5)
                    loss_fake = ((1 - kd_alpha) * loss_ce_adv) * 0.5

                # 【显存释放点】：反向传播累加假图像梯度
                scaler.scale(loss_fake).backward()
                total_loss_val += loss_fake.item()

                # -------- 最终：两波梯度完美累加，执行参数更新 --------
                scaler.step(optimizer)
                scaler.update()

                # 记录本 Batch 的总 Loss (真实图与假图的平均) 并更新进度条显示
                batch_loss_list.append(total_loss_val)
                inner_loader.set_postfix({'loss': f"{total_loss_val:.3f}"})

            # 计算该 Epoch 所有 Batch Loss 的平均值并存入大列表
            epoch_loss.append(sum(batch_loss_list) / len(batch_loss_list))

        return epoch_loss

    def train_openset_model(self, local_model, exclusive_class_logits, epochs, lr, kd_alpha, temperature,
                            num_known_classes, epsilon_range):
        """第四章双轨统一对外接口"""
        local_model.train()
        optimizer = torch.optim.Adam(local_model.parameters(), lr=lr, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler()

        track_mode, max_ratio = self._check_track_condition()

        if track_mode == "Adversarial_Track":
            epoch_losses = self.train_adv_track(
                local_model, optimizer, scaler, epochs, exclusive_class_logits,
                num_known_classes, kd_alpha, temperature, epsilon_range=epsilon_range  # <--- 在这里透传给对抗轨道
            )
        else:
            epoch_losses = self.train_mixup_track(
                local_model, optimizer, scaler, epochs, exclusive_class_logits,
                num_known_classes, kd_alpha, temperature
                # MixUp 轨道不需要 epsilon_range，所以不用传
            )

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        return local_model.state_dict(), avg_loss