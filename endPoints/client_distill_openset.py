# 文件名: endPoints/client_distill_openset.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 🌟 引入上一轮我们编写的连续软标签对比损失函数
from utils.loss_openset import SoftLabelSupConLoss


class OpenSetDistillClient:
    def __init__(self, client_id, private_dataset, device, batch_size, adv_threshold=0.75):
        self.client_id = client_id
        self.device = device
        self.batch_size = batch_size
        self.adv_threshold = adv_threshold

        self.private_loader = DataLoader(
            private_dataset, batch_size=batch_size, shuffle=True,
            num_workers=8, pin_memory=True
        )

        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')
        self.criterion_supcon = SoftLabelSupConLoss(temperature=0.07)  # 🌟 4.2.2 专用损失

    def get_local_class_logits(self, model):
        """提取本地 K+1 维类原型"""
        model.eval()
        class_logits_sum = {}
        class_counts = {}
        with torch.no_grad():
            for imgs, labels in tqdm(self.private_loader, desc=f"Client {self.client_id}", leave=False, ncols=80):
                imgs = imgs.to(self.device)
                with torch.cuda.amp.autocast():
                    out = model(imgs)
                    if isinstance(out, tuple): out = out[0]
                for i, label in enumerate(labels):
                    lbl = label.item()
                    if lbl not in class_logits_sum:
                        class_logits_sum[lbl] = out[i].clone().cpu()
                        class_counts[lbl] = 1
                    else:
                        class_logits_sum[lbl] += out[i].cpu()
                        class_counts[lbl] += 1
        mean_class_logits = {lbl: class_logits_sum[lbl] / class_counts[lbl] for lbl in class_logits_sum}
        return mean_class_logits, class_counts

    def _check_track_condition(self):
        label_counts = {}
        total_samples = 0
        for _, labels in self.private_loader:
            for lbl in labels:
                l = lbl.item()
                label_counts[l] = label_counts.get(l, 0) + 1
                total_samples += 1
        max_ratio = max(label_counts.values()) / total_samples
        if max_ratio >= self.adv_threshold:
            return "Adversarial_Track", max_ratio
        else:
            return "MixUp_Track", max_ratio

    def _compute_fv_ocs_loss(self, logits_stage2, labels, global_consensus, temperature):
        """
        🌟 [4.2.4 核心] 联邦投票开集协同损失计算 (对应公式 4-12)
        """
        target_probs_list = []
        for i in range(len(labels)):
            lbl_idx = labels[i].item()
            if lbl_idx in global_consensus:
                # 接收来自服务端的投票共识 P_vote
                target_probs_list.append(global_consensus[lbl_idx].to(self.device))
            else:
                # 如果没有共识，使用本地平滑概率作为平替
                target_probs_list.append(F.softmax(logits_stage2[i].detach() / temperature, dim=0))

        target_probs = torch.stack(target_probs_list)  # [B, K+1]

        # P_local: 本地网络当前微调输出的概率分布
        student_log_probs = F.log_softmax(logits_stage2 / temperature, dim=1)

        # 计算 KL 散度拉近距离
        loss_fv_ocs = self.criterion_kl(student_log_probs, target_probs) * (temperature ** 2)
        return loss_fv_ocs

    # =======================================================
    # [Stage 1 专用轨道]：纯净化特征拓扑重塑 (仅计算 SupCon)
    # =======================================================
    def train_mixup_track(self, local_model, optimizer, scaler, epochs, num_known_classes, mix_alpha=0.2):
        epoch_loss = []
        for epoch in range(epochs):
            batch_loss_list = []
            inner_loader = tqdm(self.private_loader, desc=f"   [Client {self.client_id} | MixUp | SupCon]", leave=False,
                                ncols=100, colour='blue')
            for imgs, labels in inner_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    gmpm_feat = local_model.forward_gmpm(imgs)
                    batch_sz = gmpm_feat.size(0)
                    idx = torch.randperm(batch_sz).to(self.device)
                    lam = np.random.beta(mix_alpha, mix_alpha)

                    # 特征混叠
                    mixed_feat = lam * gmpm_feat + (1 - lam) * gmpm_feat[idx]

                    # 标签混叠 (K+1维连续软标签)
                    labels_oh = F.one_hot(labels, num_classes=num_known_classes + 1).float()
                    labels_oh_idx = F.one_hot(labels[idx], num_classes=num_known_classes + 1).float()
                    mixed_labels = lam * labels_oh + (1 - lam) * labels_oh_idx

                    # 前向传播至潜空间
                    fused = local_model.extract_fused_features(mixed_feat)
                    z_norm = local_model.forward_stage1(fused)

                    # 🌟 严格只使用 SLCLM 损失，彻底消除显存翻倍与梯度冲突
                    loss_supcon = self.criterion_supcon(z_norm, mixed_labels)

                scaler.scale(loss_supcon).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_loss_list.append(loss_supcon.item())
                inner_loader.set_postfix({'loss': f"{loss_supcon.item():.3f}"})
            epoch_loss.append(sum(batch_loss_list) / len(batch_loss_list))
        return epoch_loss

    def train_adv_track(self, local_model, optimizer, scaler, epochs, num_known_classes, epsilon_range):
        epoch_loss = []
        for epoch in range(epochs):
            batch_loss_list = []
            inner_loader = tqdm(self.private_loader, desc=f"   [Client {self.client_id} | Adv | SupCon]", leave=False,
                                ncols=100, colour='red')
            for imgs, labels in inner_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                # --- 1. FGSM 生成假图 ---
                imgs.requires_grad = True
                out_clean = local_model(imgs)  # 借用冻结的分类头获取破坏已知类的梯度
                loss_adv = F.cross_entropy(out_clean[:, :num_known_classes], labels)
                local_model.zero_grad()
                loss_adv.backward()

                dyn_eps = torch.empty(imgs.size(0), 1, 1, 1, device=self.device).uniform_(*epsilon_range)
                adv_imgs = imgs + dyn_eps * imgs.grad.sign()
                adv_imgs = torch.clamp(adv_imgs, 0, 1).detach()
                imgs.requires_grad = False

                # --- 2. 特征层面对比学习 ---
                optimizer.zero_grad()
                combined_imgs = torch.cat([imgs, adv_imgs], dim=0)

                # 构造包含残余置信度(Gamma)的开集锚点软标签
                labels_clean = F.one_hot(labels, num_classes=num_known_classes + 1).float()
                gamma = 0.15
                labels_adv = torch.zeros_like(labels_clean)
                labels_adv.scatter_(1, labels.unsqueeze(1), gamma)
                labels_adv[:, num_known_classes] = 1 - gamma
                combined_labels = torch.cat([labels_clean, labels_adv], dim=0)

                with torch.cuda.amp.autocast():
                    x_multi = local_model.forward_gmpm(combined_imgs)
                    fused = local_model.extract_fused_features(x_multi)
                    z_norm = local_model.forward_stage1(fused)

                    # 🌟 同样，严格只使用 SLCLM 损失
                    loss_supcon = self.criterion_supcon(z_norm, combined_labels)

                scaler.scale(loss_supcon).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_loss_list.append(loss_supcon.item())
                inner_loader.set_postfix({'loss': f"{loss_supcon.item():.3f}"})
            epoch_loss.append(sum(batch_loss_list) / len(batch_loss_list))
        return epoch_loss

    # =======================================================
    # [主训练接口]：控制两阶段解耦时序
    # =======================================================
    def train_openset_model(self, local_model, global_consensus, epochs, lr, kd_alpha, temperature, num_known_classes,
                            epsilon_range):
        local_model.train()
        scaler = torch.cuda.amp.GradScaler()

        # 计算两阶段 Epoch 比例 (假设 70% 塑特征，30% 定边界)
        epochs_stage1 = max(1, int(epochs * 0.7))
        epochs_stage2 = epochs - epochs_stage1

        # ------------------ STAGE 1: 特征拓扑重塑 ------------------
        # 冻结分类器，激活主干与投影头
        for param in local_model.classifier.parameters(): param.requires_grad = False
        for param in local_model.gmpm.parameters(): param.requires_grad = True
        for param in local_model.dhfem.parameters(): param.requires_grad = True
        for param in local_model.dffm.parameters(): param.requires_grad = True
        for param in local_model.projection_head.parameters(): param.requires_grad = True

        optimizer_stage1 = torch.optim.Adam(filter(lambda p: p.requires_grad, local_model.parameters()), lr=lr,
                                            weight_decay=1e-4)

        track_mode, _ = self._check_track_condition()
        if track_mode == "Adversarial_Track":
            loss_stage1 = self.train_adv_track(local_model, optimizer_stage1, scaler, epochs_stage1, num_known_classes,
                                               epsilon_range)
        else:
            loss_stage1 = self.train_mixup_track(local_model, optimizer_stage1, scaler, epochs_stage1,
                                                 num_known_classes)

        # ------------------ STAGE 2: 决策边界微调 ------------------
        # 强制冻结主干网络！仅激活线性分类决策头
        for param in local_model.parameters(): param.requires_grad = False
        for param in local_model.classifier.parameters(): param.requires_grad = True

        optimizer_stage2 = torch.optim.Adam(local_model.classifier.parameters(), lr=lr * 2, weight_decay=1e-4)

        loss_stage2 = []
        for epoch in range(epochs_stage2):
            batch_loss_list = []
            inner_loader = tqdm(self.private_loader, desc=f"   [Client {self.client_id} | Fine-Tune | ORCDM]",
                                leave=False, ncols=100, colour='green')

            for imgs, labels in inner_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer_stage2.zero_grad()

                with torch.cuda.amp.autocast():
                    # 提取固定特征 (主干冻结，无需梯度图，极大加速并省显存)
                    with torch.no_grad():
                        x_multi = local_model.forward_gmpm(imgs)
                        fused_feat = local_model.extract_fused_features(x_multi)

                    # 计算 K+1 维分类响应
                    logits_stage2 = local_model.forward_stage2(fused_feat)

                    # 🌟 1. 计算基本的软目标交叉熵
                    loss_ce = self.criterion_ce(logits_stage2, labels)

                    # 🌟 2. [4.2.4 新增] 计算 FV-OCS 联邦投票开集协同损失
                    loss_fv = self._compute_fv_ocs_loss(logits_stage2, labels, global_consensus, temperature)

                    # 总损失 (对应论文公式 4-12 后面的 L_total)
                    total_loss = loss_ce + kd_alpha * loss_fv

                scaler.scale(total_loss).backward()
                scaler.step(optimizer_stage2)
                scaler.update()

                batch_loss_list.append(total_loss.item())
                inner_loader.set_postfix({'loss': f"{total_loss.item():.3f}"})

            loss_stage2.append(sum(batch_loss_list) / len(batch_loss_list))

        avg_loss = (sum(loss_stage1) + sum(loss_stage2)) / (len(loss_stage1) + len(loss_stage2)) if (
                    loss_stage1 or loss_stage2) else 0
        return local_model.state_dict(), avg_loss