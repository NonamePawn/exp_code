import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class DistillClient:
    def __init__(self, client_id, private_dataset, device, batch_size):
        self.client_id = client_id
        self.device = device
        self.batch_size = batch_size

        # 现在只有私有数据加载器，完全没有 public_dataset
        self.private_loader = DataLoader(private_dataset, batch_size=batch_size, shuffle=True)

        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')

    def get_local_class_logits(self, model):
        """
        [无公共数据版] 提取本地私有数据上各设备类别的平均特征响应 (Class-wise Logits)
        """
        model.eval()
        model.to(self.device)

        class_logits_sum = {}
        class_counts = {}

        with torch.no_grad():
            for imgs, labels in self.private_loader:
                imgs = imgs.to(self.device)
                out = model(imgs)
                if isinstance(out, tuple): out = out[0]

                # 按类别累加 Logits
                for i, label in enumerate(labels):
                    lbl = label.item()
                    if lbl not in class_logits_sum:
                        class_logits_sum[lbl] = out[i].clone().cpu()
                        class_counts[lbl] = 1
                    else:
                        class_logits_sum[lbl] += out[i].cpu()
                        class_counts[lbl] += 1

        # 计算每个类别的平均 Logits
        mean_class_logits = {lbl: class_logits_sum[lbl] / class_counts[lbl] for lbl in class_logits_sum}

        return mean_class_logits, class_counts

    def train_odcm_no_public(self, local_model, exclusive_class_logits, epochs, lr, alpha, temperature):
        """
        [无公共数据版] 在本地私有数据上同时进行分类和蒸馏训练
        """
        local_model.train()
        local_model.to(self.device)

        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        epoch_loss = []

        for epoch in range(epochs):
            batch_loss_list = []

            for private_imgs, private_labels in self.private_loader:
                private_imgs, private_labels = private_imgs.to(self.device), private_labels.to(self.device)
                optimizer.zero_grad()

                # --- 1. 获取本地模型预测 ---
                out_private = local_model(private_imgs)
                if isinstance(out_private, tuple): out_private = out_private[0]

                # 任务A：交叉熵分类损失 (Hard Label)
                loss_ce = self.criterion_ce(out_private, private_labels)

                # --- 2. 构造对应的教师信号进行蒸馏 ---
                teacher_logits_list = []
                for lbl in private_labels:
                    lbl_idx = lbl.item()
                    # 如果服务端发来了该类别的全局知识，则向其学习；否则用自己的预测作为占位符避免报错
                    if lbl_idx in exclusive_class_logits:
                        teacher_logits_list.append(exclusive_class_logits[lbl_idx])
                    else:
                        # 极端孤岛情况：全局没有这个类，自己跟自己学(即KD损失不产生额外影响)
                        teacher_logits_list.append(out_private[len(teacher_logits_list)].detach().cpu())

                teacher_logits = torch.stack(teacher_logits_list).to(self.device)

                # 任务B：Logits 软标签对齐 (Soft Label)
                student_log_probs = F.log_softmax(out_private / temperature, dim=1)
                with torch.no_grad():
                    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

                loss_kd = self.criterion_kl(student_log_probs, teacher_probs) * (temperature ** 2)

                # 损失融合
                total_loss = (1 - alpha) * loss_ce + alpha * loss_kd

                total_loss.backward()
                optimizer.step()
                batch_loss_list.append(total_loss.item())

            epoch_loss.append(sum(batch_loss_list) / len(batch_loss_list))

        return local_model.state_dict(), sum(epoch_loss) / len(epoch_loss)