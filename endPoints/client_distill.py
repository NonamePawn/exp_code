import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

class DistillClient:
    def __init__(self, client_id, private_dataset, device, batch_size):
        self.client_id = client_id
        self.device = device
        self.batch_size = batch_size

        self.private_loader = DataLoader(private_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=8,      # 根据你的 CPU 核心数调整，Linux推荐8，Windows推荐4
                                         pin_memory=True     # 加速数据向 GPU 显存的拷贝
                                        )

        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')

    def get_local_class_logits(self, model):
        """
        [无公共数据版] 提取本地私有数据上各设备类别的平均特征响应 (Class-wise Logits)
        """
        model.eval()
        # 【修复点 4】：需要计算时，才把这个属于自己的模型搬上 GPU
        # model.to(self.device)

        class_logits_sum = {}
        class_counts = {}

        with torch.no_grad():
            # 【修复点 5】：加上内部进度条，你会清晰地看到它处理到了第几个 Batch
            for imgs, labels in tqdm(self.private_loader, desc=f"Client {self.client_id}", leave=False, ncols=80):
                imgs = imgs.to(self.device)
                # 开启混合精度加速推理！让 A6000 飙起来
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

        # 计算每个类别的平均 Logits
        mean_class_logits = {lbl: class_logits_sum[lbl] / class_counts[lbl] for lbl in class_logits_sum}
        # 【极其重要】：计算完毕后，把模型踢回 CPU，并清空 CUDA 缓存，把显存让给下一个客户端！
        # model.cpu()
        # torch.cuda.empty_cache()

        return mean_class_logits, class_counts

    def train_odcm_no_public(self, local_model, exclusive_class_logits, epochs, lr, alpha, temperature):
        local_model.train()
        # local_model.to(self.device)

        # ==========================================
        # 🌟 修改 1: 统一使用 AdamW 优化器 (替代 SGD)
        # ==========================================
        optimizer = torch.optim.AdamW(local_model.parameters(), lr=lr, weight_decay=1e-4)

        epoch_loss = []

        # ==========================================
        # 🌟 修改 2: 设置梯度累加步数与混合精度
        # ==========================================
        accumulation_steps = 4  # 攒够 4 个小 batch 才更新一次权重
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(epochs):
            batch_loss_list = []
            inner_loader = tqdm(self.private_loader,
                                desc=f"   [Client {self.client_id} | Ep {epoch + 1}/{epochs}]",
                                leave=False, ncols=80, colour='green')

            # 🌟 关键：在进入 batch 循环前，先清空一次梯度
            optimizer.zero_grad()

            for i, (private_imgs, private_labels) in enumerate(inner_loader):
                private_imgs, private_labels = private_imgs.to(self.device), private_labels.to(self.device)

                with torch.cuda.amp.autocast():
                    out_private = local_model(private_imgs)
                    if isinstance(out_private, tuple): out_private = out_private[0]

                    loss_ce = self.criterion_ce(out_private, private_labels)

                    # 构造教师信号
                    teacher_logits_list = []
                    for lbl in private_labels:
                        lbl_idx = lbl.item()
                        if lbl_idx in exclusive_class_logits:
                            teacher_logits_list.append(exclusive_class_logits[lbl_idx])
                        else:
                            teacher_logits_list.append(out_private[len(teacher_logits_list)].detach().cpu())

                    teacher_logits = torch.stack(teacher_logits_list).to(self.device)

                    student_log_probs = F.log_softmax(out_private / temperature, dim=1)
                    with torch.no_grad():
                        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

                    loss_kd = self.criterion_kl(student_log_probs, teacher_probs) * (temperature ** 2)

                    # 真实的当前 batch 的总 Loss
                    total_loss = (1 - alpha) * loss_ce + alpha * loss_kd

                    # 🌟 修改 3: 累加机制要求每次计算的 Loss 必须除以累加步数
                    loss_to_accumulate = total_loss / accumulation_steps

                # 🌟 修改 4: 反向传播计算梯度（此时不更新权重！）
                scaler.scale(loss_to_accumulate).backward()

                # 🌟 修改 5: 只有凑够了 accumulation_steps，或者到达了最后一个 batch 时，才进行 step 更新
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(inner_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()  # 更新完后立刻清空梯度，为下一轮累加做准备

                batch_loss_list.append(total_loss.item())
                # 进度条显示真实的 total_loss
                inner_loader.set_postfix({'loss': f"{total_loss.item():.3f}"})

            epoch_loss.append(sum(batch_loss_list) / len(batch_loss_list))

        # local_model.cpu()
        # torch.cuda.empty_cache()  # 视显存情况可放开，但一般不需要

        return local_model.state_dict(), sum(epoch_loss) / len(epoch_loss)