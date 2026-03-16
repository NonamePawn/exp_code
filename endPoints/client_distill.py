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
        model.to(self.device)

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
        model.cpu()
        # torch.cuda.empty_cache()

        return mean_class_logits, class_counts

    def train_odcm_no_public(self, local_model, exclusive_class_logits, epochs, lr, alpha, temperature):
        local_model.train()
        local_model.to(self.device)

        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        epoch_loss = []

        # ==========================================
        # 🌟 新增：初始化 AMP 梯度缩放器
        # ==========================================
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(epochs):
            batch_loss_list = []
            inner_loader = tqdm(self.private_loader,
                                desc=f"   [Client {self.client_id} | Ep {epoch + 1}/{epochs}]",
                                leave=False, ncols=80, colour='green')

            for private_imgs, private_labels in inner_loader:
                private_imgs, private_labels = private_imgs.to(self.device), private_labels.to(self.device)
                optimizer.zero_grad()

                # ==========================================
                # 🌟 新增：使用 autocast 包裹前向传播与 Loss 计算
                # ==========================================
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

                    total_loss = (1 - alpha) * loss_ce + alpha * loss_kd

                # ==========================================
                # 🌟 新增：使用 Scaler 替代直接的 backward()
                # ==========================================
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_loss_list.append(total_loss.item())
                # 在进度条后缀实时更新当前 loss，看着数字跳动更有安全感！
                inner_loader.set_postfix({'loss': f"{total_loss.item():.3f}"})
            epoch_loss.append(sum(batch_loss_list) / len(batch_loss_list))

        local_model.cpu()
        # torch.cuda.empty_cache()

        return local_model.state_dict(), sum(epoch_loss) / len(epoch_loss)