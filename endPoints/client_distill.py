import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import cycle  # 用于循环读取公共数据


class DistillClient:
    def __init__(self, client_id, private_dataset, public_dataset, device, batch_size):
        self.client_id = client_id
        self.device = device
        self.batch_size = batch_size

        # 1. 私有数据加载器 (Private Data): 用于学习硬标签 (Hard Label)
        self.private_loader = DataLoader(private_dataset, batch_size=batch_size, shuffle=True)

        # 2. 公共数据加载器 (Public Data): 用于对齐知识 (Soft Label)
        # 所有的 Client 使用相同的公共数据集
        self.public_loader = DataLoader(public_dataset, batch_size=batch_size, shuffle=True)

        # 损失函数
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')

    def train_distill(self, student_model, teacher_model, epochs, lr, alpha, temperature):
        """
        [核心] 联邦蒸馏训练函数

        参数:
            student_model: 本地待更新的模型 (Student)
            teacher_model: 下发的全局模型 (Teacher) - 固定参数
            alpha:         蒸馏权重 (0=只看私有数据, 1=只模仿老师)
            temperature:   蒸馏温度 (T越大，分布越平缓，能学到更多暗知识)
        """
        student_model.train()
        student_model.to(self.device)

        # 老师模型必须冻结 (只提供知识，不更新)
        teacher_model.eval()
        teacher_model.to(self.device)

        optimizer = torch.optim.SGD(student_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        epoch_loss = []

        # 创建公共数据的无限迭代器
        # 原因：私有数据跑完一轮时，公共数据可能还没跑完或早就跑完了，cycle 保证总能取到数据
        public_iter = cycle(self.public_loader)

        for epoch in range(epochs):
            batch_loss_list = []

            # 主循环：以私有数据的一轮 (Epoch) 为基准
            for private_imgs, private_labels in self.private_loader:
                private_imgs, private_labels = private_imgs.to(self.device), private_labels.to(self.device)

                # 从循环迭代器中取出一批公共数据
                public_imgs, _ = next(public_iter)
                public_imgs = public_imgs.to(self.device)

                optimizer.zero_grad()

                # ---------------------------------------
                # Task 1: 私有数据上的分类学习 (Private Loss)
                # ---------------------------------------
                student_out_private = student_model(private_imgs)
                if isinstance(student_out_private, tuple): student_out_private = student_out_private[0]

                loss_ce = self.criterion_ce(student_out_private, private_labels)

                # ---------------------------------------
                # Task 2: 公共数据上的知识蒸馏 (Public KD Loss)
                # ---------------------------------------
                # 学生在公共考卷上的答案
                student_out_public = student_model(public_imgs)
                if isinstance(student_out_public, tuple): student_out_public = student_out_public[0]

                # 老师在公共考卷上的答案 (标准答案)
                with torch.no_grad():
                    teacher_out_public = teacher_model(public_imgs)
                    if isinstance(teacher_out_public, tuple): teacher_out_public = teacher_out_public[0]

                # 计算 KL 散度：让学生的 Logits 分布趋近于老师
                # 公式：KL( log_softmax(Student/T), softmax(Teacher/T) )
                loss_kd = self.criterion_kl(
                    F.log_softmax(student_out_public / temperature, dim=1),
                    F.softmax(teacher_out_public / temperature, dim=1)
                ) * (temperature ** 2)

                # ---------------------------------------
                # 总损失结合
                # ---------------------------------------
                total_loss = (1 - alpha) * loss_ce + alpha * loss_kd

                total_loss.backward()
                optimizer.step()

                batch_loss_list.append(total_loss.item())

            epoch_loss.append(sum(batch_loss_list) / len(batch_loss_list))

        return student_model.state_dict(), sum(epoch_loss) / len(epoch_loss)