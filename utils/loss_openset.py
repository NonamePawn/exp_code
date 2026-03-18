# 文件名: utils/loss_openset.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftLabelSupConLoss(nn.Module):
    """
    4.2.2 软标签监督对比学习损失函数 (SLCLM)
    将 K+1 维软标签转化为几何拓扑拉力，拉近已知类，排斥虚拟退化样本。
    """

    def __init__(self, temperature=0.07):
        super(SoftLabelSupConLoss, self).__init__()
        self.temperature = temperature  # 温度系数 τ

    def forward(self, features, soft_labels):
        """
        :param features: 投影后的特征向量 z_i, 形状 [Batch, 128], 且必须已经经过 L2 归一化
        :param soft_labels: 包含原图与假图的软标签 y, 形状 [Batch, K+1]
        """
        device = features.device
        batch_size = features.shape[0]

        # =======================================================
        # 1. 计算连续亲和力权重矩阵 W (对应论文公式 4-6)
        # W_{i,j} = \sum y_{i,k} * y_{j,k} (通过矩阵乘法极其高效地实现)
        # =======================================================
        W = torch.matmul(soft_labels, soft_labels.T)  # 形状 [B, B]

        # 屏蔽对角线 (自己和自己的亲和力不参与对比计算)
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        W.masked_fill_(mask, 0.0)

        # =======================================================
        # 2. 计算特征在单位超球面上的余弦相似度矩阵
        # =======================================================
        # 因为 features 已经 L2 归一化，点积即为余弦相似度
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        # 数值稳定性优化 (Log-Sum-Exp Trick)，防止 exp() 溢出
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # =======================================================
        # 3. 计算对比对数概率 (对应论文公式 4-7 中的 log 部分)
        # =======================================================
        exp_logits = torch.exp(logits)
        exp_logits.masked_fill_(mask, 0.0)  # 分母中不包含自身
        denominator = exp_logits.sum(dim=1, keepdim=True)

        # log_prob = z_i·z_j / τ - log(Σ exp(z_i·z_a / τ))
        log_prob = logits - torch.log(denominator + 1e-8)

        # =======================================================
        # 4. 计算加权的 SupCon-ST 损失
        # =======================================================
        # 计算每个样本的分母：所有其他样本对它的亲和力之和
        W_sum = W.sum(dim=1, keepdim=True)
        W_sum = torch.clamp(W_sum, min=1e-8)  # 防止除以 0

        # 公式 4-7: L(i) = - (1 / W_sum) * Σ (W_{i,j} * log_prob)
        loss_i = - (W * log_prob).sum(dim=1) / W_sum.squeeze()

        # 返回 Batch 的平均损失
        return loss_i.mean()