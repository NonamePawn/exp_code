import torch
import torch.nn as nn
from .GMPM import GMPM
from .DHFEM import DHFEM
from .DFFM import DFFM


class FDMFF(nn.Module):
    """
    基于联邦学习的跨中心CT图像溯源网络 (全流程主模型)
    """

    def __init__(self, num_classes=13, use_radimagenet=True, use_raddino=True):
        super().__init__()

        # 1. 预处理模块 (负责 1通道 -> 5通道 的扩展)
        self.gmpm = GMPM()

        # 2. 特征提取总成模块
        self.dhfem = DHFEM(use_radimagenet=use_radimagenet,
                           use_raddino=use_raddino)

        # 3. 动态特征融合总成模块
        self.dffm = DFFM(cnn_dim=2048, vit_dim=768, num_classes=num_classes,
                         use_radimagenet=use_radimagenet,
                         use_raddino=use_raddino)

    def forward(self, x):
        """
        全流程数据流转
        输入:
            x (Tensor): 原始 DataLoader 给出的单通道切片，维度 (B, 1, 512, 512)
        输出:
            logits (Tensor): 分类预测结果，维度 (B, num_classes)
        """

        # 步骤 1: 预处理与通道增强
        # (B, 1, 512, 512) -> (B, 5, 512, 512)
        x_multi = self.gmpm(x)

        # 步骤 2: 双流异构特征提取
        # local_feat: (B, 2048, 16, 16) | global_feat: (B, 1369, 768)
        local_feat, global_feat = self.dhfem(x_multi)

        # 步骤 3: 维度对齐、交叉融合与溯源分类
        # logits: (B, num_classes)
        logits = self.dffm(local_feat, global_feat)

        return logits