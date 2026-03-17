# 文件名: model/FDMFF_openset.py
import torch
import torch.nn as nn
from .GMPM import GMPM
from .DHFEM import DHFEM
from .DFFM import DFFM


class FDMFF_OpenSet(nn.Module):
    """
    基于联邦学习的跨中心CT图像溯源网络 (第四章：开集双轨专用主模型)
    采用模块组合方式构建，与闭集 FDMFF 完全解耦
    """

    def __init__(self, num_classes=13, use_radimagenet=True, use_raddino=True):
        super().__init__()

        self.num_known_classes = num_classes

        # 1. 预处理模块
        self.gmpm = GMPM()

        # 2. 特征提取总成模块
        self.dhfem = DHFEM(use_radimagenet=use_radimagenet,
                           use_raddino=use_raddino)

        # 3. 动态特征融合总成模块 (核心改动：分类头扩维到 K+1，用于容纳未知类锚点)
        self.dffm = DFFM(cnn_dim=2048, vit_dim=768,
                         num_classes=num_classes + 1,  # 直接在这里 +1，极其干净
                         use_radimagenet=use_radimagenet,
                         use_raddino=use_raddino)

    def forward_gmpm(self, x):
        """
        阶段一前向传播：仅执行 GMPM 模块
        目的：供 MixUp 轨道在 5 通道高频特征上进行线性插值拦截
        """
        return self.gmpm(x)

    def forward_features(self, x_multi):
        """
        阶段二前向传播：接收 5 通道特征，走完后续的双流提取与融合分类
        目的：接收 MixUp 混合后的特征，计算损失
        """
        local_feat, global_feat = self.dhfem(x_multi)
        logits = self.dffm(local_feat, global_feat)
        return logits

    def forward(self, x):
        """
        完整的端到端前向传播
        目的：供对抗合成轨道 (FGSM) 和正常推理时使用
        """
        x_multi = self.forward_gmpm(x)
        logits = self.forward_features(x_multi)
        return logits