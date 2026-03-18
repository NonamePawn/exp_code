# 文件名: model/FDMFF_openset.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入第三章已经写好的基础底层模块
from .GMPM import GMPM
from .DHFEM import DHFEM
from .DFFM import DFFM


class FDMFF_OpenSet(nn.Module):
    """
    基于联邦学习的跨中心CT图像溯源网络 (第四章：开集双轨专用主模型)

    架构亮点：
    1. 组合了 GMPM, DHFEM, DFFM 模块提取高维物理指纹。
    2. 🌟 [4.2.2 新增] 包含 projection_head，用于计算单位超球面上的软标签对比损失 (SLCLM)。
    3. 🌟 [4.2.3 新增] 包含独立的 K+1 维 classifier，支持解耦微调与开集拒识 (ORCDM)。
    """

    def __init__(self, num_classes=13, use_radimagenet=True, use_raddino=True, fused_dim=1536):
        """
        :param num_classes: 已知 CT 设备类别的数量 K (如 13)
        :param fused_dim: DFFM 模块融合后的特征维度 (例如 CNN的2048 + ViT的768 = 2816)
        """
        super().__init__()

        self.num_known_classes = num_classes

        # =======================================================
        # [基础提取层] 第三章模块直接复用，不破坏原有结构
        # =======================================================
        # 1. 门控多通道预处理模块 (滤除解剖结构，提取高频特征)
        self.gmpm = GMPM()

        # 2. 双流异构特征提取模块 (RadImageNet + RAD-DINO)
        self.dhfem = DHFEM(use_radimagenet=use_radimagenet,
                           use_raddino=use_raddino)

        # 3. 动态特征融合总成模块
        # 💡 [排错提示]: 此处的 DFFM 只需要负责把双流特征融合即可。
        # 如果你原先的 DFFM 内部带有 nn.Linear 分类器，建议在这里仅提取它融合后的特征向量 (fused_feat)。
        self.dffm = DFFM(cnn_dim=2048, vit_dim=768,
                         num_classes=num_classes + 1,  # 预留 K+1 维兼容性
                         use_radimagenet=use_radimagenet,
                         use_raddino=use_raddino)

        # =======================================================
        # [开集解耦层] 第四章核心新增模块
        # =======================================================
        # 4. 投影头 (Projection Head) -> 对应 4.2.2 软标签监督对比学习
        # 将极高维度的融合特征降维到 128 维的潜空间，防止对比学习时的维度灾难
        self.projection_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )

        # 5. 开集分类决策头 (Open-Set Classifier) -> 对应 4.2.3 决策边界寻优
        # 严格指向 K+1 维 (前 K 维为已知类，最后 1 维为未知类)
        self.classifier = nn.Linear(fused_dim, num_classes + 1)

    # =======================================================
    # [前向传播接口族] 供客户端在不同阶段灵活调用
    # =======================================================

    def forward_gmpm(self, x):
        """
        [特征拦截口]：仅执行 GMPM 模块
        目的：供 MixUp 轨道在提取出的 5 通道高频特征上进行线性插值。
        """
        return self.gmpm(x)

    def extract_fused_features(self, x_multi):
        """
        [主干提取器]：接收 GMPM 的输出，走完双流提取与动态融合，获取高阶物理指纹
        :param x_multi: GMPM 输出的 5 通道特征图
        :return: 融合后的高维特征向量 (如 2816 维)
        """
        # 1. 双流分别提取局部和全局特征
        local_feat, global_feat = self.dhfem(x_multi)

        # 2. 动态融合 (请确保你的 DFFM 可以返回融合后的特征向量，而不是直接返回概率)
        out = self.dffm(local_feat, global_feat, return_features=True)

        # 如果 DFFM 默认返回了 (logits, features) 元组，提取 features
        if isinstance(out, tuple):
            fused_feat = out[1]  # 根据你 DFFM 实际的返回值索引进行调整
        else:
            fused_feat = out

        return fused_feat

    def forward_stage1(self, fused_feat):
        """
        [Stage 1 专用]：软标签监督对比学习 (SLCLM)
        目的：将特征输入投影头，并进行 L2 范数归一化，映射至单位超球面。
        """
        # 1. 非线性降维
        z = self.projection_head(fused_feat)
        # 2. L2 归一化 (必须！这是对比学习计算余弦相似度的前提)
        z_normalized = F.normalize(z, p=2, dim=1)

        return z_normalized

    def forward_stage2(self, fused_feat):
        """
        [Stage 2 专用]：开集分类决策微调 (ORCDM)
        目的：在主干特征彻底冻结后，将特征送入线性分类器进行交叉熵寻优。
        """
        logits = self.classifier(fused_feat)
        return logits

    def forward(self, x):
        """
        [端到端完整前向传播]：供 FGSM 对抗生成 (需要对 x 求导) 以及最终测试推理时使用
        """
        # 1. 预处理
        x_multi = self.forward_gmpm(x)
        # 2. 提取深层指纹
        fused_feat = self.extract_fused_features(x_multi)
        # 3. 输出 K+1 维分类结果
        logits = self.forward_stage2(fused_feat)

        return logits