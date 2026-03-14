import torch
import torch.nn as nn

# 导入你的三大核心模块
from model.GMPM import GMPM
from model.DHFEM import DHFEM
from model.DFFM import DFFM


class FDMFF(nn.Module):
    """
    全流程融合模型 (FDMFF - 暂定名)
    包含: GMPM (预处理) -> DHFEM (双流特征提取) -> DFFM (动态特征融合与分类)
    """

    def __init__(self, num_classes=13, medicalnet_pretrained_path=None, raddino_pretrained_path=None):
        super(FDMFF, self).__init__()

        # 1. 门控多通道预处理模块 (1通道输入 -> 5通道输出)
        self.gmpm = GMPM(
            use_highpass=True,
            use_wavelet=True,
            use_polar=True,
            use_adaptive=True,
            use_original=True
        )

        # 2. 双流异构特征提取模块 (接收 5通道输入)
        self.dhfem = DHFEM(
            in_channels=5,
            use_local_stream=True,
            use_global_stream=False,
            medicalnet_pretrained_path=medicalnet_pretrained_path,
            raddino_pretrained_path = raddino_pretrained_path
        )
        # 3. 动态特征融合模块 (接收 768 维 Token，输出分类结果)
        # 默认 DHFEM 输出维度是 768
        self.dffm = DFFM(dim=768, num_classes=num_classes)

    def forward(self, x):
        """
        前向传播逻辑
        输入 x: (Batch, 1, H, W) 单通道灰度张量
        """
        # --- 阶段 1: 物理先验增强预处理 ---
        # 输出形状: (Batch, 5, H, W)
        x_multi = self.gmpm(x)

        # --- 阶段 2: 双流异构特征提取 ---
        # 返回字典包含 local_tokens, global_cls, global_tokens
        features_dict = self.dhfem(x_multi)

        local_tokens = features_dict['local_tokens']  # (Batch, N_local, 768)
        global_tokens = features_dict['global_tokens']  # (Batch, N_global, 768)

        # --- 新增的转换逻辑开始 ---
        # 1. 通道降维: (Batch, 2048, 16, 16) -> (Batch, 768, 16, 16)
        local_tokens = self.local_proj(local_tokens)

        # 2. 展平为 Token 序列: 把 16x16 的空间拉直成 256 个 Token
        B, C, H, W = local_tokens.shape
        # reshape 后变成 (Batch, 768, 256)，然后再 permute 翻转维度变成 (Batch, 256, 768)
        local_tokens = local_tokens.view(B, C, -1).permute(0, 2, 1)
        # --- 新增的转换逻辑结束 ---


        # 3. 此时双流的维度完美对齐，送入融合模块！
        logits = self.dffm(local_tokens, global_tokens)
        return logits