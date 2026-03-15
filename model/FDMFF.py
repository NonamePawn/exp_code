import torch
import torch.nn as nn

# 导入你的三大核心模块
from model.GMPM import GMPM
from model.DHFEM import DHFEM
from model.DFFM import DFFM


class FDMFF(nn.Module):
    """
    全流程融合模型 (FDMFF - 单流隔离消融版)
    包含: GMPM (预处理) -> DHFEM (仅启用局部分支) -> DFFM (自注意力融合)
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

        # 2. 双流异构特征提取模块
        self.dhfem = DHFEM(
            in_channels=5,
            use_local_stream=True,
            use_global_stream=False,  # <--- 【硬隔离】这里设为 False，RAD-DINO 完全不实例化，不吃显存不联网！
            medicalnet_pretrained_path=medicalnet_pretrained_path,
            raddino_pretrained_path=raddino_pretrained_path
        )

        # [补充漏掉的降维层]
        self.local_proj = nn.Conv2d(in_channels=2048, out_channels=768, kernel_size=1)

        # 3. 动态特征融合模块
        self.dffm = DFFM(dim=768, num_classes=num_classes)

    def forward(self, x):
        """前向传播"""
        # --- 阶段 1: 物理先验增强预处理 ---
        x_multi = self.gmpm(x)

        # --- 阶段 2: 特征提取 ---
        features_dict = self.dhfem(x_multi)
        local_tokens = features_dict['local_tokens']

        # 因为关了全局流，这里拿到的 global_tokens 是 None
        global_tokens = features_dict['global_tokens']

        # --- 阶段 3: 局部特征降维与展平 ---
        if local_tokens is not None:
            local_tokens = self.local_proj(local_tokens)
            B, C, H, W = local_tokens.shape
            local_tokens = local_tokens.view(B, C, -1).permute(0, 2, 1)

        # ==========================================
        # 阶段 4: 【核心隔离魔术】欺骗融合模块！
        # ==========================================
        if global_tokens is None:
            # 如果没有全局特征，我们直接把局部特征"借"给它。
            # 这样 DFFM 依然接收到了两个 768 维的输入，代码照常运行。
            # 原本的 "Cross-Attention" 就变成了 "Self-Attention"！
            global_tokens = local_tokens

        # --- 阶段 5: 送入融合模块计算分类结果 ---
        logits = self.dffm(local_tokens, global_tokens)

        return logits