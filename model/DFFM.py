import torch
import torch.nn as nn
import torch.nn.functional as F


class DFFM(nn.Module):
    """
    动态特征融合模块
    输入: local_feat (B, 2048, 16, 16), global_feat (B, 1369, 768)
    输出: logits (B, num_classes)
    """

    def __init__(self, cnn_dim=2048, vit_dim=768, num_classes=13, use_radimagenet=True, use_raddino=True):
        super().__init__()
        self.use_radimagenet = use_radimagenet
        self.use_raddino = use_raddino

        # ==========================================
        # 根据消融开关初始化组件
        # ==========================================
        if self.use_radimagenet and self.use_raddino:
            classifier_in_dim = vit_dim * 2  # 1536

            # 维度对齐层
            self.local_proj = nn.Conv2d(cnn_dim, vit_dim, kernel_size=1)
            # 交叉注意力
            self.attn_l_g = nn.MultiheadAttention(embed_dim=vit_dim, num_heads=8, batch_first=True)
            self.attn_g_l = nn.MultiheadAttention(embed_dim=vit_dim, num_heads=8, batch_first=True)
            self.norm_l = nn.LayerNorm(vit_dim)
            self.norm_g = nn.LayerNorm(vit_dim)

        elif self.use_radimagenet and not self.use_raddino:
            classifier_in_dim = cnn_dim  # 2048

        elif not self.use_radimagenet and self.use_raddino:
            classifier_in_dim = vit_dim  # 768
        else:
            raise ValueError("至少需要开启一个分支！")

        # ==========================================
        # 统一的分类决策头
        # ==========================================
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, local_feat, global_feat):
        # 1. 双分支融合逻辑
        if self.use_radimagenet and self.use_raddino:
            # 空间上采样对齐: 16x16 -> 37x37
            local_aligned = F.interpolate(local_feat, size=(37, 37), mode='bilinear', align_corners=False)
            # 降维并展平: (B, 2048, 37, 37) -> (B, 768, 37, 37) -> (B, 1369, 768)
            local_mapped = self.local_proj(local_aligned)
            local_tokens = local_mapped.flatten(2).transpose(1, 2)

            # 交叉注意力交互
            attn_out_l, _ = self.attn_l_g(query=local_tokens, key=global_feat, value=global_feat)
            attn_out_g, _ = self.attn_g_l(query=global_feat, key=local_tokens, value=local_tokens)

            # 残差连接
            fused_local = self.norm_l(local_tokens + attn_out_l)
            fused_global = self.norm_g(global_feat + attn_out_g)

            # 全局池化与拼接: (B, 1369, 768) -> (B, 768) -> (B, 1536)
            vec_local = fused_local.mean(dim=1)
            vec_global = fused_global.mean(dim=1)
            final_vector = torch.cat([vec_local, vec_global], dim=1)

        # 2. 仅 CNN 逻辑
        elif self.use_radimagenet and not self.use_raddino:
            final_vector = F.adaptive_avg_pool2d(local_feat, (1, 1)).flatten(1)

        # 3. 仅 ViT 逻辑
        elif not self.use_radimagenet and self.use_raddino:
            final_vector = global_feat.mean(dim=1)

        # 分类输出
        logits = self.classifier(final_vector)
        return logits