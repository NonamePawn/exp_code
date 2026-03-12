import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttentionBlock(nn.Module):
    """
    基础跨域交叉注意力块
    实现了 Q 和 K, V 来自不同特征流的注意力机制
    """

    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5

        # 定义投影矩阵 W_Q, W_K, W_V
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)

    def forward(self, x_q, x_kv):
        """
        x_q: 提供 Query 的特征流 (例如局部特征)
        x_kv: 提供 Key 和 Value 的特征流 (例如全局特征)
        """
        # 生成 Q, K, V
        Q = self.proj_q(x_q)  # (Batch, N_q, dim)
        K = self.proj_k(x_kv)  # (Batch, N_kv, dim)
        V = self.proj_v(x_kv)  # (Batch, N_kv, dim)

        # 计算点积相似度并生成动态权重图 Softmax(Q * K^T / sqrt(d_k))
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 乘以 Value 得到交互后的特征
        out = attn @ V

        # 返回注意力计算结果以及当前分支保留的原始物理特征 V'
        return out, V


class DFFM(nn.Module):
    """
    动态特征融合模块 (Dynamic Feature Fusion Module)
    包含：跨域对齐投影、双向交叉注意力交互、动态特征集成、多级分类决策头
    """

    def __init__(self, dim=768, num_classes=13, dropout_rate=0.5):
        """
        dim: 输入特征的维度 (DHFEM 输出默认是 768)
        num_classes: CT设备类别的数量 (根据大论文数据集表3-3，默认13种型号)
        """
        super().__init__()

        # 1. 双向交叉注意力交互层
        # Local -> Global: 局部从全局获取补充信息
        self.attn_l_from_g = CrossAttentionBlock(dim)
        # Global -> Local: 全局从局部获取补充信息
        self.attn_g_from_l = CrossAttentionBlock(dim)

        # 2. 层归一化 (Layer Normalization)，保持训练稳定性
        self.norm_l = nn.LayerNorm(dim)
        self.norm_g = nn.LayerNorm(dim)

        # 3. 多级分类决策头 (MLP)
        # 融合后的特征维度是 dim * 2 (因为 Concat 拼接了两个分支)
        fused_dim = dim * 2

        # 包含 W1, ReLU (σ), Dropout (掩码m), W2
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.BatchNorm1d(fused_dim // 2),
            nn.ReLU(inplace=True),  # σ
            nn.Dropout(p=dropout_rate),  # ⊙ m (随机掩码向量)
            nn.Linear(fused_dim // 2, num_classes)  # W2
        )

    def forward(self, local_features, global_features):
        """
        local_features: 来自 MedicalNet 的局部特征, 形状 (Batch, N_local, dim)
        global_features: 来自 RAD-DINO 的全局特征, 形状 (Batch, N_global, dim)
        """
        # ==========================================
        # 1. 双向交叉注意力交互 (Bidirectional Cross-Attention)
        # ==========================================

        # Attn_{L <- G}: Q来自Local, K/V来自Global
        attn_l, v_g_prime = self.attn_l_from_g(x_q=local_features, x_kv=global_features)

        # Attn_{G <- L}: Q来自Global, K/V来自Local
        attn_g, v_l_prime = self.attn_g_from_l(x_q=global_features, x_kv=local_features)

        # ==========================================
        # 2. 动态特征集成 (Dynamic Feature Integration)
        # ==========================================

        # 残差连接 + LayerNorm: LN(V_L' + Attn_{L <- G})
        # 注意: 为了维度匹配进行相加，我们使用 local_features (或从它投影出的 V)
        # 此处使用投影后的 v_l_prime 的形状是 (Batch, N_local, dim)
        out_l = self.norm_l(v_l_prime + attn_l)

        # 残差连接 + LayerNorm: LN(V_G' + Attn_{G <- L})
        out_g = self.norm_g(v_g_prime + attn_g)

        # 在送入分类器前，将序列特征压缩为一维向量 (Global Average Pooling)
        out_l_pooled = out_l.mean(dim=1)  # (Batch, dim)
        out_g_pooled = out_g.mean(dim=1)  # (Batch, dim)

        # 在通道维度上进行特征堆叠 Concat
        v_fused = torch.cat([out_l_pooled, out_g_pooled], dim=1)  # (Batch, dim * 2)

        # ==========================================
        # 3. 多级分类决策头 (Multi-level Classification Head)
        # ==========================================

        # 得到最终的设备类别 Logits (P_result 未过 Softmax，以便与交叉熵损失配合)
        logits = self.classifier(v_fused)

        return logits