import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =========================================================================
# 第一部分：基于拉普拉斯能量的显著块挖掘 (适配 3 通道输入)
# =========================================================================

class LaplacianPatchSelector:
    """
    输入: (B, 3, H, W)
    逻辑: 为了计算纹理丰富度，我们先将 3 通道 mean 成为单通道灰度图，
          计算拉普拉斯能量，选出位置后，返回原始的 3 通道 Patch。
    """

    def __init__(self, patch_size=64, top_k=8):
        self.patch_size = patch_size
        self.top_k = top_k
        # 拉普拉斯核保持不变 (1, 1, 3, 3)
        kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).unsqueeze(0).unsqueeze(0)
        self.kernel = nn.Parameter(kernel, requires_grad=False)

    def select_patches(self, x):
        """
        输入 x: (B, 3, H, W) 或 (3, H, W) -> 这里的 3 是 dataset 里的 [Adaptive, Noise, Tissue]
        输出: (K, 3, 64, 64)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)  # (1, 3, H, W)

        b, c, h, w = x.shape
        p = self.patch_size

        # 1. 切块 (Unfold)
        # 对 3 个通道同时切分 -> (N, 3, 64, 64)
        # unfold 操作比较 tricky，通常先 reshape 比较好处理，或者分别对 H, W unfold
        patches = x.unfold(2, p, p).unfold(3, p, p)
        # shape: (B, C, Num_H, Num_W, P, P) -> (B, C, Num_Patches, P, P)
        patches = patches.contiguous().view(b, c, -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4)  # (B, Num_Patches, C, P, P)
        patches = patches.reshape(-1, c, p, p)  # (Total_Patches, 3, 64, 64)

        # 2. 计算能量分数 (使用灰度图计算)
        # 将 3 通道平均，变成 (Total_Patches, 1, 64, 64)
        patches_gray = patches.mean(dim=1, keepdim=True)

        with torch.no_grad():
            # 计算梯度: Conv2d (N, 1, 64, 64) -> (N, 1, 64, 64)
            grads = F.conv2d(patches_gray, self.kernel.to(x.device), padding=1)

        # 3. 能量评分: Var(Grad)
        scores = torch.var(grads, dim=(1, 2, 3))  # (Total_Patches, )

        # 4. Top-K 筛选
        k = min(self.top_k, patches.size(0))
        _, top_indices = torch.topk(scores, k)

        # 返回选中的原始 3 通道 Patch
        selected_patches = patches[top_indices]  # (K, 3, 64, 64)

        return selected_patches


# =========================================================================
# 第二部分：双流解耦变分自编码器 (Dual-Stream Disentangled VAE)
# =========================================================================

class ContentEncoder(nn.Module):
    """
    内容编码器 (E_c)
    修改: 输入通道 1 -> 3
    """

    def __init__(self, latent_dim=64):
        super(ContentEncoder, self).__init__()
        self.net = nn.Sequential(
            # [修改点 1] in_channels=3
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_var = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x):
        features = self.net(x)
        mu = self.fc_mu(features)
        logvar = self.fc_var(features)
        return mu, logvar


class StyleEncoder(nn.Module):
    """
    风格编码器 (E_s)
    修改: 输入通道 1 -> 3
    """

    def __init__(self, latent_dim=32):
        super(StyleEncoder, self).__init__()
        self.net = nn.Sequential(
            # [修改点 2] in_channels=3
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_var = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x):
        features = self.net(x)
        mu = self.fc_mu(features)
        logvar = self.fc_var(features)
        return mu, logvar


class Decoder(nn.Module):
    """
    解码器 (D)
    修改: 输出通道 1 -> 3
    """

    def __init__(self, content_dim=64, style_dim=32):
        super(Decoder, self).__init__()
        self.input_dim = content_dim + style_dim

        self.fc = nn.Linear(self.input_dim, 128 * 8 * 8)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # [修改点 3] out_channels=3
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z_content, z_style):
        z = torch.cat([z_content, z_style], dim=1)
        x = self.fc(z)
        x = x.view(-1, 128, 8, 8)
        x_recon = self.net(x)
        return x_recon


class SourceClassifier(nn.Module):
    """
    溯源分类器 (保持不变)
    """

    def __init__(self, style_dim, num_classes):
        super(SourceClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(style_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, z_style):
        return self.net(z_style)


class DualStreamVAE(nn.Module):
    """
    主模型类 (结构逻辑不变)
    """

    def __init__(self, num_classes, content_dim=64, style_dim=32):
        super(DualStreamVAE, self).__init__()

        # 编码器 (现在接受 3 通道)
        self.content_encoder = ContentEncoder(latent_dim=content_dim)
        self.style_encoder = StyleEncoder(latent_dim=style_dim)

        # 解码器 (现在输出 3 通道)
        self.decoder = Decoder(content_dim=content_dim, style_dim=style_dim)

        # 分类器
        self.classifier = SourceClassifier(style_dim=style_dim, num_classes=num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x: (B, 3, 64, 64)

        # A. 内容流
        mu_c, logvar_c = self.content_encoder(x)
        z_c = self.reparameterize(mu_c, logvar_c)

        # B. 风格流
        mu_s, logvar_s = self.style_encoder(x)
        z_s = self.reparameterize(mu_s, logvar_s)

        # C. 重构 (输出也是 3 通道)
        recon_img = self.decoder(z_c, z_s)

        # D. 分类
        pred_logits = self.classifier(z_s)

        return recon_img, pred_logits, (mu_c, logvar_c), (mu_s, logvar_s), (z_c, z_s)

    def get_style_embedding(self, x):
        mu_s, _ = self.style_encoder(x)
        return mu_s


# =========================================================================
# 辅助函数与测试
# =========================================================================

class OrthogonalityLoss(nn.Module):
    def __init__(self):
        super(OrthogonalityLoss, self).__init__()

    def forward(self, z_c, z_s):
        z_c_norm = F.normalize(z_c, dim=1)
        z_s_norm = F.normalize(z_s, dim=1)
        cosine_sim = torch.mean(torch.abs(torch.sum(z_c_norm * z_s_norm, dim=1)))
        return cosine_sim


if __name__ == "__main__":
    # 模拟 Dataset 出来的 3 通道数据
    # Shape: (1, 3, 512, 512)
    raw_ct_image = torch.randn(1, 3, 512, 512)

    # 1. 筛选 Patch (测试多通道筛选逻辑)
    selector = LaplacianPatchSelector(patch_size=64, top_k=8)
    patches = selector.select_patches(raw_ct_image)
    print(f"Selected Patches Shape: {patches.shape}")  # 应为 [8, 3, 64, 64]

    # 2. 模型前向
    model = DualStreamVAE(num_classes=5)
    recon, logits, _, _, _ = model(patches)

    print(f"Reconstruction Shape: {recon.shape}")  # [8, 3, 64, 64]
    print(f"Logits Shape: {logits.shape}")  # [8, 5]
    print("✅ Model Updated for 3-Channel Input.")