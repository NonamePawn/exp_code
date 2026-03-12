import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================
# 1. 手工特征提取分支 (Hand-crafted Features)
# ==========================================

class GaussianHighPass(nn.Module):
    """
    高通滤波器：利用5x5二维高斯平滑核提取空域高频残差特征
    """

    def __init__(self, kernel_size=5, sigma=1.0):
        super().__init__()
        # 生成二维高斯核
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        # 将核转换为 PyTorch 张量，且不需要计算梯度 (固定物理先验)
        self.weight = nn.Parameter(kernel.unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.padding = kernel_size // 2

    def _create_gaussian_kernel(self, kernel_size, sigma):
        # 根据高斯公式生成核
        grid = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        x, y = torch.meshgrid(grid, grid, indexing='ij')
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()  # 归一化
        return kernel

    def forward(self, x):
        # 提取平滑的低频解剖结构 I_low
        low_freq = F.conv2d(x, self.weight.to(x.device), padding=self.padding)
        # 原图减去低频，得到空域高频残差 I_spatial
        return x - low_freq


class HaarWaveletHighPass(nn.Module):
    """
    小波变换器：利用 Haar 小波去除低频 (LL=0)，仅保留高频重建图像
    """

    def __init__(self):
        super().__init__()
        # Haar小波的 LL(低频) 滤波器本质上是 2x2 的均值滤波
        # 这里用一种巧妙的方法：原图 - 低频重建图 = 高频重建图
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # 获取低频分量 (下采样)
        ll = self.avg_pool(x)
        # 将低频分量重建回原图大小 (上采样，相当于 IDWT 的低频部分)
        ll_reconstructed = F.interpolate(ll, scale_factor=2, mode='nearest')
        # 获取高频细节：用原图减去低频重建图
        return x - ll_reconstructed


class PolarArtifactDetector(nn.Module):
    """
    极坐标伪影检测器：转换极坐标 -> 一维高斯平滑 -> 减去低频 -> 逆变换
    由于纯 PyTorch 写极坐标变换非常复杂，实际工程中通常直接用一维卷积模拟环状伪影过滤。
    这里为您提供一个简化的可微物理约束近似版本。
    (注：若需完美还原大论文，建议后续 pip install kornia 并使用 kornia.geometry.transform.warp_polar)
    """

    def __init__(self):
        super().__init__()
        # 简化版：用1x15和15x1的正交卷积近似提取条纹特征
        self.conv_theta = nn.Conv2d(1, 1, kernel_size=(1, 15), padding=(0, 7), bias=False)
        self.conv_theta.weight.requires_grad = False
        nn.init.constant_(self.conv_theta.weight, 1 / 15.0)

    def forward(self, x):
        # 近似低频环状背景
        low_freq = self.conv_theta(x)
        return x - low_freq


# ==========================================
# 2. 动态滤波补偿分支 (Dynamic Filter)
# ==========================================

class AdaptiveConstrainedFilter(nn.Module):
    """
    自适应滤波分支：5x5约束卷积核。中心被强制为-1，外围权重和被强制归一化为1
    """

    def __init__(self, kernel_size=5):
        super().__init__()
        # 定义可学习的权重参数 w
        self.weight = nn.Parameter(torch.randn(1, 1, kernel_size, kernel_size))
        self.padding = kernel_size // 2

    def forward(self, x):
        w = self.weight
        c = w.size(2) // 2  # 中心坐标 (如5x5的中心是2)

        # 1. 复制权重以避免原地操作报错
        w_mask = torch.ones_like(w)
        w_mask[:, :, c, c] = 0.0  # 将中心位置清零

        # 2. 提取外围权重并强制归一化 (外围和为1)
        w_outer = w * w_mask
        sum_outer = w_outer.sum(dim=(2, 3), keepdim=True) + 1e-8  # 加上1e-8防止除以0
        w_norm = w_outer / sum_outer

        # 3. 将中心权重强制设为 -1
        w_norm = w_norm.clone()  # 确保梯度计算正常
        w_norm[:, :, c, c] = -1.0

        # 4. 执行卷积
        return F.conv2d(x, w_norm, padding=self.padding)


# ==========================================
# 3. 通道门控网络与总模块组装 (Gating & Main)
# ==========================================

class GMPM(nn.Module):
    """
    门控多通道预处理模块 (GMPM)
    您可以根据需要，在初始化时通过传入 False 来关闭某个分支，完美支持大论文 3.3.5 节的消融实验。
    """

    def __init__(self,
                 use_highpass=True,
                 use_wavelet=True,
                 use_polar=True,
                 use_adaptive=True,
                 use_original=True):
        super().__init__()

        self.use_highpass = use_highpass
        self.use_wavelet = use_wavelet
        self.use_polar = use_polar
        self.use_adaptive = use_adaptive
        self.use_original = use_original

        # 初始化各个分支
        if self.use_highpass: self.highpass = GaussianHighPass()
        if self.use_wavelet: self.wavelet = HaarWaveletHighPass()
        if self.use_polar: self.polar = PolarArtifactDetector()
        if self.use_adaptive: self.adaptive = AdaptiveConstrainedFilter()

        # 计算激活的分支数量，决定通道门控网络的输入维度
        self.num_branches = sum([use_highpass, use_wavelet, use_polar, use_adaptive, use_original])

        # 通道门控网络：GAP -> W1 -> ReLU -> W2 -> Softmax
        if self.num_branches > 1:
            hidden_dim = max(2, self.num_branches // 2)
            self.gating_network = nn.Sequential(
                nn.Linear(self.num_branches, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.num_branches),
                nn.Softmax(dim=1)  # 保证权重之和为1
            )

    def forward(self, x):
        # 假设输入 x 的形状为 (Batch, 1, H, W) 单通道灰度图
        features = []

        # 1. 获取各分支的特征图
        if self.use_highpass: features.append(self.highpass(x))
        if self.use_wavelet: features.append(self.wavelet(x))
        if self.use_polar: features.append(self.polar(x))
        if self.use_adaptive: features.append(self.adaptive(x))
        if self.use_original: features.append(x)  # 原图拼接分支

        # 在通道维度上拼接：形状变为 (Batch, num_branches, H, W)
        X_concat = torch.cat(features, dim=1)

        if self.num_branches == 1:
            return X_concat

        # 2. 通道门控网络计算权重
        # 全局平均池化 (GAP)，将 (B, C, H, W) 压缩为 (B, C)
        z = F.adaptive_avg_pool2d(X_concat, (1, 1)).view(x.size(0), -1)
        # 获取各通道门控权重 s，形状 (B, C)
        s = self.gating_network(z)
        # 调整权重维度为 (B, C, 1, 1) 以便利用广播机制与特征图相乘
        s = s.unsqueeze(-1).unsqueeze(-1)

        # 3. 动态加权特征
        X_weighted = X_concat * s

        # 大论文中提到："修改了第一层卷积的输入通道数，使其与GMPM输出的通道数量对齐"
        # 这意味着我们将输出加权后的 5 通道张量给下游双流网络，而不是把它简单求和成 1 通道。
        return X_weighted