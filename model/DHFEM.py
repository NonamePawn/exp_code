import torch
import torch.nn as nn
import torchvision.models as models
import timm  # 用于加载 Transformer (ViT) 模型
import os
import torch.nn.functional as F
from transformers import ViTModel
# 设置国内镜像，防止服务器下载超时
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# ==========================================
# 1. 局部纹理流: MedicalNet 2D 重构版本 (内嵌 3D->2D 权重转换)
# ==========================================

class MedicalNet2D_Wrapper(nn.Module):
    """
    MedicalNet (23 datasets) 适配器: 将 3D ResNet 权重转换为 2D 并自动进行 5 通道适配
    """

    def __init__(self, base_model_name='resnet50', in_channels=5, pretrained_path=None):
        super().__init__()
        self.in_channels = in_channels

        # 1. 实例化一个标准的 2D ResNet (空壳)
        if base_model_name == 'resnet50':
            self.backbone = models.resnet50(weights=None)
            self.feature_dim = 2048
        elif base_model_name == 'resnet18':
            self.backbone = models.resnet18(weights=None)
            self.feature_dim = 512
        else:
            raise ValueError("不支持的 ResNet 版本，请选择 resnet18 或 resnet50")

        # 2. 修改第一层卷积以接受 5 通道输入
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        # 3. 加载并转换 MedicalNet 3D 权重
        if pretrained_path is not None:
            self._load_3d_pretrained_weights(pretrained_path)

    def _load_3d_pretrained_weights(self, pth_path):
        print(f"====== 开始为 MedicalNet2D 加载并转换 3D 预训练权重 ======")
        checkpoint = torch.load(pth_path, map_location='cpu')

        # 兼容不同保存格式
        state_dict_3d = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        model_state_dict = self.backbone.state_dict()
        new_state_dict = {}
        loaded_layers = 0

        for key, weight_3d in state_dict_3d.items():
            # MedicalNet 的 key 通常带有 'module.' 前缀，需要去掉以对齐 2D ResNet
            key_2d = key.replace('module.', '')

            if key_2d in model_state_dict:
                weight_2d = weight_3d

                # 【核心手术 1】：将 3D 卷积核 (out, in, D, H, W) 压缩为 2D 卷积核 (out, in, H, W)
                if len(weight_3d.shape) == 5:
                    weight_2d = weight_3d.mean(dim=2)

                # 【核心手术 2 - 致命Bug已修复】：
                # 严格限制：只修改名字叫 'conv1.weight' 且 名字里绝不包含 'layer' 的最外层卷积！
                # 这样就保护了深层网络 (如 layer1.0.conv1.weight) 不被错误地拍扁成 5 通道。
                if 'conv1.weight' in key_2d and 'layer' not in key_2d and weight_2d.shape[1] != self.in_channels:
                    print(f"    -> 正在对齐最外层 {key_2d} 的输入通道: {weight_2d.shape[1]} -> {self.in_channels}")
                    # 取均值并复制成 5 份
                    mean_weight = weight_2d.mean(dim=1, keepdim=True)
                    weight_2d = mean_weight.repeat(1, self.in_channels, 1, 1)

                if weight_2d.shape == model_state_dict[key_2d].shape:
                    new_state_dict[key_2d] = weight_2d
                    loaded_layers += 1
            else:
                pass  # 忽略维度或名称不匹配的废弃层 (比如全连接层)

        self.backbone.load_state_dict(new_state_dict, strict=False)
        print(f"====== MedicalNet 权重装载完成！共完美转换 {loaded_layers} 个张量 ======")

    def forward(self, x):
        # ResNet 前向传播，去掉最后的全连接层和全局池化层，只保留特征图提取局部 token
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # 输出的形状是 (Batch, 2048, H', W')
        # 这些将作为局部流的 Patch Tokens 喂给后续的融合模块
        return x

# ==========================================
# 2. 全局关联流: RAD-DINO (ViT) 适配器
# ==========================================

class RADDINO_Wrapper(nn.Module):
    """
    RAD-DINO 适配器 (使用官方 Transformers 库，并自动进行 5 通道手术)
    """

    def __init__(self, in_channels=5, pretrained_path=None):
        super().__init__()
        self.in_channels = in_channels

        print("====== 正在通过 transformers 加载官方 RAD-DINO ======")
        # 直接从 HuggingFace 镜像拉取 RAD-DINO 的结构和纯净权重
        self.vit = ViTModel.from_pretrained('microsoft/rad-dino', add_pooling_layer=False)
        # -----------------------------------------------------------------
        # 【刚刚新增的 2 行修复代码】：修改 HF 底层的说明书，撤销写死的 3 通道检查
        self.vit.config.num_channels = self.in_channels
        self.vit.embeddings.patch_embeddings.num_channels = self.in_channels
        # -----------------------------------------------------------------
        # 进行 5 通道手术
        original_conv = self.vit.embeddings.patch_embeddings.projection

        # 1. 制造一个全新的 5 通道卷积层
        new_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding
        )

        # 2. 将官方预训练的单通道权重取均值，复制成 5 份注入新卷积核
        with torch.no_grad():
            mean_weight = original_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(mean_weight.repeat(1, self.in_channels, 1, 1))
            if original_conv.bias is not None:
                new_conv.bias.copy_(original_conv.bias)

        # 3. 替换掉 ViT 内部的旧层
        self.vit.embeddings.patch_embeddings.projection = new_conv
        print("====== RAD-DINO 5通道适配与加载完美完成！======")

    def forward(self, x):
        # 专门为 ViT 缩放尺寸到 224x224
        x_resized = F.interpolate(x, size=(518, 518), mode='bilinear', align_corners=False)

        # transformers 的输出结构是一个对象，我们需要提取 last_hidden_state
        outputs = self.vit(pixel_values=x_resized)
        tokens = outputs.last_hidden_state  # 形状: (Batch, N, 768)

        cls_token = tokens[:, 0, :]  # (Batch, 768)
        patch_tokens = tokens[:, 1:, :]  # (Batch, N-1, 768)

        return cls_token, patch_tokens

# ==========================================
# 3. 双流异构特征提取总模块 (DHFEM)
# ==========================================

class DHFEM(nn.Module):
    """
    双流异构特征提取模块
    """

    def __init__(self, in_channels=5, use_local_stream=True, use_global_stream=True,
                 medicalnet_pretrained_path=None, raddino_pretrained_path=None):
        super().__init__()

        self.use_local_stream = use_local_stream
        self.use_global_stream = use_global_stream

        if self.use_local_stream:
            self.local_stream = MedicalNet2D_Wrapper(
                in_channels=in_channels,
                pretrained_path=medicalnet_pretrained_path
            )

        if self.use_global_stream:
            self.global_stream = RADDINO_Wrapper(
                in_channels=in_channels,
                pretrained_path=raddino_pretrained_path
            )

    def forward(self, x):
        output = {}
        if self.use_local_stream:
            output['local_tokens'] = self.local_stream(x)
        else:
            output['local_tokens'] = None

        if self.use_global_stream:
            cls_token, patch_tokens = self.global_stream(x)
            output['global_cls'] = cls_token
            output['global_tokens'] = patch_tokens
        else:
            output['global_cls'] = None
            output['global_tokens'] = None

        return output