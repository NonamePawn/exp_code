import torch
import torch.nn as nn
import torchvision.models as models
import timm  # 用于加载 Transformer (ViT) 模型
import os
import torch.nn.functional as F

# ==========================================
# 1. 局部纹理流: MedicalNet 2D 重构版本 (内嵌 3D->2D 权重转换)
# ==========================================

class MedicalNet2D_Wrapper(nn.Module):
    """
    MedicalNet 2D 适配器
    大论文要求：将3D ResNet扁平化为2D，并修改第一层卷积接收 5 通道输入，最后截断全局分类头。
    增加了内部自动加载和转换 3D 预训练权重的方法。
    """

    def __init__(self, in_channels=5, base_model_name='resnet50', pretrained_path=None):
        super().__init__()
        self.in_channels = in_channels

        # 1. 实例化一个标准的 2D ResNet
        if base_model_name == 'resnet50':
            self.backbone = models.resnet50(weights=None)
            feature_dim = 2048
        elif base_model_name == 'resnet18':
            self.backbone = models.resnet18(weights=None)
            feature_dim = 512
        else:
            raise ValueError("不支持的 ResNet 版本，请选择 resnet18 或 resnet50")

        # 2. 修改第一层卷积以适配 GMPM 的多通道输出 (例如 5 通道)
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )

        # 3. 截断全局语义分类头 (移除 fc 层)
        # 获取除了最后的池化层和全连接层之外的所有层，形成特征提取器
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-2])

        # 4. 添加一个1x1卷积进行降维或通道对齐 (为了后续和 DINO 特征融合)
        self.proj = nn.Conv2d(feature_dim, 768, kernel_size=1)

        # =======================================
        # 5. 如果传入了预训练路径，立即执行“权重手术”
        # =======================================
        if pretrained_path is not None and os.path.exists(pretrained_path):
            self._load_3d_pretrained_weights(pretrained_path)
        elif pretrained_path is not None:
            print(f"[警告] 找不到预训练文件: {pretrained_path}，将使用随机初始化。")

    def _load_3d_pretrained_weights(self, pth_path):
        """
        内部私有方法：读取 3D MedicalNet 权重，压缩为 2D 并解决通道冲突。
        """
        print(f"====== 开始为 MedicalNet2D 加载并转换 3D 预训练权重 ======")
        checkpoint_3d = torch.load(pth_path, map_location='cpu')

        # 提取真正的字典内容 (处理 DataParallel 产生的嵌套)
        state_dict_3d = checkpoint_3d.get('state_dict', checkpoint_3d)
        state_dict_2d = self.state_dict()
        new_state_dict = {}
        loaded_layers = 0

        for key_2d in state_dict_2d.keys():
            # 解决名称前缀映射 (self.backbone -> module)
            key_3d = key_2d.replace('backbone.', 'module.')
            if key_3d not in state_dict_3d:
                key_3d = key_2d.replace('backbone.', '')

            if key_3d in state_dict_3d:
                weight_3d = state_dict_3d[key_3d]

                # --- 步骤 A：3D 到 2D 的通道均值投影 ---
                # [out_c, in_c, Depth, Height, Width] -> [out_c, in_c, Height, Width]
                if len(weight_3d.shape) == 5 and len(state_dict_2d[key_2d].shape) == 4:
                    weight_2d = weight_3d.mean(dim=2)
                else:
                    weight_2d = weight_3d.clone()

                # --- 步骤 B：处理 conv1 的输入通道扩充 (3 -> 5) ---
                if 'conv1.weight' in key_2d and weight_2d.shape[1] != self.in_channels:
                    print(f"    -> 正在对齐 {key_2d} 的输入通道: {weight_2d.shape[1]} -> {self.in_channels}")
                    # 取 3 个通道的均值，复制成 in_channels (5) 份
                    mean_weight = weight_2d.mean(dim=1, keepdim=True)
                    weight_2d = mean_weight.repeat(1, self.in_channels, 1, 1)

                # 安全检查与装载
                if weight_2d.shape == state_dict_2d[key_2d].shape:
                    new_state_dict[key_2d] = weight_2d
                    loaded_layers += 1
                else:
                    new_state_dict[key_2d] = state_dict_2d[key_2d]
            else:
                # 官方权重里没有的层 (比如自建的 proj 层) 保持随机初始化
                new_state_dict[key_2d] = state_dict_2d[key_2d]

        self.load_state_dict(new_state_dict, strict=False)
        print(f"====== MedicalNet 权重装载完成！共转换 {loaded_layers} 个张量 ======")

    def forward(self, x):
        features = self.feature_extractor(x)  # (B, 2048, H/32, W/32)
        features = self.proj(features)  # (B, 768, H/32, W/32)

        # 展平以便于后续的注意力机制交互
        B, C, H, W = features.shape
        local_tokens = features.view(B, C, -1).permute(0, 2, 1)  # (B, N, 768)
        return local_tokens


# ==========================================
# 2. 全局关联流: RAD-DINO (ViT) 适配器
# ==========================================

class RADDINO_Wrapper(nn.Module):
    """
    RAD-DINO 适配器 (内嵌权重加载与通道手术功能)
    """

    def __init__(self, in_channels=5, pretrained_path=None):
        super().__init__()
        self.in_channels = in_channels

        # 1. 实例化基础空壳 ViT
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)

        # 2. 修改 Patch Embedding 层的输入通道数
        original_proj = self.vit.patch_embed.proj
        self.vit.patch_embed.proj = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=original_proj.out_channels,
            kernel_size=original_proj.kernel_size,
            stride=original_proj.stride,
            padding=original_proj.padding
        )

        # 3. 自动加载预训练权重并做“通道手术”
        if pretrained_path is not None and os.path.exists(pretrained_path):
            self._load_pretrained_weights(pretrained_path)
        elif pretrained_path is not None:
            print(f"[警告] 找不到 RAD-DINO 预训练文件: {pretrained_path}，使用随机初始化。")

    def _load_pretrained_weights(self, pth_path):
        print(f"====== 开始为 RAD-DINO 加载并转换预训练权重 ======")

        # 【新增逻辑】：判断并读取 safetensors 格式
        if pth_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            checkpoint = load_file(pth_path)
            state_dict_pretrained = checkpoint
        else:
            checkpoint = torch.load(pth_path, map_location='cpu')
            state_dict_pretrained = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))

        model_state_dict = self.vit.state_dict()
        new_state_dict = {}
        loaded_layers = 0

        for key in model_state_dict.keys():
            # HuggingFace 的权重可能带有不同前缀，RAD-DINO 通常匹配 timm 的 keys
            if key in state_dict_pretrained:
                weight = state_dict_pretrained[key]

                # --- 核心手术：处理 patch_embed 的通道冲突 ---
                if 'patch_embed.proj.weight' in key and weight.shape[1] != self.in_channels:
                    print(f"    -> 正在对齐 {key} 的输入通道: {weight.shape[1]} -> {self.in_channels}")
                    # 取预训练通道的均值，然后复制成 in_channels (5) 份
                    mean_weight = weight.mean(dim=1, keepdim=True)
                    weight = mean_weight.repeat(1, self.in_channels, 1, 1)

                if weight.shape == model_state_dict[key].shape:
                    new_state_dict[key] = weight
                    loaded_layers += 1
            else:
                new_state_dict[key] = model_state_dict[key]

        self.vit.load_state_dict(new_state_dict, strict=False)
        print(f"====== RAD-DINO 权重装载完成！共转换 {loaded_layers} 个张量 ======")
        
    def forward(self, x):
        # 专门为 ViT 缩放尺寸到 224x224
        x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        x_tokens = self.vit.patch_embed(x_resized)
        x_tokens = self.vit._pos_embed(x_tokens)
        x_tokens = self.vit.norm_pre(x_tokens)
        tokens = self.vit.blocks(x_tokens)
        tokens = self.vit.norm(tokens)

        cls_token = tokens[:, 0, :]  # (B, 768)
        patch_tokens = tokens[:, 1:, :]  # (B, N, 768)
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