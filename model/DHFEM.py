import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModel

class RadImageNetExtractor(nn.Module):
    """
    独立类 1: 负责提取局部空间特征的 CNN 网络
    采用输入适配策略，保持原网络 3 通道结构零修改
    """

    def __init__(self, in_channels=1, pretrained_path=None):
        super().__init__()

        # 1. 通道适配器 (Channel Adapter)
        self.adapter = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0)

        # 2. 原封不动的标准 ResNet50 (3 通道)
        self.cnn = models.resnet50(weights=None)

        # 尝试加载 RadImageNet 预训练权重
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.cnn.load_state_dict(new_state_dict, strict=False)
            print(f"✅ RadImageNet 预训练权重加载成功！(路径: {pretrained_path})")
        except FileNotFoundError:
            print(f"⚠️ 警告: 未找到 {pretrained_path}，使用随机初始化权重。")

        # 剥离最后两层，保留空间特征图
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])

    def forward(self, x_multi):
        # x_multi: (B, 5, 512, 512) -> adapter -> (B, 3, 512, 512) -> cnn -> (B, 2048, 16, 16)
        x_3c = self.adapter(x_multi)
        return self.cnn(x_3c)


class RadDINOExtractor(nn.Module):
    """
    独立类 2: 负责提取全局序列特征的 ViT 网络
    纯本地加载模式，彻底告别网络请求
    """

    def __init__(self, in_channels=1, pretrained_path=None):
        super().__init__()

        # 1. 通道适配器
        self.adapter = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0)

        # 2. 纯本地加载核心逻辑
        print(f"⏳ 正在从本地路径 [{pretrained_path}] 加载 RadDINO 预训练权重...")

        try:
            # 【核心修改】把 ViTModel 换成 AutoModel
            self.vit = AutoModel.from_pretrained(
                pretrained_path,
                local_files_only=True
            )
            print("✅ RadDINO 本地预训练权重加载成功！(原生态 3 通道 DINOv2 架构)")
        except Exception as e:
            print(f"\n❌ RadDINO 加载失败！")
            print(f"请检查本地文件夹 '{pretrained_path}' 中是否完整包含必须的模型文件。")
            print(f"通常需要: config.json 和 pytorch_model.bin (或 model.safetensors)")
            raise e

    def forward(self, x_multi):
        # x_multi: (B, 5, 512, 512) -> 插值 -> (B, 5, 518, 518) -> adapter -> (B, 3, 518, 518)
        x_resized = F.interpolate(x_multi, size=(518, 518), mode='bilinear', align_corners=False)
        x_3c = self.adapter(x_resized)

        # 送入原生态 ViT
        vit_outputs = self.vit(pixel_values=x_3c)

        # 剥离 cls_token，保留 1369 个空间 patch tokens
        # 输出: (B, 1369, 768)
        global_tokens = vit_outputs.last_hidden_state[:, 1:, :]
        return global_tokens


class DHFEM(nn.Module):
    """
    特征提取总成模块
    """

    def __init__(self, use_radimagenet=True, use_raddino=True,
                 radimagenet_path='./pretrained/radImageNet_resNet50.pt',
                 raddino_path='./pretrained/rad_dino_weights'):
        """
        参数说明:
        radimagenet_path: RadImageNet 的 .pth 权重文件路径
        raddino_path: RadDINO 的本地文件夹路径
        """
        super().__init__()
        self.use_radimagenet = use_radimagenet
        self.use_raddino = use_raddino

        if self.use_radimagenet:
            self.cnn_extractor = RadImageNetExtractor(in_channels=1, pretrained_path=radimagenet_path)

        if self.use_raddino:
            self.vit_extractor = RadDINOExtractor(in_channels=1, pretrained_path=raddino_path)

    def forward(self, x_multi):
        local_feat = None
        global_feat = None

        if self.use_radimagenet:
            local_feat = self.cnn_extractor(x_multi)

        if self.use_raddino:
            global_feat = self.vit_extractor(x_multi)

        return local_feat, global_feat