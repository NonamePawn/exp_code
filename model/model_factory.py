import torch
import timm

# 1. 导入你的全新 FDMFF 模型
from model.FDMFF import FDMFF

# 2. 在这里注册你的模型
CUSTOM_MODELS = {
    # 键名就是你在命令行里传入的 --model 的名字
    'fdmff': FDMFF,
}

def get_model(model_name, num_classes):
    print(f"[*] Model Factory: Loading {model_name}...")

    if model_name in CUSTOM_MODELS:
        # 如果是你自定义的模型，实例化它
        # 如果需要传入预训练路径，可以在这里加参数，例如 medicalnet_pretrained_path='xxx.pth'
        model = CUSTOM_MODELS[model_name](num_classes=num_classes,medicalnet_pretrained_path='./pretrained/',raddino_pretrained_path='./pretrained/rad_dino_base.safetensors')
    else:
        # 找不到就去 timm 库里找通用基线模型 (并强制单通道输入)
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes, in_chans=1)

    return model