import torch
import timm

# 1. 【关键】这里导入你真正写的模型类
from model.model import DualStreamVAE

# 2. 【关键】在这里注册你的模型
CUSTOM_MODELS = {
    # 给你的模型起个名字，比如 'my_design'
    # 后面填你的类名
    'my_model': DualStreamVAE,
}

def get_model(model_name, num_classes):
    print(f"[*] Model Factory: Loading {model_name}...")

    if model_name in CUSTOM_MODELS:
        # 这里会实例化你的 MySuperNet
        model = CUSTOM_MODELS[model_name](num_classes=num_classes)
    else:
        # 找不到就去 timm 库里找
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    return model