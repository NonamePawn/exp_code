import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from typing import List, Dict, Tuple


class ForensicsDataset(Dataset):
    """
    医学图像溯源数据集。
    特点：
    1. Lazy Loading: 读取单通道 .npy
    2. On-the-fly Feature Extraction: 实时生成 3通道特征 (自适应窗, 噪声残差, 软组织窗)
    3. Dynamic Crop: 训练时随机裁剪，测试时中心裁剪
    """

    def __init__(self, data_list: List[Dict], input_size: int = 256, is_train: bool = True):
        self.data_list = data_list
        self.input_size = input_size
        self.is_train = is_train

    def _normalize(self, img: np.ndarray, low: float, high: float) -> np.ndarray:
        """归一化到 [0, 1]"""
        img = np.clip(img, low, high)
        denom = high - low
        if denom == 0: denom = 1
        return (img - low) / denom

    def _make_3_channels(self, raw_img: np.ndarray) -> np.ndarray:
        """
        [关键步骤]
        从单通道 Raw Data 实时生成 3 通道特征张量。
        """
        # --- Ch0: Adaptive Wide (自适应全范围) ---
        p1 = np.percentile(raw_img, 1)
        p99 = np.percentile(raw_img, 99)
        if p99 - p1 < 100: p1, p99 = -1024, 2000
        ch0 = self._normalize(raw_img, p1, p99)

        # --- Ch1: Noise Residual (高频噪声指纹) ---
        # 算法: Image - GaussianBlur(Image)
        # 这一步非常快，CPU 处理通常只需几毫秒
        blurred = cv2.GaussianBlur(ch0, (3, 3), 0)
        residual = ch0 - blurred
        # 归一化残差 (通常很小，放大并平移)
        ch1 = np.clip((residual + 0.1) / 0.2, 0, 1)

        # --- Ch2: Soft Tissue (解剖结构上下文) ---
        # W:400, L:40 => [-160, 240]
        ch2 = self._normalize(raw_img, -160, 240)

        # Stack: (H, W, 3) -> Transpose to (3, H, W)
        img_merged = np.stack([ch0, ch1, ch2], axis=-1)
        return img_merged.transpose(2, 0, 1).astype(np.float32)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data_list[idx]

        # 1. 加载单通道 Raw Data (硬盘读取)
        try:
            raw_img = np.load(item['path'])
        except Exception as e:
            # 容错处理：返回全黑图
            print(f"Error loading {item['path']}: {e}")
            raw_img = np.zeros((512, 512), dtype=np.float32)

        # 2. CPU生成特征 (计算)
        img_3c = self._make_3_channels(raw_img)
        img_tensor = torch.from_numpy(img_3c)

        # 3. 裁剪 (数据增强)
        _, h, w = img_tensor.shape
        target = self.input_size

        # Pad if smaller
        if h < target or w < target:
            pad_h, pad_w = max(0, target - h), max(0, target - w)
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h))
            _, h, w = img_tensor.shape

        # Crop
        if self.is_train:
            # Random Crop
            top = np.random.randint(0, h - target + 1)
            left = np.random.randint(0, w - target + 1)
        else:
            # Center Crop
            top = (h - target) // 2
            left = (w - target) // 2

        img_tensor = img_tensor[:, top:top + target, left:left + target]
        label_tensor = torch.tensor(item['label'], dtype=torch.long)

        return img_tensor, label_tensor


def get_dataloader(data_list, batch_size=32, is_train=True):
    """
    DataLoader 工厂函数。
    重要：num_workers 建议设为 4 或 8，以掩盖 CPU 计算 3通道特征 的时间。
    """
    dataset = ForensicsDataset(data_list, input_size=512, is_train=is_train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=16,  # 多进程加速预处理
        pin_memory=True,
        drop_last=is_train
    )


if __name__ == "__main__":
    import os
    # --- 单元测试 ---
    print("Dataset module check...")
    # 模拟数据
    mock_data = [{"path": "mock.npy", "label": 0}]
    # 创建一个假文件用于测试
    np.save("mock.npy", np.random.randn(512, 512).astype(np.float32))

    try:
        ds = ForensicsDataset(mock_data, is_train=True)
        img, lbl = ds[0]
        print(f"Output Shape: {img.shape}")  # 应为 (3, 256, 256)
        print("Dataset test passed.")
    except Exception as e:
        print(f"Dataset test failed: {e}")
    finally:
        if os.path.exists("mock.npy"): os.remove("mock.npy")