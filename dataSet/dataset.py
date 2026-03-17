import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from typing import List, Dict, Tuple


class ForensicsDataset(Dataset):
    """
    医学图像溯源数据集 (单通道版)。
    -------------------------------------------
    修改说明：
    1. 改为单通道输出，使用自适应归一化处理。
    2. 增强了 Padding 逻辑，解决图片尺寸小于 target 时的报错问题。
    """

    def __init__(self, data_list: List[Dict], input_size: int = 512, is_train: bool = True):
        self.data_list = data_list
        self.input_size = input_size
        self.is_train = is_train

    def _normalize(self, img: np.ndarray, low: float, high: float) -> np.ndarray:
        """将像素值归一化到 [0, 1] 区间"""
        img = np.clip(img, low, high)
        denom = high - low
        if denom == 0: denom = 1
        return (img - low) / denom

    def _process_single_channel(self, raw_img: np.ndarray) -> np.ndarray:
        """
        [核心修改] 只生成单通道特征：自适应窗归一化。
        """
        # 计算 1% 和 99% 分位数以确定显示范围
        p1 = np.percentile(raw_img, 1)
        p99 = np.percentile(raw_img, 99)

        # 如果动态范围太窄，使用标准 CT 肺窗/骨窗范围作为保底
        if p99 - p1 < 100:
            p1, p99 = -1024, 2000

        ch0 = self._normalize(raw_img, p1, p99)

        # 增加通道维度: (H, W) -> (1, H, W)
        return ch0[np.newaxis, :, :].astype(np.float32)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data_list[idx]

        # 1. 读取原始单通道 .npy 数据
        try:
            raw_img = np.load(item['path'])
        except Exception as e:
            raw_img = np.zeros((self.input_size, self.input_size), dtype=np.float32)

        # 2. 转换为单通道张量
        img_1c = self._process_single_channel(raw_img)
        img_tensor = torch.from_numpy(img_1c)

        # 3. 尺寸对齐与裁剪
        c, h, w = img_tensor.shape
        target = self.input_size

        # 【安全补齐】如果原图小于目标尺寸，补齐到刚好等于 target
        if h < target or w < target:
            pad_h = max(0, target - h)
            pad_w = max(0, target - w)
            # 【修复点】torchvision 的 F.pad 顺序必须是 (左, 上, 右, 下)
            img_tensor = F.pad(img_tensor, (0, 0, pad_w, pad_h))
            _, h, w = img_tensor.shape

        # 【安全裁剪】这里加上的 + 1 是防报错的灵魂！
        if self.is_train:
            top = np.random.randint(0, h - target + 1)
            left = np.random.randint(0, w - target + 1)
        else:
            top = (h - target) // 2
            left = (w - target) // 2

        img_tensor = img_tensor[:, top:top + target, left:left + target]
        label_tensor = torch.tensor(item['label'], dtype=torch.long)

        return img_tensor, label_tensor


def get_dataloader(data_list, batch_size=32, is_train=True):
    """
    DataLoader 工厂函数
    """
    dataset = ForensicsDataset(data_list, input_size=512, is_train=is_train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=16,  # 多线程加速
        pin_memory=True,
        drop_last=is_train
    )