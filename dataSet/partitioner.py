import os
import random
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any


class FederatedPartitioner:
    """
    负责扫描预处理后的数据，并为联邦学习生成索引列表。
    支持 IID (随机均匀) 和 Non-IID (狄利克雷分布) 划分。
    """

    def __init__(self, data_root: str, num_clients: int = 4, seed: int = 42):
        self.data_root = Path(data_root)
        self.num_clients = num_clients
        self.seed = seed

        # 设定随机种子，保证实验可复现
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.class_map = {}
        self.data_indices = self._scan_data()

    def _scan_data(self) -> List[Dict[str, Any]]:
        """遍历目录建立索引: File Path -> Label"""
        if not self.data_root.exists():
            # 这里的路径检查对于 Docker 环境排错很重要
            raise FileNotFoundError(f"Data root {self.data_root} does not exist!")

        samples = []
        # 第一级子目录作为设备类别 (e.g., Canon, Siemens...)
        device_dirs = [d for d in self.data_root.iterdir() if d.is_dir()]
        device_dirs.sort()

        self.class_map = {d.name: i for i, d in enumerate(device_dirs)}

        for d_dir in device_dirs:
            label_id = self.class_map[d_dir.name]
            files = list(d_dir.rglob("*.npy"))
            for f in files:
                samples.append({
                    "path": str(f),
                    "label": label_id,
                    "class_name": d_dir.name
                })

        print(
            f"[Partitioner] Indexed {len(samples)} samples from {len(self.class_map)} classes: {self.class_map.keys()}")
        return samples

    def get_all_data(self) -> List[Dict]:
        """获取全量数据 (通常用于测试集或验证集)"""
        return self.data_indices

    def split_iid(self) -> Dict[int, List[Dict]]:
        """
        [IID 划分]
        完全随机打乱数据，然后平均分给每个客户端。
        每个客户端的数据分布大致与全局分布一致。
        """
        indices = self.data_indices.copy()
        random.shuffle(indices)

        client_data = defaultdict(list)
        chunk_size = len(indices) // self.num_clients

        for i in range(self.num_clients):
            start = i * chunk_size
            # 最后一个客户端拿走剩余所有数据 (处理除不尽的情况)
            end = (i + 1) * chunk_size if i != self.num_clients - 1 else len(indices)
            client_data[i] = indices[start:end]

        return client_data

    def split_non_iid_dirichlet(self, alpha: float = 0.5) -> Dict[int, List[Dict]]:
        """
        [Non-IID 划分]
        使用狄利克雷分布 (Dirichlet Distribution) 模拟数据异质性。
        alpha 越小，Non-IID 程度越严重 (每个客户端只拥有少数几类设备的数据)。
        """
        num_classes = len(self.class_map)
        client_data = defaultdict(list)

        # 按类别分桶
        class_buckets = [[] for _ in range(num_classes)]
        for item in self.data_indices:
            class_buckets[item['label']].append(item)

        # 生成分布矩阵 (num_classes, num_clients)
        dist_matrix = np.random.dirichlet([alpha] * self.num_clients, num_classes)

        # 分配
        for cls_idx, cls_samples in enumerate(class_buckets):
            random.shuffle(cls_samples)
            total_samples = len(cls_samples)

            proportions = dist_matrix[cls_idx]
            # 计算切分点
            split_points = (np.cumsum(proportions) * total_samples).astype(int)[:-1]
            splits = np.split(cls_samples, split_points)

            for client_id, batch in enumerate(splits):
                if len(batch) > 0:
                    client_data[client_id].extend(batch)

        return client_data


if __name__ == "__main__":
    # --- Docker 环境绝对路径配置 ---
    # 我们主要对 'train' 数据集进行切分
    PROCESSED_TRAIN_DIR = "/assets_processed/train"

    if os.path.exists(PROCESSED_TRAIN_DIR):
        print(f"Partitioning data from: {PROCESSED_TRAIN_DIR}")

        partitioner = FederatedPartitioner(PROCESSED_TRAIN_DIR, num_clients=3)

        # 1. 测试 IID
        print("\n--- Testing IID Split ---")
        iid_clients = partitioner.split_iid()
        for cid, data in iid_clients.items():
            print(f"Client {cid}: {len(data)} samples")

        # 2. 测试 Non-IID
        print("\n--- Testing Non-IID Split (Alpha=0.5) ---")
        non_iid_clients = partitioner.split_non_iid_dirichlet(alpha=0.5)
        for cid, data in non_iid_clients.items():
            labels = [x['label'] for x in data]
            # 打印前3个常见的类别，看看是否偏斜
            counts = Counter(labels).most_common(3)
            print(f"Client {cid}: {len(data)} samples. Top classes: {counts}")

    else:
        print(f"Error: Directory {PROCESSED_TRAIN_DIR} not found. Please run preprocessor.py first.")