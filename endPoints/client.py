import torch
import torch.nn as nn
import copy
from tqdm import tqdm
from dataSet.dataset import get_dataloader


class Client:
    def __init__(self, client_id, data_list, device, batch_size=32):
        """
        :param client_id: 客户端ID
        :param data_list: 分配给该客户端的文件路径列表 (Subset)
        :param device: 计算设备
        """
        self.client_id = client_id
        self.device = device
        self.data_len = len(data_list)

        # 复用 dataset.py 的工厂函数，只加载属于自己的数据
        self.train_loader = get_dataloader(
            data_list=data_list,
            batch_size=batch_size,
            is_train=True
        )
        self.criterion = nn.CrossEntropyLoss()

    def train(self, global_model, local_epochs, lr, verbose=False):
        """
        本地训练主逻辑
        :param verbose: 是否显示详细的 Epoch 进度条（建议设为 False 以免控制台刷屏）
        :return: (更新后的参数字典, 平均Loss)
        """
        # 1. 模型同步 (Download): 必须深拷贝，否则会修改服务器的原版模型
        local_model = copy.deepcopy(global_model)
        local_model.to(self.device)
        local_model.train()

        # 2. 优化器: FedAvg 通常每轮重新初始化 SGD
        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

        epoch_losses = []

        # 3. 训练循环
        # 如果 verbose=True，显示 Epoch 进度条；否则使用静默 range
        iterator = range(local_epochs)
        if verbose:
            iterator = tqdm(iterator, desc=f"Client {self.client_id}", leave=False)

        for epoch in iterator:
            batch_losses = []
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = local_model(images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())

            # 记录该 Epoch 平均 Loss
            ep_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
            epoch_losses.append(ep_loss)

            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix(loss=f"{ep_loss:.4f}")

        # 计算所有 Local Epochs 的总平均 Loss
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0

        # 4. 上传 (Upload): 只返回 state_dict 以节省显存
        return local_model.state_dict(), avg_loss