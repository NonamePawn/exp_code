import torch
import copy
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class Server:
    def __init__(self, global_model, val_loader, device):
        self.global_model = global_model
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def aggregate(self, client_weights_list):
        """FedAvg: 联邦平均算法"""
        w_avg = copy.deepcopy(client_weights_list[0])

        for key in w_avg.keys():
            for i in range(1, len(client_weights_list)):
                w_avg[key] += client_weights_list[i][key]
            w_avg[key] = torch.div(w_avg[key], len(client_weights_list))

        self.global_model.load_state_dict(w_avg)

    def evaluate(self):
        """
        [普通验证] 只计算 Acc 和 Loss，不收集数据
        """
        self.global_model.eval()
        self.global_model.to(self.device)

        loss_sum = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            # 🌟 优化1：加上蓝色的测试专属进度条
            val_pbar = tqdm(self.val_loader, desc="   📊 Server Evaluating", leave=False, ncols=120, colour='blue')

            for images, labels in val_pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                # 🌟 优化2：开启混合精度极速推理
                with torch.cuda.amp.autocast():
                    output = self.global_model(images)

                    # 兼容性处理：如果模型返回 (logits, features)，只取 logits
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output

                    loss = self.criterion(logits, labels)

                loss_sum += loss.item()

                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 🌟 优化3：进度条实时反馈
                current_acc = 100 * correct / total if total > 0 else 0
                val_pbar.set_postfix({'Acc': f"{current_acc:.2f}%", 'Loss': f"{loss.item():.3f}"})

        # 防止除零
        acc = 100 * correct / total if total > 0 else 0
        avg_loss = loss_sum / len(self.val_loader) if len(self.val_loader) > 0 else 0

        # 🌟 优化4：测试完毕释放显存
        self.global_model.cpu()
        # torch.cuda.empty_cache()

        return acc, avg_loss

    def compute_exclusive_class_logits(self, all_client_class_logits, all_client_class_counts):
        """
        [大论文核心] 计算基于类别的排他性聚合知识 (Class-wise Virtual Teacher)
        输入:
            all_client_class_logits: {client_id: {class_id: mean_logit_tensor}}
            all_client_class_counts: {client_id: {class_id: sample_count}}
        """
        global_class_sum = {}
        global_class_count = {}

        # 1. 计算全局所有类别的 Logits 累加和及总样本数
        for cid, class_logits in all_client_class_logits.items():
            for lbl, logits in class_logits.items():
                count = all_client_class_counts[cid][lbl]
                if lbl not in global_class_sum:
                    global_class_sum[lbl] = logits * count
                    global_class_count[lbl] = count
                else:
                    global_class_sum[lbl] += logits * count
                    global_class_count[lbl] += count

        exclusive_teacher_logits = {}

        # 2. 为每个客户端剔除其自身后，计算专属的类原型老师
        for cid in all_client_class_logits.keys():
            exclusive_teacher_logits[cid] = {}
            for lbl in all_client_class_logits[cid].keys():
                count = all_client_class_counts[cid][lbl]
                # 排他性：减去自己的部分
                exc_sum = global_class_sum[lbl] - all_client_class_logits[cid][lbl] * count
                exc_count = global_class_count[lbl] - count

                if exc_count > 0:
                    exclusive_teacher_logits[cid][lbl] = exc_sum / exc_count
                else:
                    # 如果只有当前客户端有这个类别，就沿用自己上一轮的知识作为平滑
                    exclusive_teacher_logits[cid][lbl] = all_client_class_logits[cid][lbl]

        return exclusive_teacher_logits

    def run_final_test(self, model_path, test_loader=None):
        """
        [最终测试] 完全复刻 main_central.py 的逻辑
        :param test_loader: 传入专门的测试集加载器，如果不传则默认用 val_loader
        """
        # 1. 加载最佳权重
        print(f"📥 Server loading best model from: {model_path}")
        state_dict = torch.load(model_path, map_location=self.device)
        self.global_model.load_state_dict(state_dict)

        self.global_model.eval()
        self.global_model.to(self.device)

        # 2. 决定使用哪个数据集 (Test 优先)
        target_loader = test_loader if test_loader is not None else self.val_loader
        print(f"🧪 Running Inference on {len(target_loader.dataset)} samples...")

        # 3. 容器
        all_logits = []
        all_preds = []
        all_labels = []
        all_features = []

        with torch.no_grad():
            for images, labels in tqdm(target_loader, desc="   📊 Extracting", ncols=100, colour='magenta'):
                images = images.to(self.device)

                # 🌟 优化：开启混合精度加速特征提取
                with torch.cuda.amp.autocast():
                    # ======== 仅修改这一行：增加 return_features=True ========
                    output = self.global_model(images, return_features=True)

                # 判断模型是否返回了特征
                if isinstance(output, tuple):
                    logits = output[0]
                    features = output[1]
                else:
                    logits = output
                    # 如果模型没返回特征，用 logits 充当特征
                    features = logits

                _, preds = torch.max(logits, 1)

                # 收集基础数据
                all_logits.append(logits.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.numpy())

                # 处理特征维度 (参考 main_central.py: if len > 2 then mean)
                if features is not None:
                    if len(features.shape) > 2:
                        features = features.mean(dim=[2, 3])
                    all_features.append(features.cpu().numpy())

        # 拼接数据
        logits = np.concatenate(all_logits, axis=0)
        preds = np.concatenate(all_preds, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        features = np.concatenate(all_features, axis=0) if all_features else np.array([])

        self.global_model.cpu()
        # torch.cuda.empty_cache()

        return features, labels, preds, logits

    def predict_openset(self, logits, threshold_delta=0.8):
        """
        4.2.3 动态拒识判定准则 (对应公式 4-9)
        :param logits: 模型输出 [Batch, K+1]
        :param threshold_delta: 拒识博弈阈值 δ (据你论文图4-7，0.8为最优平衡点)
        :return: 最终预测类别张量 (-1 代表未知类)
        """
        # 转化为概率分布
        probs = F.softmax(logits, dim=1)

        # 提取已知类 (前 K 维) 的最大概率
        known_probs = probs[:, :-1]
        max_known_probs, _ = torch.max(known_probs, dim=1)

        # 提取第 K+1 维 (未知类) 的概率
        unknown_probs = probs[:, -1]

        # 核心判定：如果未知类概率 >= δ * 已知类最大概率，则触发拒识拦截
        is_unknown = unknown_probs >= (threshold_delta * max_known_probs)

        # 获取基础的已知类预测
        final_preds = torch.argmax(known_probs, dim=1)

        # 将被拦截的样本类别标记为 -1 (代表 Open-set Unknown)
        final_preds[is_unknown] = -1

        return final_preds