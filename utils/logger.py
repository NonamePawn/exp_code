import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from thop import profile  # 用于计算模型运算量 (FLOPs)
import time


class ExperimentLogger:
    """
    [工具 1] 实验日志记录器
    -------------------------------------------
    作用：就像飞机的黑匣子。
    它负责在训练过程中，把每一个 Epoch 的 Loss、准确率(Acc)、F1分数
    实时写入到一个 CSV 表格里。

    好处：即使训练中途断电或报错，之前的训练记录也都保存在 CSV 里了，不会白跑。
    """

    def __init__(self, save_dir):
        # Path 是 pathlib 库的对象，它能自动处理 Windows(\) 和 Linux(/) 的路径分隔符问题
        self.save_dir = Path(save_dir)

        # mkdir: 创建文件夹
        # parents=True: 如果父目录不存在（比如 central_logs/），自动创建
        # exist_ok=True: 如果文件夹已经存在，不要报错，直接用
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 定义日志文件的完整路径，例如: ./central_logs/exp1/training_log.csv
        self.log_path = self.save_dir / "training_log.csv"

        # 用于在内存中暂时存储所有记录的列表
        self.history = []

    def log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, val_f1, time_elapsed):
        """
        在每个 Epoch 结束时调用此函数。
        """
        # 1. 把所有指标打包成一个字典
        record = {
            "epoch": epoch + 1,  # 当前轮数 (习惯上从 1 开始计数)
            "train_loss": round(train_loss, 4),  # 保留4位小数，看起来整洁
            "train_acc": round(train_acc, 2),  # 百分比保留2位小数
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "val_f1": round(val_f1, 2),  # F1 分数
            "time_sec": round(time_elapsed, 1)  # 这一轮跑了多少秒
        }

        # 2. 加入内存列表
        self.history.append(record)

        # 3. [关键步骤] 立即保存到硬盘
        # 我们每次都重新把整个 history 列表写入 CSV。
        # 这样做虽然看起来有点重复，但最安全！
        # 如果训练在第 50 轮崩溃，你依然能在 CSV 里看到前 49 轮的数据。
        df = pd.DataFrame(self.history)
        df.to_csv(self.log_path, index=False)

        return record

def save_analysis_data(save_dir, features, labels, preds, logits, phase='test'):
    """
    [工具 2] 保存可视化分析数据 (.npz)
    -------------------------------------------
    作用：训练完了，我们通常需要画图写论文。
    这个函数把画图所需的所有原始数据打包保存成一个压缩文件 (.npz)。

    以后你可以随时读取这个文件来画：
    1. 混淆矩阵 (Confusion Matrix) -> 需要 labels 和 preds
    2. t-SNE 特征分布图 -> 需要 features 和 labels
    3. ROC 曲线 -> 需要 logits (概率分布)
    """
    # 拼接保存路径，例如: ./central_logs/exp1/test_analysis_data.npz
    save_path = Path(save_dir) / f"{phase}_analysis_data.npz"

    # numpy.savez 是 numpy 专用的压缩存储格式
    np.savez(save_path,
             features=features,  # 模型的中间层特征向量 (用于看类间距离)
             labels=labels,  # 真实标签
             preds=preds,  # 预测标签
             logits=logits)  # 模型输出的原始分数

    print(f"[*] Analysis data saved to: {save_path}")


def get_f1_score(y_true, y_pred):
    """
    [工具 4] 计算 F1 分数
    -------------------------------------------
    作用：在医学图像中，类别往往不平衡（比如患病的人少，健康的人多）。
    光看准确率 (Accuracy) 可能会骗人（全猜健康也有 90% 准确率）。
    F1 Score 综合了 Precision 和 Recall，是更公正的指标。

    average='macro': 对每个类别单独算 F1 然后取平均，给小类别同样的权重。
    """
    # y_true: 真实标签列表
    # y_pred: 预测标签列表
    return f1_score(y_true, y_pred, average='macro')

# --- 辅助打印工具 ---
def print_box(title, content_lines):
    width = 76
    print(f"\n╔{'═' * width}╗")
    print(f"║ {title.center(width - 2)} ║")
    print(f"╠{'═' * width}╣")
    for line in content_lines:
        print(f"║ {line.ljust(width - 2)} ║")
    print(f"╚{'═' * width}╝")