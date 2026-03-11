import matplotlib

# 【关键】必须在 import pyplot 之前设置
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import os

# ================= 配置区域 =================
RESULT_DIR = '../result/central_pre__resnet50'  # 数据所在的文件夹 (父目录)
PLOT_SUBDIR = 'plot'  # 图片保存的子文件夹名
CSV_FILE = 'training_log.csv'
NPZ_FILE = 'test_analysis_data.npz'


# ===========================================

def plot_training_curves(log_path, save_dir):
    """1. 读取CSV并保存训练曲线图"""
    print(f"1. [处理中] 训练曲线...")

    if not os.path.exists(log_path):
        print(f"   [跳过] 找不到文件 {log_path}")
        return

    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        print(f"   [错误] 读取 CSV 失败: {e}")
        return

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    if 'train_loss' in df.columns:
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='tab:blue')
    elif 'loss' in df.columns:
        plt.plot(df['epoch'], df['loss'], label='Train Loss', color='tab:blue')
    if 'val_loss' in df.columns:
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss', color='tab:red', linestyle='--')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy
    plt.subplot(1, 2, 2)
    if 'train_acc' in df.columns:
        plt.plot(df['epoch'], df['train_acc'], label='Train Acc', color='tab:green')
    elif 'acc' in df.columns:
        plt.plot(df['epoch'], df['acc'], label='Train Acc', color='tab:green')
    if 'val_acc' in df.columns:
        plt.plot(df['epoch'], df['val_acc'], label='Val Acc', color='tab:orange', linestyle='--')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   --> 已保存: {save_path}")


def plot_confusion_matrix(y_true, y_pred_idx, save_dir):
    """2. 保存混淆矩阵 (数值 x100, 1位小数)"""
    print(f"2. [处理中] 混淆矩阵...")

    # 归一化并 x100
    cm = confusion_matrix(y_true, y_pred_idx, normalize='true')
    cm = cm * 100

    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues', square=True, vmin=0, vmax=100)

    plt.title('Confusion Matrix (%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   --> 已保存: {save_path}")

    # 保存详细报告 (txt也放在plot文件夹里方便查看)
    report_path = os.path.join(save_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(classification_report(y_true, y_pred_idx))
    print(f"   --> 详细报告已保存: {report_path}")


def plot_roc_curve(y_true, y_probs, save_dir):
    """3. 保存 ROC 曲线"""
    print(f"3. [处理中] ROC 曲线...")

    if y_probs.ndim == 1:
        print("   [跳过] 预测数据是一维的，无法绘制ROC。")
        return

    n_classes = y_probs.shape[1]
    classes = np.unique(y_true)
    # 处理类别不全的情况
    if len(classes) != n_classes:
        classes = range(n_classes)

    y_true_bin = label_binarize(y_true, classes=classes)

    plt.figure(figsize=(10, 8))

    # 循环画每一类的曲线
    for i in range(n_classes):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
        except Exception:
            pass  # 某类不存在时忽略

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, 'roc_curve.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   --> 已保存: {save_path}")


def plot_tsne_distribution(y_true, features, save_dir, max_samples=2000):
    """4. 保存 t-SNE 特征分布图"""
    print(f"4. [处理中] t-SNE 特征分布图...")

    if features is None:
        print("   [跳过] 未找到特征数据 (features)。")
        return

    n_samples = features.shape[0]
    if n_samples > max_samples:
        print(f"   采样 {max_samples} / {n_samples} 个点进行可视化...")
        indices = np.random.choice(n_samples, max_samples, replace=False)
        features = features[indices]
        y_true = y_true[indices]

    print("   正在计算 t-SNE...")
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
    X_embedded = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("tab10", len(np.unique(y_true)))
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y_true, palette=palette, legend='full', s=60, alpha=0.8)

    plt.title('t-SNE Feature Visualization')

    save_path = os.path.join(save_dir, 'tsne_features.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   --> 已保存: {save_path}")


def main():
    if not os.path.exists(RESULT_DIR):
        print(f"警告: 找不到目录 {RESULT_DIR}")
        return

    # 【核心修改】创建 plot 子文件夹
    plot_save_dir = os.path.join(RESULT_DIR, PLOT_SUBDIR)
    # exist_ok=True 表示如果文件夹已存在也不会报错
    os.makedirs(plot_save_dir, exist_ok=True)
    print(f"图片保存目录: {plot_save_dir}\n")

    # 1. 训练曲线 (读取数据仍在外面，保存路径传入子文件夹)
    csv_full_path = os.path.join(RESULT_DIR, CSV_FILE)
    plot_training_curves(csv_full_path, plot_save_dir)

    # 2. 读取 NPZ 数据
    npz_full_path = os.path.join(RESULT_DIR, NPZ_FILE)
    if not os.path.exists(npz_full_path):
        print(f"找不到 NPZ 文件: {npz_full_path}")
        return

    try:
        data = np.load(npz_full_path)
    except Exception as e:
        print(f"读取 NPZ 失败: {e}")
        return

    # 提取数据
    if 'labels' in data:
        y_true = data['labels']
    elif 'y_true' in data:
        y_true = data['y_true']
    else:
        print("Error: 找不到真实标签"); return

    if 'preds' in data:
        y_pred_raw = data['preds']
    elif 'y_pred' in data:
        y_pred_raw = data['y_pred']
    elif 'probs' in data:
        y_pred_raw = data['probs']
    else:
        print("Error: 找不到预测数据"); return

    if y_pred_raw.ndim > 1:
        y_probs = y_pred_raw
        y_pred_idx = np.argmax(y_pred_raw, axis=1)
    else:
        y_probs = y_pred_raw
        y_pred_idx = y_pred_raw.astype(int)

    features = None
    if 'features' in data:
        features = data['features']
    elif 'feats' in data:
        features = data['feats']
    elif 'embeddings' in data:
        features = data['embeddings']

    # 3. 执行绘图 (传入 plot_save_dir)
    plot_confusion_matrix(y_true, y_pred_idx, plot_save_dir)
    plot_roc_curve(y_true, y_probs, plot_save_dir)
    plot_tsne_distribution(y_true, features, plot_save_dir)

    print("\n========== 完成 ==========")
    print(f"所有图片已保存在: {plot_save_dir}")


if __name__ == '__main__':
    main()