import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def split_ct_dataset(src_root, dst_root, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    src_root: D:/exp_data
    dst_root: 划分后的保存路径
    """
    # 1. 收集数据结构: {类别: [病人文件夹路径1, 病人文件夹路径2, ...]}
    class_to_patients = {}
    classes = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]

    for cls in classes:
        cls_path = os.path.join(src_root, cls)
        patients = [os.path.join(cls, p) for p in os.listdir(cls_path) if os.path.isdir(os.path.join(cls_path, p))]
        class_to_patients[cls] = patients

    # 准备存储路径
    sets = ['train', 'val', 'test']
    for s in sets:
        for cls in classes:
            os.makedirs(os.path.join(dst_root, s, cls), exist_ok=True)

    # 2. 按类别进行划分
    for cls, patients in class_to_patients.items():
        print(f"Processing class: {cls}, Total patients: {len(patients)}")

        # 首先分出 60% 的训练集
        train_pts, temp_pts = train_test_split(
            patients,
            test_size=(1 - train_ratio),
            random_state=42,
            shuffle=True
        )

        # 将剩余的 40% 平分为验证集和测试集 (各占总体的 10%)
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)
        val_pts, test_pts = train_test_split(
            temp_pts,
            test_size=(1 - relative_val_ratio),
            random_state=42,
            shuffle=True
        )

        # 3. 物理拷贝文件
        split_map = {'train': train_pts, 'val': val_pts, 'test': test_pts}

        for set_name, pts_list in split_map.items():
            for pt_rel_path in tqdm(pts_list, desc=f"  Copying {set_name}"):
                # pt_rel_path 是 "设备名/病人ID"
                src_patient_path = os.path.join(src_root, pt_rel_path)
                dst_patient_path = os.path.join(dst_root, set_name, pt_rel_path)

                # 如果目标文件夹已存在则先删除，确保干净
                if os.path.exists(dst_patient_path):
                    shutil.rmtree(dst_patient_path)

                # 直接拷贝整个病人文件夹
                shutil.copytree(src_patient_path, dst_patient_path)


if __name__ == "__main__":
    # 配置路径
    source_dir = r'D:\exp_data'
    target_dir = r'D:\exp_data_split'  # 建议保存在新文件夹，防止弄乱原数据

    split_ct_dataset(source_dir, target_dir)
    print("数据集划分完成！")