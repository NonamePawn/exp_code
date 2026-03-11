import os
import pydicom
import pandas as pd
from tqdm import tqdm
from collections import Counter


def check_metadata_consistency(src_root, sample_per_patient=2):
    """
    检查数据集中的元数据填充情况
    - src_root: 数据集根目录
    - sample_per_patient: 每个病人抽查多少张切片（提高效率）
    """
    report = []
    classes = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]

    print(f"开始普查元数据，共发现 {len(classes)} 个类别...")

    for cls in classes:
        cls_path = os.path.join(src_root, cls)
        patients = [p for p in os.listdir(cls_path) if os.path.isdir(os.path.join(cls_path, p))]

        # 统计指标
        total_patients = len(patients)
        tag_found_count = 0
        body_part_values = []

        # 备选方案统计（如果 BodyPartExamined 缺失，看看有没有别的标签能用）
        alternative_tags = {'SeriesDescription': 0, 'ProtocolName': 0}

        for pt in tqdm(patients, desc=f"Scanning {cls}", leave=False):
            pt_path = os.path.join(cls_path, pt)
            dcm_files = [f for f in os.listdir(pt_path) if f.lower().endswith('.dcm')]

            if not dcm_files:
                continue

            # 抽样检查
            samples = dcm_files[:sample_per_patient]
            pt_has_tag = False

            for s in samples:
                ds = pydicom.dcmread(os.path.join(pt_path, s), stop_before_pixels=True)  # 只读头部，不读图像，极快

                # 检查主标签：BodyPartExamined
                val = getattr(ds, 'BodyPartExamined', None)
                if val and str(val).strip():
                    body_part_values.append(str(val).strip().upper())
                    pt_has_tag = True

                # 检查备选标签
                if hasattr(ds, 'SeriesDescription'): alternative_tags['SeriesDescription'] += 1
                if hasattr(ds, 'ProtocolName'): alternative_tags['ProtocolName'] += 1

            if pt_has_tag:
                tag_found_count += 1

        # 计算该类别的填充率
        fill_rate = (tag_found_count / total_patients) * 100 if total_patients > 0 else 0
        common_vals = Counter(body_part_values).most_common(3)  # 取出现频率最高的3个值

        report.append({
            'Category': cls,
            'TotalPatients': total_patients,
            'FillRate (%)': f"{fill_rate:.1f}%",
            'TopValues': common_vals,
            'HasSeriesDesc': f"{alternative_tags['SeriesDescription'] / (total_patients * sample_per_patient) * 100:.1f}%"
        })

    # 输出结果表格
    df = pd.DataFrame(report)
    print("\n" + "=" * 50)
    print("元数据质量普查报告")
    print("=" * 50)
    print(df.to_string(index=False))

    return df


if __name__ == "__main__":
    # 执行验证
    raw_data_path = r'D:\exp_data'
    results_df = check_metadata_consistency(raw_data_path)

    # 也可以保存为 CSV 方便在 Excel 里看
    # results_df.to_csv("metadata_report.csv", index=False)