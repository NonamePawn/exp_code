import os
import pydicom
from tqdm import tqdm

# 这里填你报错时正在读取的数据集路径
# 根据你的代码，应该在 '/assets' 下的 train, val 或 test
DATA_ROOT = '/assets'


def check_dicom_files(root_dir):
    print(f"正在扫描 {root_dir} 下的所有 DICOM 文件...")
    bad_files = []

    # 遍历目录
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.dcm') or f.endswith('.DCM'):  # 假设后缀是dcm，如果没后缀也可以去掉这个if
                file_list.append(os.path.join(root, f))

    print(f"共找到 {len(file_list)} 个文件，开始体检...")

    # 使用进度条遍历读取
    for file_path in tqdm(file_list):
        try:
            ds = pydicom.dcmread(file_path)
            # 关键：尝试读取 pixel_array，这会触发解码，从而检测文件是否损坏
            _ = ds.pixel_array
        except Exception as e:
            print(f"\n❌ 发现坏文件: {file_path}")
            print(f"   错误信息: {e}")
            bad_files.append(file_path)

    print("\n" + "=" * 30)
    print(f"扫描结束！共发现 {len(bad_files)} 个损坏文件。")

    if len(bad_files) > 0:
        print("建议执行以下命令删除它们：")
        for bad in bad_files:
            print(f"rm {bad}")


if __name__ == '__main__':
    # 重点扫描 train 目录，因为你是在训练阶段报错的
    check_dicom_files(os.path.join(DATA_ROOT, 'train'))
    # 如果不放心，也可以把下面两行取消注释，扫描验证集和测试集
    # check_dicom_files(os.path.join(DATA_ROOT, 'val'))
    # check_dicom_files(os.path.join(DATA_ROOT, 'test'))