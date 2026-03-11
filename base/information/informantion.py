import pydicom
import matplotlib.pyplot as plt
import os
import glob

# 1. 获取当前目录下所有的 dcm 文件
file_path = "./"  # 当前文件夹
dcm_files = glob.glob(os.path.join(file_path, "*.dcm"))

if not dcm_files:
    print("未在当前目录下找到 .dcm 文件，请检查路径。")
else:
    print(f"找到 {len(dcm_files)} 个 DICOM 文件。\n")

    # 2. 读取并展示第一个文件作为示例
    sample_file = dcm_files[0]
    ds = pydicom.dcmread(sample_file)

    # 3. 提取并打印基本信息
    print("--- 图像基本信息 ---")
    print(f"检查模态: {ds.Modality}")
    print(f"图像尺寸: {ds.Rows} x {ds.Columns}")
    print(f"厂商信息：{ds.Manufacturer}")
    print(f"设备信号：{ds.ManufacturerModelName}")
    print("------------------\n")

    # 4. 展示图像
    # 注意：某些 DICOM 图像需要处理 Window Center/Width 才能正常显示
    plt.figure(figsize=(8, 8))
    plt.imshow(ds.pixel_array, cmap="gray") # 使用骨窗/灰色调
    plt.title(f"DICOM Preview: {os.path.basename(sample_file)}")
    plt.axis('off')
    plt.show()