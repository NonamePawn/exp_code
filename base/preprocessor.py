# preprocessor.py (精简版 - 节省空间)
import os
import pydicom
import numpy as np
from pathlib import Path
from tqdm import tqdm


class ForensicsPreprocessor:
    def __init__(self, min_size: int = 256):
        self.min_size = min_size

    def read_dicom_raw(self, dcm_path: str) -> np.ndarray:
        try:
            dcm = pydicom.dcmread(dcm_path)
            image = dcm.pixel_array.astype(np.float32)
            intercept = getattr(dcm, 'RescaleIntercept', -1024)
            slope = getattr(dcm, 'RescaleSlope', 1)
            image = image * slope + intercept
            return image
        except:
            return None

    def run(self, src_root: str, dst_root: str):
        src_path = Path(src_root)
        dst_path = Path(dst_root)
        all_dcms = list(src_path.rglob("*.dcm"))

        print(f"Converting {len(all_dcms)} files to 1-Channel Raw .npy")

        for dcm_file in tqdm(all_dcms):
            raw_img = self.read_dicom_raw(str(dcm_file))
            if raw_img is None: continue

            # 尺寸过滤
            if raw_img.shape[0] < self.min_size or raw_img.shape[1] < self.min_size:
                continue

            # --- 关键修改：直接保存原始单通道数据，不做任何处理 ---
            # 这样体积最小，保留了所有原始信息
            relative_path = dcm_file.relative_to(src_path)
            save_path = dst_path / relative_path.with_suffix('.npy')
            save_path.parent.mkdir(parents=True, exist_ok=True)

            np.save(str(save_path), raw_img)


if __name__ == "__main__":
    # 路径配置保持不变 (使用 Docker 内部绝对路径)
    BASE_SRC = "/assets"
    BASE_DST = "/assets_processed"
    SUBSETS = ["train", "val", "test"]

    processor = ForensicsPreprocessor(min_size=256)

    for subset in SUBSETS:
        s_dir = os.path.join(BASE_SRC, subset)
        d_dir = os.path.join(BASE_DST, subset)
        if os.path.exists(s_dir):
            processor.run(s_dir, d_dir)