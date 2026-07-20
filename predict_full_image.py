import os
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from PIL import Image

from deeplab import DeeplabV3
from utils.utils import cvtColor, preprocess_input


def stretch_to_8bit(img_array):
    """
    【修復模組】：將 16位 遙感影像做 2% - 98% 線性拉伸並轉換為 8位 (兼容 PIL 和神經網路)
    """
    img_array = np.clip(img_array, 0, None)  # 消除負數 NoData
    out = np.zeros_like(img_array, dtype=np.uint8)
    for i in range(img_array.shape[2]):
        band = img_array[:, :, i]
        min_val = np.percentile(band, 2)
        max_val = np.percentile(band, 98)
        # 避免分母為 0 的極端情況
        if max_val - min_val < 1e-8:
            out[:, :, i] = 0
        else:
            stretched = (band - min_val) / (max_val - min_val) * 255.0
            out[:, :, i] = np.clip(stretched, 0, 255).astype(np.uint8)
    return out


def process_block(img_array, model, patch_size, stride, batch_size, device):
    """【核心子模組】：對單個數據塊進行重疊滑窗預測與機率融合"""
    h, w, c = img_array.shape

    # 邊緣 Padding，確保滑窗能覆蓋到底
    pad_h = math.ceil((h - patch_size) / stride) * stride + patch_size - h if h > patch_size else patch_size - h
    pad_w = math.ceil((w - patch_size) / stride) * stride + patch_size - w if w > patch_size else patch_size - w
    if pad_h < 0: pad_h = 0
    if pad_w < 0: pad_w = 0

    padded_img = np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    padded_h, padded_w = padded_img.shape[:2]

    # 這兩個矩陣只佔用極小的記憶體
    prob_map = np.zeros((padded_h, padded_w), dtype=np.float32)
    count_map = np.zeros((padded_h, padded_w), dtype=np.float32)

    y_coords = list(range(0, padded_h - patch_size + 1, stride))
    x_coords = list(range(0, padded_w - patch_size + 1, stride))

    batch_imgs = []
    batch_coords = []

    def _predict_batch(imgs, coords):
        img_tensor = torch.from_numpy(np.stack(imgs, axis=0)).to(device)
        with torch.no_grad():
            # 保持完整的 [Batch, Class, H, W] 4维输出
            outputs = model(img_tensor)
            # 提取类别 1 (目标/侵蚀沟) 的机率
            probs = F.softmax(outputs, dim=1)[:, 1, :, :].cpu().numpy()
        for i, (yy, xx) in enumerate(coords):
            prob_map[yy:yy+patch_size, xx:xx+patch_size] += probs[i]
            count_map[yy:yy+patch_size, xx:xx+patch_size] += 1.0

    for yy in y_coords:
        for xx in x_coords:
            patch = padded_img[yy:yy + patch_size, xx:xx + patch_size, :]
            patch_pil = cvtColor(Image.fromarray(patch))
            patch_data = np.transpose(preprocess_input(np.array(patch_pil, np.float32)), (2, 0, 1))
            batch_imgs.append(patch_data)
            batch_coords.append((yy, xx))

            # 滿一個 Batch 就推理
            if len(batch_imgs) == batch_size:
                _predict_batch(batch_imgs, batch_coords)
                batch_imgs = []
                batch_coords = []

    # 處理尾部數據
    if len(batch_imgs) > 0:
        _predict_batch(batch_imgs, batch_coords)

    count_map[count_map == 0] = 1
    avg_prob_map = prob_map / count_map
    binary_mask = (avg_prob_map > 0.5).astype(np.uint8)

    # 剝離 Padding，返回當前塊的原始尺寸
    return binary_mask[:h, :w]


def predict_large_image_blockwise(image_path, out_path, deeplab, patch_size=256, overlap_rate=0.5, batch_size=16,
                                  block_size=4096):
    """分塊讀取超大影像並進行無縫預測拼圖 (高速版)"""
    print(f"\n🚀 開始分塊處理超大影像: {os.path.basename(image_path)}")
    start_time = time.time()

    model = deeplab.net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stride = int(patch_size * (1 - overlap_rate))

    # 緩衝區大小：每次讀取時向外多讀一圈，保證拼接邊界擁有足夠的上下文特徵
    buffer_size = patch_size

    with rasterio.open(image_path) as src:
        meta = src.meta.copy()
        width, height = src.width, src.height
        res_x, res_y = abs(src.transform[0]), abs(src.transform[4])

        print(f"📊 影像尺寸: {width} x {height} 像素")
        print(f"🌍 實際覆蓋面積: {(width * res_x * height * res_y) / 1e6:.2f} 平方公里 (km²)")

        # 更新元資料為單通道壓縮掩膜
        meta.update({"count": 1, "dtype": 'uint8', "compress": 'lzw'})
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with rasterio.open(out_path, 'w', **meta) as dst:
            y_blocks = list(range(0, height, block_size))
            x_blocks = list(range(0, width, block_size))
            total_blocks = len(y_blocks) * len(x_blocks)

            print(f"📦 影像已被劃分為 {total_blocks} 個數據塊 (Block Size: {block_size}) 進行高速推理...")
            pbar = tqdm(total=total_blocks, desc="正在分塊推理")

            for y in y_blocks:
                for x in x_blocks:
                    # 1. 計算帶緩衝區的讀取窗口 (Buffer Window)
                    read_x = max(0, x - buffer_size)
                    read_y = max(0, y - buffer_size)
                    read_w = min(width - read_x, x + block_size + buffer_size - read_x)
                    read_h = min(height - read_y, y + block_size + buffer_size - read_y)

                    window = Window(read_x, read_y, read_w, read_h)

                    # 從硬碟按需讀取
                    img_array = src.read([1, 2, 3], window=window)
                    img_array = np.transpose(img_array, (1, 2, 0))

                    # ========================================================
                    # 【關鍵修復點】：16位轉8位拉伸，完美解決 PIL 無法處理 <i2 的問題
                    # ========================================================
                    if img_array.dtype != np.uint8:
                        img_array = stretch_to_8bit(img_array)

                    # 2. 對當前塊進行深度學習預測
                    block_mask = process_block(img_array, model, patch_size, stride, batch_size, device)

                    # 3. 剝離邊緣的緩衝區，只保留最中心精確的有效區域
                    crop_y_start = y - read_y
                    crop_x_start = x - read_x
                    valid_h = min(block_size, height - y)
                    valid_w = min(block_size, width - x)

                    final_mask_block = block_mask[
                        crop_y_start: crop_y_start + valid_h, crop_x_start: crop_x_start + valid_w]

                    # 4. 直接寫入輸出 TIF 的對應位置
                    out_window = Window(x, y, valid_w, valid_h)
                    dst.write(final_mask_block, 1, window=out_window)

                    pbar.update(1)
            pbar.close()

    time_cost = time.time() - start_time
    print(f"🎉 提取完成！")
    print(f"⏱️ 總耗時: {time_cost // 3600:.0f} 小時 {(time_cost % 3600) // 60:.0f} 分 {time_cost % 60:.0f} 秒")
    print(f"💾 結果已保存至: {out_path}\n")


if __name__ == "__main__":
    # ==========================================
    # 實際運行配置區域
    # ==========================================
    print("⏳ 正在載入 DeepLabV3+ 模型權重...")
    deeplab_model = DeeplabV3()

    # 原始大圖的路徑
    INPUT_IMAGE = r"G:\model_code\filetreatpython\GFdata\GF6_nenjiang\GF6dat\NJ_NND.dat"
    # 輸出的完整 TIF 掩膜路徑 (請確保以 .tif 結尾)
    OUTPUT_MASK = r"G:\model_code\filetreatpython\nejiang\Full_image_predict\GF6_predict.tif"

    predict_large_image_blockwise(
        image_path=INPUT_IMAGE,
        out_path=OUTPUT_MASK,
        deeplab=deeplab_model,
        patch_size=256,
        overlap_rate=0.5,
        batch_size=16,  # 根據你的顯存調整，12G顯存可設為 16 或 32
        block_size=4096  # 4096 是一個非常平衡的值，既不會 OOM 也能保證高速
    )