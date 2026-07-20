import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
'''
if __name__ == "__main__":
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    # ---------------------------------------------------------------------------#
    miou_mode = 0
    # ------------------------------#
    #   分类个数+1、如2+1
    # ------------------------------#
    num_classes = 2
    # --------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    # --------------------------------------------#
    name_classes = ["_background_", "gully"]
    # -------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    # -------------------------------------------------------#
    VOCdevkit_path = 'VOCdevkit'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/test.txt"), 'r').read().splitlines()
    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    # ---------------------------------------------------------#
    #   新增：定义可视化图片的输出路径
    # ---------------------------------------------------------#
    vis_dir = os.path.join(miou_out_path, 'visualization-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        # 创建可视化文件夹
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        print("Load model.")
        deeplab = DeeplabV3()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            # 1. 读取原图并进行预测
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            pr_image = deeplab.get_miou_png(image)  # 这是带有 0 和 1 的 PIL Image

            # 保存给 compute_mIoU 计算指标用的单通道灰度图
            pr_image.save(os.path.join(pred_dir, image_id + ".png"))

            # ---------------------------------------------------------#
            #   新增：生成并保存误检(红)、漏检(蓝)、正确(白)的可视化图像
            # ---------------------------------------------------------#
            gt_path = os.path.join(gt_dir, image_id + ".png")
            if os.path.exists(gt_path):
                # 转换为 numpy 数组进行逻辑运算
                pr_np = np.array(pr_image)
                gt_np = np.array(Image.open(gt_path))

                # 初始化一个全黑的 RGB 图像矩阵 (高度, 宽度, 3通道)
                h, w = gt_np.shape
                vis_np = np.zeros((h, w, 3), dtype=np.uint8)

                # 定义颜色 (RGB格式)
                COLOR_TP = [255, 255, 255]  # 白色：正确预测出目标 (True Positive)
                COLOR_FP = [255, 0, 0]  # 亮红色：误检区域，把背景预测成了目标 (False Positive)
                COLOR_FN = [0, 0, 255]  # 亮蓝色：漏检区域，目标未被预测出来 (False Negative)

                # 赋予对应像素颜色 (基于目标类别值为 1，背景为 0 进行判断)
                vis_np[(gt_np == 1) & (pr_np == 1)] = COLOR_TP
                vis_np[(gt_np == 0) & (pr_np == 1)] = COLOR_FP
                vis_np[(gt_np == 1) & (pr_np == 0)] = COLOR_FN

                # 将 numpy 数组转回 PIL Image 并保存到专属可视化文件夹
                vis_pil = Image.fromarray(vis_np)
                vis_pil.save(os.path.join(vis_dir, image_id + "_vis.png"))

        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

        # ---------------------------------------------------------#
        #   计算 F1-score
        # ---------------------------------------------------------#
        epsilon = 1e-7
        F1_scores = 2 * (Precision * PA_Recall) / (Precision + PA_Recall + epsilon)

        print("\n" + "-" * 45)
        print("|%15s | %25s|" % ("Class Name", "F1-Score"))
        print("-" * 45)

        with open(os.path.join(miou_out_path, "F1_score.txt"), 'w') as f:
            for i in range(num_classes):
                name = str(name_classes[i])
                f1 = F1_scores[i]
                print("|%15s | %24.2f%%|" % (name, f1 * 100))
                print("-" * 45)
                f.write(f"{name}: {f1}\n")

            mF1 = np.nanmean(F1_scores)
            print("|%15s | %24.2f%%|" % ("mF1 (Mean F1)", mF1 * 100))
            print("-" * 45)
            f.write(f"mF1: {mF1}\n")