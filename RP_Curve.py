import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import precision_recall_curve, average_precision_score
from torchvision import transforms

# ==========================================
# 0. 导入你的模型类
# ==========================================
from nets.deeplabv3_plus import DeepLab


def build_your_model():
    """
    在这里实例化你的网络模型。
    参数配置需与训练 14106 和 14306 权重时保持绝对一致。
    """
    model = DeepLab(
        num_classes=2,
        backbone="mobilenetv2",
        downsample_factor=8,
        pretrained=False,
        use_axial_attention=True,  # 开启轴向注意力
        use_strip_pooling=False  # 关闭条带池化
    )
    return model


# ==========================================
# 1. 配置路径、环境与绘图参数
# ==========================================
# 测试集原图文件夹
IMAGE_DIR = r"G:\model_code\deeplabv3-plus_ResNet_strip\20_Test_img"
# 测试集真实标签(GT)文件夹
GT_DIR = r"G:\model_code\deeplabv3-plus_ResNet_strip\20_OLD_Mask"

# 两个权重文件的具体路径
WEIGHT_14106 = r"G:\model_code\deeplabv3-plus_ResNet_strip\logs\loss_2026_03_04_09_00_58_MobileNetv2+轴向注意力\best_epoch_weights.pth"
WEIGHT_14306 = r"G:\model_code\deeplabv3-plus_ResNet_strip\logs\loss_2026_03_07_19_43_15_MobileNetv2+axial_14306\best_epoch_weights.pth"

# ------------------------------------------
# ⭐ 新增：绘图自定义参数 (可按期刊要求随时修改)
# ------------------------------------------
FIG_WIDTH_CM = 7.5  # 图片宽度 (厘米)
FIG_HEIGHT_CM = 4  # 图片高度 (厘米)
FIG_DPI = 600  # 图片分辨率 (通常核心期刊要求 300-600)
FONT_SIZE = 7.5  # 全局字体大小 (pt)
# ------------------------------------------

# 运行设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的计算设备: {DEVICE}")

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
])


def load_model_and_weights(weight_path):
    """实例化模型并加载指定权重"""
    model = build_your_model()  # 实例化模型
    state_dict = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def main():
    print(">>> 步骤 1: 正在加载模型 1 (无难负样本 14106) ...")
    model_14106 = load_model_and_weights(WEIGHT_14106)

    print(">>> 步骤 2: 正在加载模型 2 (有难负样本 14306) ...")
    model_14306 = load_model_and_weights(WEIGHT_14306)

    # 存放全局展平像素的列表
    y_true_all = []
    y_scores_14106_all = []
    y_scores_14306_all = []

    # 获取所有图片文件名
    img_names = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.tif', '.jpg'))]
    print(f">>> 步骤 3: 开始对 {len(img_names)} 张测试图像进行推理与概率提取...")

    # 关闭梯度计算，节省显存
    with torch.no_grad():
        for idx, img_name in enumerate(img_names):
            # --- 读取真实标签 ---
            gt_name = img_name.split('.')[0] + ".png"  # 根据实际后缀修改
            gt_path = os.path.join(GT_DIR, gt_name)

            gt_mask = np.array(Image.open(gt_path))
            gt_mask = (gt_mask > 0).astype(np.uint8)  # 二值化：侵蚀沟为1，背景为0

            # --- 读取原图并预处理 ---
            img_path = os.path.join(IMAGE_DIR, img_name)
            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

            # --- 模型 1 推理 (14106) ---
            out_1 = model_14106(img_tensor)
            prob_1 = torch.softmax(out_1, dim=1)
            gully_prob_1 = prob_1[0, 1, :, :].cpu().numpy()

            # --- 模型 2 推理 (14306) ---
            out_2 = model_14306(img_tensor)
            prob_2 = torch.softmax(out_2, dim=1)
            gully_prob_2 = prob_2[0, 1, :, :].cpu().numpy()

            # --- 展平并收集数据 ---
            y_true_all.append(gt_mask.flatten())
            y_scores_14106_all.append(gully_prob_1.flatten())
            y_scores_14306_all.append(gully_prob_2.flatten())

            print(f"  - 已处理 [{idx + 1}/{len(img_names)}]: {img_name}")

    print(">>> 步骤 4: 正在拼接像素数据并释放内存...")
    y_true = np.concatenate(y_true_all)
    y_scores_14106 = np.concatenate(y_scores_14106_all)
    y_scores_14306 = np.concatenate(y_scores_14306_all)

    del y_true_all, y_scores_14106_all, y_scores_14306_all

    print(">>> 步骤 5: 正在计算 Precision 和 Recall...")
    precision_1, recall_1, _ = precision_recall_curve(y_true, y_scores_14106)
    ap_1 = average_precision_score(y_true, y_scores_14106)

    precision_2, recall_2, _ = precision_recall_curve(y_true, y_scores_14306)
    ap_2 = average_precision_score(y_true, y_scores_14306)

    print(">>> 步骤 6: 正在绘制符合期刊规范的 PR 曲线...")

    # 1. 设置全局字体样式和字号
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = FONT_SIZE
    # 设置坐标轴的线宽，适应小图
    plt.rcParams['axes.linewidth'] = 0.6

    # 2. 将全局配置的厘米尺寸转换为英寸 (1 inch = 2.54 cm)
    cm_to_inch = 1 / 2.54
    fig_width_inch = FIG_WIDTH_CM * cm_to_inch
    fig_height_inch = FIG_HEIGHT_CM * cm_to_inch

    # 3. 使用自定义的尺寸和 DPI 创建画布
    plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=FIG_DPI)

    # 4. 绘制曲线
    plt.plot(recall_1, precision_1, color='#1f77b4', lw=1.2, alpha=0.9,
             label=f'14106 (AP: {ap_1:.4f})')
    plt.plot(recall_2, precision_2, color='#d62728', lw=1.2, alpha=0.9,
             label=f'14306 (AP: {ap_2:.4f})')

    plt.title('Precision-Recall Curve Comparison')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])

    # 网格线设置
    plt.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)

    # 图例设置
    plt.legend(loc="lower left", frameon=True, shadow=False)

    # 减少留白边缘
    plt.tight_layout(pad=0.3)

    # 保存图片
    save_path = r'F:\论文\中文小论文\编辑批注修改0615\PR_Curve_Comparison_Journal.tif'
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    print(f"\n✅ 绘图完成！图表已成功保存为: {os.path.abspath(save_path)}")
    print(f"📐 当前输出尺寸: 宽 {FIG_WIDTH_CM} cm, 高 {FIG_HEIGHT_CM} cm, 分辨率 {FIG_DPI} DPI")

    plt.show()


if __name__ == "__main__":
    main()