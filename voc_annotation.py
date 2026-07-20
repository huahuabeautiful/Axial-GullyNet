import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

# -------------------------------------------------------#
#   【配置区域】
#   这里直接设置 训练集、验证集、测试集 的比例
#   保证三者之和为 1 即可
# -------------------------------------------------------#
train_percent = 0.8  # 训练集 80%
val_percent = 0.1  # 验证集 10%
test_percent = 0.1  # 测试集 10%

# -------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
# -------------------------------------------------------#
VOCdevkit_path = 'VOCdevkit'

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')

    if not os.path.exists(saveBasePath):
        os.makedirs(saveBasePath)

    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num = len(total_seg)
    list_index = list(range(num))

    # 计算具体的数量
    # train_percent 对应具体的训练集数量
    num_train = int(num * train_percent)
    # val_percent 对应具体的验证集数量
    num_val = int(num * val_percent)
    # 剩下的都给测试集，防止因取整导致的数量对不上
    num_test = num - num_train - num_val

    # 打印分配情况
    print(f"总数量: {num}")
    print(f"训练集 (train): {num_train}")
    print(f"验证集 (val)  : {num_val}")
    print(f"测试集 (test) : {num_test}")
    print(f"训练+验证 (trainval): {num_train + num_val}")

    # 打乱索引
    random.shuffle(list_index)

    # 按照计算的数量进行切片
    train_indices = list_index[:num_train]
    val_indices = list_index[num_train: num_train + num_val]
    test_indices = list_index[num_train + num_val:]

    # 建立集合以便快速查找（用于生成 trainval）
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)

    # 打开文件准备写入
    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

    for i in list_index:
        name = total_seg[i][:-4] + '\n'

        if i in train_set:
            ftrain.write(name)
            ftrainval.write(name)  # 训练集属于 trainval
        elif i in val_set:
            fval.write(name)
            ftrainval.write(name)  # 验证集属于 trainval
        else:
            ftest.write(name)  # 测试集独立

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("Generate txt in ImageSets done.")

    print("Check datasets format, this may take a while.")
    print("检查数据集格式是否符合要求，这可能需要一段时间。")
    # 【修复】这里将 np.int 修改为 int，解决了报错问题
    classes_nums = np.zeros([256], int)

    for i in tqdm(list_index):
        name = total_seg[i]
        png_file_name = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("未检测到标签图片%s，请查看具体路径下文件是否存在以及后缀是否为png。" % (png_file_name))

        png = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("标签图片%s的shape为%s，不属于灰度图或者八位彩图，请仔细检查数据集格式。" % (name, str(np.shape(png))))
            print("标签图片需要为灰度图或者八位彩图，标签的每个像素点的值就是这个像素点所属的种类。" % (
            name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)

    print("打印像素点的值与数量。")
    print('-' * 37)
    print("| %15s | %15s |" % ("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |" % (str(i), str(classes_nums[i])))
            print('-' * 37)

    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("检测到标签中像素点的值仅包含0与255，数据格式有误。")
        print("二分类问题需要将标签修改为背景的像素点值为0，目标的像素点值为1。")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("检测到标签中仅仅包含背景像素点，数据格式有误，请仔细检查数据集格式。")

    print("JPEGImages中的图片应当为.jpg文件、SegmentationClass中的图片应当为.png文件。")
    print("如果格式有误，参考:")
    print("https://github.com/bubbliiiing/segmentation-format-fix")