# Axial-GullyNet: Automatic Extraction of Erosion Gullies from Remote Sensing Images Using Axial Attention
<img width="2024" height="1206" alt="图3" src="https://github.com/user-attachments/assets/32408cb0-7f62-4632-9eac-c8f28723e240" />

作者时使用GF6卫星影像以侵蚀沟为目标制作了14106对数据集，8：1：1划分训练、验证和测试集的进行训练，结果如下：
<img width="1496" height="1183" alt="图片4" src="https://github.com/user-attachments/assets/c498f3f9-f557-42f0-bd1a-39faa620a152" />



<p align="center">

<img src="https://img.shields.io/badge/Python-3.8+-blue.svg">
<img src="https://img.shields.io/badge/PyTorch-2.x-red.svg">
<img src="https://img.shields.io/badge/Task-Semantic%20Segmentation-green.svg">
<img src="https://img.shields.io/badge/Application-Gully%20Extraction-orange.svg">

</p>


## 📌 Overview


**Axial-GullyNet** is a deep learning-based semantic segmentation framework designed for automatic extraction of erosion gullies from high-resolution remote sensing imagery.


The model is developed for monitoring erosion gullies in the **Northeast Black Soil Region of China**, where gullies usually present complex spatial characteristics, including narrow widths, discontinuous distributions, weak spectral differences, and strong interference from roads, rivers, and agricultural boundaries.


To address these challenges, Axial-GullyNet integrates an **Axial Attention mechanism** into the DeepLabV3+ framework, enhancing the ability of the network to model long-range spatial dependencies while maintaining detailed boundary information.


The proposed framework can effectively improve the continuity, completeness, and accuracy of gully extraction results from high-resolution satellite images.


---

# ✨ Main Contributions


The main features of this repository include:


- ✅ DeepLabV3+ semantic segmentation framework for gully extraction
- ✅ Axial Attention module for capturing long-range spatial relationships
- ✅ Improved recognition capability for narrow and fragmented gullies
- ✅ Complete training, prediction, and evaluation pipeline
- ✅ Support for large-scale remote sensing image inference
- ✅ Applicable to high-resolution satellite imagery such as GF-6 images

---

# 🏗 Network Architecture


The overall architecture of Axial-GullyNet is shown below:


```
Input Remote Sensing Image

          │

          ▼

     Feature Encoder

(Xception / MobileNetV2 / ResNet)

          │

          ▼

   Axial Attention Module

          │

          ▼

       ASPP Module

          │

          ▼

     Decoder Feature Fusion

          │

          ▼

     Gully Segmentation Mask

```

The Axial Attention module decomposes traditional 2D attention into two independent directional attentions:

- Horizontal attention
- Vertical attention

Compared with standard convolution operations, Axial Attention can:

- Capture global spatial dependencies
- Enhance feature representation of elongated gullies
- Improve continuity of extracted gully structures
- Preserve fine boundary details

---

# 📂 Repository Structure


```
Axial-GullyNet

│

├── train.py
│   └── Model training

│

├── predict.py
│   └── Single image prediction

│

├── predict_full_image.py
│   └── Large-scale remote sensing image prediction

│

├── get_miou.py
│   └── Accuracy evaluation

│

├── RP_Curve.py
│   └── Precision-Recall curve analysis

│

├── summary.py
│   └── Network structure visualization

│

├── nets

│   ├── deeplabv3_plus.py
│   │   └── DeepLabV3+ framework

│   ├── axial_attention.py
│   │   └── Axial Attention module

│   ├── xception.py
│   │   └── Xception backbone

│   ├── mobilenetv2.py
│   │   └── MobileNetV2 backbone

│   └── resnet.py
│       └── ResNet backbone

│

├── utils

│   ├── dataloader.py
│   ├── utils_fit.py
│   ├── utils_metrics.py
│   └── callbacks.py

│

├── voc_annotation.py
│   └── Dataset preparation

└── requirements.txt

```

---

# 🛠 Installation

## 1. Clone repository

```bash

git clone https://github.com/yourname/Axial-GullyNet.git

cd Axial-GullyNet

```
---

## 2. Create environment


Recommended environment:

```
Python >= 3.8

PyTorch >= 2.0

CUDA >= 11.8

```

Install dependencies:

```bash

pip install -r requirements.txt

```
Example:

```bash

pip install torch torchvision numpy opencv-python pillow matplotlib tqdm scipy

```
---

# 📊 Dataset Preparation


The dataset follows the VOC semantic segmentation format.


The directory structure should be:


```
VOCdevkit

└── VOC2007

    ├── JPEGImages

    │       └── image.jpg


    ├── SegmentationClass

    │       └── mask.png


    └── ImageSets

            └── Segmentation

                    ├── train.txt

                    └── val.txt

```

The label mask contains two categories:

| Pixel Value | Category |
|---|---|
|0|Background|
|1|Erosion gully|

---

# 🚀 Training


Modify the configuration parameters in:

```
train.py

```

Example:

```python

num_classes = 2

backbone = "xception"

input_shape = [512,512]

model_path = ""

```

Start training:

```bash

python train.py

```

The best model weights will be saved automatically:

```
logs/

```
---

# 🔍 Prediction


## Single image prediction


Run:


```bash

python predict.py

```

Output:

```
results/

├── prediction.png

└── mask.png

```
---

# 🌍 Large-scale Remote Sensing Image Prediction


For full satellite images, the framework provides sliding-window inference:

```bash

python predict_full_image.py

```

The prediction module supports:
- Large-size satellite images
- Sliding window cropping
- Overlapping prediction
- Automatic mask generation
This function is suitable for GF-1 and other high-resolution remote sensing images.

---

# 📈 Model Evaluation


Run:


```bash

python get_miou.py

```


The framework calculates:


- Mean Intersection over Union (mIoU)
- Pixel Accuracy (PA)
- Precision
- Recall
- F1-score


---

# 📌 Application


Axial-GullyNet is mainly designed for:


- Black soil erosion monitoring
- Agricultural land degradation assessment
- Remote sensing ecological survey
- Large-scale gully inventory mapping


Typical input data:


- GF-1 satellite imagery
- High-resolution optical remote sensing images


---

# 📧 Contact
For questions, suggestions, or collaboration:
Email:804638568@qq.com
---

## 🙏 致谢 (Acknowledgements)

*   本项目在早期开发与学习阶段，受益于 B 站 UP 主 **“东北Abner说AI”** 提供的 U-Net 模型课程，在此表示诚挚的感谢。
*   部分基础架构代码借鉴并改进自开源项目：[milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)。


We thank the open-source community for providing excellent deep learning resources.

# ⚠️ 中文声明（Chinese Statement）


## 原创性与使用声明


本项目 **Axial-GullyNet** 为作者针对遥感影像侵蚀沟智能提取任务自主设计与实现的深度学习模型框架，相关模型结构、改进方法、实验方案及数据处理流程均属于作者原创研究成果。


目前，与本项目相关的科研论文仍处于投稿审稿阶段，尚未正式公开发表。因此，在论文正式发表之前，未经作者明确授权，任何个人或组织不得对本项目中的模型结构、代码实现、实验结果、数据处理流程等内容进行以下行为：


- ❌ 直接复制、修改后作为自身研究成果发表；
- ❌ 用于论文、报告、专利或其他科研成果中的二次包装与署名替代；
- ❌ 删除作者信息后重新发布；
- ❌ 用于商业产品开发、商业服务或商业用途。


本项目代码仅用于：

- ✅ 学术交流；
- ✅ 方法学习；
- ✅ 非商业科研实验复现。


如需在科研工作中参考、修改或基于本项目开展进一步研究，请遵守以下要求：


1. 请保留原始代码中的作者信息及版权声明；
2. 请在相关论文、报告或研究成果中明确引用本项目；
3. 若使用本项目中的核心方法、模型结构或实验结果，请提前联系作者获得许可。


作者保留对本项目代码、模型结构及相关研究成果的最终解释权。


**Copyright © 2026 Axial-GullyNet Authors. All Rights Reserved.**

