import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2
# 导入我们刚刚创建的 resnet
from nets.resnet import resnet50, resnet101

# 尝试导入mobilenetv3（如果你本地有的话）
try:
    from nets.mobilenetv3 import mobilenetv3_large, mobilenetv3_small
except ImportError:
    pass

# 导入轴向注意力模块(保留你原有的逻辑)
try:
    from .axial_attention import AxialAttention
except ImportError:
    try:
        from nets.axial_attention import AxialAttention
    except ImportError:
        pass


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


# ==================== 新增：条带池化分支模块 ====================
class StripPoolingBranch(nn.Module):
    def __init__(self, dim_in, dim_out, bn_mom=0.1):
        super(StripPoolingBranch, self).__init__()
        # 水平方向条带池化 (1 x W)
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))
        # 垂直方向条带池化 (H x 1)
        self.pool_w = nn.AdaptiveAvgPool2d((None, 1))

        self.conv_h = nn.Conv2d(dim_in, dim_out, 1, bias=False)
        self.conv_w = nn.Conv2d(dim_in, dim_out, 1, bias=False)

        # 融合后的卷积层
        self.conv_out = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        _, _, h, w = x.size()

        # 提取水平方向连续特征并恢复尺寸
        x_h = self.pool_h(x)
        x_h = self.conv_h(x_h)
        x_h = F.interpolate(x_h, size=(h, w), mode='bilinear', align_corners=True)

        # 提取垂直方向连续特征并恢复尺寸
        x_w = self.pool_w(x)
        x_w = self.conv_w(x_w)
        x_w = F.interpolate(x_w, size=(h, w), mode='bilinear', align_corners=True)

        # 将两个方向的特征相加，融合长距离上下文
        return self.conv_out(x_h + x_w)


# ==============================================================


class ASPP(nn.Module):
    # 修改：新增 use_strip_pooling 参数控制是否使用条带池化
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1, use_strip_pooling=False):
        super(ASPP, self).__init__()
        self.use_strip_pooling = use_strip_pooling

        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        # 动态决定通道数和实例化分支
        if self.use_strip_pooling:
            self.branch6_strip = StripPoolingBranch(dim_in, dim_out, bn_mom=bn_mom)
            cat_channels = dim_out * 6  # 开启时为6个分支
        else:
            cat_channels = dim_out * 5  # 关闭时为传统的5个分支

        self.conv_cat = nn.Sequential(
            nn.Conv2d(cat_channels, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)

        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   根据标志位动态拼接分支的内容
        # -----------------------------------------#
        if self.use_strip_pooling:
            strip_feature = self.branch6_strip(x)
            feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature, strip_feature], dim=1)
        else:
            feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)

        result = self.conv_cat(feature_cat)
        return result


class DeepLab(nn.Module):
    # 修改：在初始化传参列表里保留了 use_axial_attention 和 use_strip_pooling
    def __init__(self, num_classes, backbone="mobilenetv2", pretrained=True, downsample_factor=16,
                 use_axial_attention=False, use_strip_pooling=False):
        super(DeepLab, self).__init__()

        # ---------------------------------- #
        #   判断使用的主干网络并获取特征
        # ---------------------------------- #
        if backbone == "xception":
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256

        elif backbone == "mobilenetv2":
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24

        elif backbone == "mobilenetv3":
            try:
                self.backbone = mobilenetv3_large(pretrained=pretrained, downsample_factor=downsample_factor)
                in_channels = 160
                low_level_channels = 24
            except NameError:
                raise ValueError("未找到mobilenetv3的实现，请检查nets/mobilenetv3.py是否存在。")

        # ================================== #
        #   新增的 ResNet50 与 ResNet101 分支
        # ================================== #
        elif backbone == "resnet50":
            self.backbone = resnet50(pretrained=pretrained, downsample_factor=downsample_factor)
            in_channels = 2048
            low_level_channels = 256

        elif backbone == "resnet101":
            self.backbone = resnet101(pretrained=pretrained, downsample_factor=downsample_factor)
            in_channels = 2048
            low_level_channels = 256
        # ================================== #

        else:
            raise ValueError(
                'Unsupported backbone - `{}`, Use mobilenetv2, xception, resnet50, resnet101...'.format(backbone))

        # -----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        #   修改：将参数传给 ASPP
        # -----------------------------------------#
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor,
                         use_strip_pooling=use_strip_pooling)

        # ----------------------------------#
        #   浅层特征边
        # ----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )

        # ========================================== #
        #   新增：实例化轴向注意力机制
        #   在 cat_conv 之后，输入通道数为 256
        # ========================================== #
        self.use_axial_attention = use_axial_attention
        if self.use_axial_attention:
            self.axial_attention = AxialAttention(in_channels=256)

        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        # -----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        # -----------------------------------------#
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)

        # -----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        # -----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))

        # ========================================== #
        #   新增：在此处施加轴向注意力机制
        #   利用十字交叉的注意力，修复侵蚀沟的断裂点，强化分叉结构
        # ========================================== #
        if self.use_axial_attention:
            x = self.axial_attention(x)

        # -----------------------------------------#
        #   获得预测结果并上采样至原图大小
        # -----------------------------------------#
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x