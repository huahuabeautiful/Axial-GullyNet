import torch
import torch.nn as nn
import torch.nn.functional as F


class AxialAttention(nn.Module):
    """
    轴向注意力机制 (修正版)
    同时考虑高度和宽度方向的空间注意力，用于修复侵蚀沟断裂
    """

    def __init__(self, in_channels, groups=8):
        super(AxialAttention, self).__init__()
        self.groups = groups
        self.in_channels = in_channels

        # 高度方向的注意力
        self.height_query = nn.Conv2d(in_channels, in_channels // groups, kernel_size=1)
        self.height_key = nn.Conv2d(in_channels, in_channels // groups, kernel_size=1)
        self.height_value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 宽度方向的注意力
        self.width_query = nn.Conv2d(in_channels, in_channels // groups, kernel_size=1)
        self.width_key = nn.Conv2d(in_channels, in_channels // groups, kernel_size=1)
        self.width_value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 输出投影
        self.proj = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # ================= 高度方向注意力 (Height) =================
        # Q: (B, W, H, C//g) -> (B*W, H, C//g)
        query_h = self.height_query(x).permute(0, 3, 2, 1).contiguous().view(batch_size * width, height, -1)
        # K: (B, W, C//g, H) -> (B*W, C//g, H)
        key_h = self.height_key(x).permute(0, 3, 1, 2).contiguous().view(batch_size * width, -1, height)
        # V: (B, W, H, C) -> (B*W, H, C)
        value_h = self.height_value(x).permute(0, 3, 2, 1).contiguous().view(batch_size * width, height, -1)

        # 注意力矩阵: (B*W, H, C//g) x (B*W, C//g, H) -> (B*W, H, H)
        attention_h = torch.bmm(query_h, key_h)
        attention_h = F.softmax(attention_h, dim=-1)

        # 应用注意力: (B*W, H, H) x (B*W, H, C) -> (B*W, H, C)
        out_h = torch.bmm(attention_h, value_h)
        # 还原形状: (B*W, H, C) -> (B, W, H, C) -> (B, C, H, W)
        out_h = out_h.view(batch_size, width, height, -1).permute(0, 3, 2, 1)

        # ================= 宽度方向注意力 (Width) =================
        # Q: (B, H, W, C//g) -> (B*H, W, C//g)
        query_w = self.width_query(x).permute(0, 2, 3, 1).contiguous().view(batch_size * height, width, -1)
        # K: (B, H, C//g, W) -> (B*H, C//g, W)
        key_w = self.width_key(x).permute(0, 2, 1, 3).contiguous().view(batch_size * height, -1, width)
        # V: (B, H, W, C) -> (B*H, W, C)
        value_w = self.width_value(x).permute(0, 2, 3, 1).contiguous().view(batch_size * height, width, -1)

        # 注意力矩阵: (B*H, W, C//g) x (B*H, C//g, W) -> (B*H, W, W)
        attention_w = torch.bmm(query_w, key_w)
        attention_w = F.softmax(attention_w, dim=-1)

        # 应用注意力: (B*H, W, W) x (B*H, W, C) -> (B*H, W, C)
        out_w = torch.bmm(attention_w, value_w)
        # 还原形状: (B*H, W, C) -> (B, H, W, C) -> (B, C, H, W)
        out_w = out_w.view(batch_size, height, width, -1).permute(0, 3, 1, 2)

        # ================= 融合两个方向的特征 =================
        out = torch.cat([out_h, out_w], dim=1)
        out = self.proj(out)

        return self.gamma * out + x