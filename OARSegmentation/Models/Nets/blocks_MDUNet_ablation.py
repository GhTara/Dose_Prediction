from __future__ import absolute_import, print_function

import torch
import torch.nn as nn


class conv_block_3(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.Mish(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.Mish(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block_7(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_7, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.Mish(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.Mish(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_3_1(nn.Module):
    def __init__(self, ch_in, ch_out, act):
        super(conv_3_1, self).__init__()

        # self.conv_1 = conv_block_1(ch_in, ch_out)
        # self.conv_2 = conv_block_2(ch_in, ch_out)
        self.conv_3 = nn.Sequential(
            conv_block_3(ch_in, ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.Mish(inplace=True) if act == 'relu' else nn.Mish(inplace=True)
        )
        # self.conv_5 = conv_block_5(ch_in, ch_out)
        self.conv_7 = nn.Sequential(
            conv_block_7(ch_in, ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True) if act == 'relu' else nn.Mish(inplace=True)
        )
        # self.conv_9 = conv_block_9(ch_in, ch_out)

        self.conv = nn.Sequential(
            nn.Conv3d(ch_out * 2, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True) if act == 'relu' else nn.Mish(inplace=True)
        )

    def forward(self, x):
        # x1 = self.conv_1(x)
        # x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        # x5 = self.conv_5(x)
        x7 = self.conv_7(x)
        # x9 = self.conv_9(x)

        x = torch.cat((x3, x7), dim=1)
        x = self.conv(x)

        return x


class dilated_conv_block_5(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super(dilated_conv_block_5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            # if act == 'relu' else nn.Mish(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            # if act == 'relu' else nn.Mish(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class dilated_conv_block_7(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super(dilated_conv_block_7, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=3, dilation=3, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            # if act == 'relu' else nn.Mish(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=3, dilation=3, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            # if act == 'relu' else nn.Mish(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DualDilatedBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DualDilatedBlock, self).__init__()

        self.conv_3 = conv_block_3(ch_in, ch_out)
        self.conv_5 = dilated_conv_block_5(ch_in, ch_out)
        self.conv_7 = dilated_conv_block_7(ch_in, ch_out)

        self.conv = nn.Sequential(
            nn.Conv3d(ch_out * 3, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            # if act == 'relu' else nn.Mish(inplace=True)
        )

    def forward(self, x):
        x3 = self.conv_3(x)
        x5 = self.conv_5(x)
        x7 = self.conv_7(x)

        x = torch.cat((x3, x5, x7), dim=1)
        x = self.conv(x)

        return x
