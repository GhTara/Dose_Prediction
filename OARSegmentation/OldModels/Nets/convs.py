import torch
import torch.nn as nn


class conv_block(nn.Module):
    """
    Convolutional block with one convolutional layer
    and ReLU activation function.
    """

    def __init__(self, ch_in, ch_out, kernel_size, padding=1, bias=False):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_trans_block(nn.Module):
    """
    Convolutional block with one convolutional layer
    and ReLU activation function.
    """

    def __init__(self, ch_in, ch_out, kernel_size, padding=1, bias=False):
        super(conv_trans_block, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(ch_in, ch_out, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class MultiScaleConv(nn.Module):
    """
      Multiscale convolutional block with 3 convolutional blocks
      with kernel size of 3x3, 5x5 and 7x7. Which is then concatenated
      and fed into a 1x1 convolutional block.
      """

    def __init__(self, ch_in, ch_out):
        super(MultiScaleConv, self).__init__()
        self.conv3x3x3 = conv_block(ch_in=ch_in, ch_out=ch_out, kernel_size=3, padding=1)
        self.conv5x5x5 = conv_block(ch_in=ch_in, ch_out=ch_out, kernel_size=5, padding=2)
        self.conv7x7x7 = conv_block(ch_in=ch_in, ch_out=ch_out, kernel_size=7, padding=3)
        self.conv1x1x1 = conv_block(ch_in=ch_out * 3, ch_out=ch_out, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.conv3x3x3(x)
        x2 = self.conv5x5x5(x)
        x3 = self.conv7x7x7(x)
        comb = torch.cat((x1, x2, x3), 1)
        out = self.conv1x1x1(comb)
        return out
