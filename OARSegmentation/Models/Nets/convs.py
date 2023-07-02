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
      Multi scale convolutional block with 3 convolutional blocks
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


# class MultiScaleConvTRANS(nn.Module):
#
#     def __init__(
#             self,
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride,
#             padding,
#             output_padding,
#             groups,
#             bias,
#             dilation,
#     ):
#         super(MultiScaleConvTRANS, self).__init__()
#         self.conv3x3x3 = conv_trans_block(ch_in=in_channels, ch_out=out_channels, kernel_size=3, padding=1)
#         self.conv5x5x5 = conv_trans_block(ch_in=in_channels, ch_out=out_channels, kernel_size=5, padding=2)
#         self.conv7x7x7 = conv_trans_block(ch_in=in_channels, ch_out=out_channels, kernel_size=7, padding=3)
#         self.conv1x1x1 = conv_trans_block(ch_in=out_channels * 3, ch_out=out_channels, kernel_size=1, padding=0)
#
#     def forward(self, x):
#         x1 = self.conv3x3x3(x)
#         x2 = self.conv5x5x5(x)
#         x3 = self.conv7x7x7(x)
#         comb = torch.cat((x1, x2, x3), 1)
#         out = self.conv1x1x1(comb)
#         return out

def test():
    # feature_size: int = 16
    # hidden_size: int = 768
    # norm_name: Union[Tuple, str] = "instance"
    # res_block: bool = True
    # spatial_dims: int = 3
    # out_channels = 3
    # model = ModifiedUnetrUpBlock(
    #     spatial_dims=spatial_dims,
    #     in_channels=2,
    #     out_channels=3,
    #     kernel_size=3,
    #     upsample_kernel_size=2,
    #     norm_name=norm_name,
    #     res_block=res_block, )
    model = MultiScaleConv(3, 1, 3, norm_name='INSTANCE', act_name='prelu', res_flag=True)
    vol1 = torch.randn((1, 1, 64, 64, 64))
    vol2 = torch.randn((1, 3, 128, 128, 128))
    # out.shape : (1, 3, 128, 128, 128)

    # pred = model(vol1, vol2)
    pred = model(vol1)
    print(pred.shape)


if __name__ == '__main__':
    test()
