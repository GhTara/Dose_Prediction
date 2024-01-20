import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(SingleConv, self).__init__()

        self.single_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=True),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)


class DenseConvolve(nn.Module):
    def __init__(self, in_ch, growth_rate=16, stride=(1, 1, 1)):
        super(DenseConvolve, self).__init__()

        self.single_conv = nn.Sequential(
            nn.Conv3d(in_ch, growth_rate, kernel_size=(3, 3, 3), padding=1, stride=stride, bias=True),
            nn.InstanceNorm3d(growth_rate, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat((self.single_conv(x), x), dim=1)


class DenseDownsample(nn.Module):
    def __init__(self, in_ch, growth_rate=16, stride=(2, 2, 2)):
        super(DenseDownsample, self).__init__()

        self.single_conv = nn.Sequential(
            nn.Conv3d(in_ch, growth_rate, kernel_size=(3, 3, 3), padding=1, stride=stride, bias=True),
            nn.InstanceNorm3d(growth_rate, affine=True),
            nn.ReLU(inplace=True)
        )

        self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):
        return torch.cat((self.single_conv(x), self.pooling(x)), dim=1)


class UNetUpsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNetUpsample, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(3, 3, 3), padding=1, stride=(1, 1, 1), bias=True),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch, growth_rate=16):
        super(Encoder, self).__init__()
        self.encoder_1 = nn.Sequential(
            DenseConvolve(in_ch, growth_rate),
            DenseConvolve(in_ch + growth_rate, growth_rate),
        )
        self.encoder_2 = nn.Sequential(
            DenseDownsample(in_ch + 2 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 3 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 4 * growth_rate, growth_rate)
        )
        self.encoder_3 = nn.Sequential(
            DenseDownsample(in_ch + 5 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 6 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 7 * growth_rate, growth_rate)
        )
        self.encoder_4 = nn.Sequential(
            DenseDownsample(in_ch + 8 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 9 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 10 * growth_rate, growth_rate)
        )
        self.encoder_5 = nn.Sequential(
            DenseDownsample(in_ch + 11 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 12 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 13 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 14 * growth_rate, growth_rate),
            DenseConvolve(in_ch + 15 * growth_rate, growth_rate)
        )

    def forward(self, x):
        out_encoder_1 = self.encoder_1(x)
        out_encoder_2 = self.encoder_2(out_encoder_1)
        out_encoder_3 = self.encoder_3(out_encoder_2)
        out_encoder_4 = self.encoder_4(out_encoder_3)
        out_encoder_5 = self.encoder_5(out_encoder_4)

        return [out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5]


class Decoder(nn.Module):
    def __init__(self, in_ch, growth_rate, upsample_chan):
        super(Decoder, self).__init__()

        self.upconv_4 = UNetUpsample(in_ch + 16 * growth_rate, upsample_chan)
        self.decoder_conv_4 = nn.Sequential(
            SingleConv(in_ch + 11 * growth_rate + upsample_chan, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                       padding=1),
            SingleConv(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        )
        self.upconv_3 = UNetUpsample(256, upsample_chan)
        self.decoder_conv_3 = nn.Sequential(
            SingleConv(in_ch + 8 * growth_rate + upsample_chan, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                       padding=1),
            SingleConv(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        )
        self.upconv_2 = UNetUpsample(128, upsample_chan)
        self.decoder_conv_2 = nn.Sequential(
            SingleConv(in_ch + 5 * growth_rate + upsample_chan, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            SingleConv(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        )
        self.upconv_1 = UNetUpsample(64, upsample_chan)
        self.decoder_conv_1 = nn.Sequential(
            SingleConv(in_ch + 2 * growth_rate + upsample_chan, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            SingleConv(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        )

        self.final_conv = nn.Conv3d(32, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=True)

    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5 = out_encoder

        out_decoder_4 = self.decoder_conv_4(
            torch.cat((self.upconv_4(out_encoder_5), out_encoder_4), dim=1)
        )
        out_decoder_3 = self.decoder_conv_3(
            torch.cat((self.upconv_3(out_decoder_4), out_encoder_3), dim=1)
        )
        out_decoder_2 = self.decoder_conv_2(
            torch.cat((self.upconv_2(out_decoder_3), out_encoder_2), dim=1)
        )
        out_decoder_1 = self.decoder_conv_1(
            torch.cat((self.upconv_1(out_decoder_2), out_encoder_1), dim=1)
        )

        final_output = self.final_conv(out_decoder_1)
        return [final_output]


class HD_UNet(nn.Module):
    def __init__(self, in_ch, growth_rate, upsample_chan):
        super(HD_UNet, self).__init__()
        self.encoder = Encoder(in_ch, growth_rate)
        self.decoder = Decoder(in_ch, growth_rate, upsample_chan)

        # init
        self.initialize()

    @staticmethod
    def init_conv_IN(modules):
        for m in modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def initialize(self):
        print('# random init encoder weight using nn.init.kaiming_uniform !')
        self.init_conv_IN(self.decoder.modules)
        print('# random init decoder weight using nn.init.kaiming_uniform !')
        self.init_conv_IN(self.encoder.modules)

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)

        # Output is a list: [Output]
        return out_decoder


class Model(nn.Module):
    def __init__(self, in_ch, growth_rate, upsample_chan):
        super(Model, self).__init__()

        self.model = HD_UNet(in_ch, growth_rate, upsample_chan)

    def forward(self, x):
        return self.model(x)

