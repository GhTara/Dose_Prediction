import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep

from DosePrediction.Train.blocks_MDUNet import conv_3_1, DualDilatedBlock
# from DosePrediction.Train.model_monai import *


class MultiAttGate(nn.Module):
    def __init__(self, in_ch):
        super(MultiAttGate, self).__init__()

        self.initial_conv = nn.Conv3d(in_ch, in_ch,
                                      kernel_size=1, stride=1, padding='same')
        self.intermediate = nn.Sequential(
            nn.ReLU(inplace=True),
            conv_3_1(ch_in=in_ch, ch_out=in_ch),
            nn.BatchNorm3d(in_ch),
            nn.Sigmoid()
        )

    def forward(self, down_inp, sample_inp):
        # z1
        down_inp = self.initial_conv(down_inp)
        # z2
        sample_inp = self.initial_conv(sample_inp)
        # z12
        inp = torch.add(down_inp, sample_inp)
        # x12
        inp = self.intermediate(inp)
        out = torch.mul(down_inp, inp)
        return out


class AttGate(nn.Module):
    def __init__(self, in_ch):
        super(AttGate, self).__init__()

        self.initial_conv = nn.Conv3d(in_ch, in_ch,
                                      kernel_size=1, stride=1, padding='same')
        self.intermediate = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, in_ch, 1, stride=1, padding='same'),
            nn.BatchNorm3d(in_ch),
            nn.Sigmoid()
        )

    def forward(self, down_inp, sample_inp):
        # z1
        down_inp = self.initial_conv(down_inp)
        # z2
        sample_inp = self.initial_conv(sample_inp)
        # z12
        inp = torch.add(down_inp, sample_inp)
        # x12
        inp = self.intermediate(inp)
        out = torch.mul(down_inp, inp)
        return out


##############################
#        Generator
##############################

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


class MultiSingleConv(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
    ):
        super(MultiSingleConv, self).__init__()

        self.cov_ = conv_3_1(ch_in=in_channels, ch_out=out_channels)

    def forward(self, inp):
        out = self.cov_(inp)
        return out


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=True),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch, list_ch):
        super(Encoder, self).__init__()
        self.encoder_1 = nn.Sequential(
            SingleConv(in_ch, list_ch[1], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1)
        )
        self.encoder_2 = nn.Sequential(
            SingleConv(list_ch[1], list_ch[2], kernel_size=3, stride=2, padding=1),
            SingleConv(list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1)
        )
        self.encoder_3 = nn.Sequential(
            SingleConv(list_ch[2], list_ch[3], kernel_size=3, stride=2, padding=1),
            SingleConv(list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1)
        )
        self.encoder_4 = nn.Sequential(
            SingleConv(list_ch[3], list_ch[4], kernel_size=3, stride=2, padding=1),
            SingleConv(list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1)
        )
        self.encoder_5 = nn.Sequential(
            SingleConv(list_ch[4], list_ch[5], kernel_size=3, stride=2, padding=1),
            SingleConv(list_ch[5], list_ch[5], kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        out_encoder_1 = self.encoder_1(x)
        out_encoder_2 = self.encoder_2(out_encoder_1)
        out_encoder_3 = self.encoder_3(out_encoder_2)
        out_encoder_4 = self.encoder_4(out_encoder_3)
        out_encoder_5 = self.encoder_5(out_encoder_4)

        return [out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5]


class DilatedEncoder(nn.Module):
    def __init__(self, in_ch, list_ch):
        super(DilatedEncoder, self).__init__()
        self.encoder_1 = DualDilatedBlock(ch_in=in_ch, ch_out=list_ch[1])

        self.encoder_2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DualDilatedBlock(ch_in=list_ch[1], ch_out=list_ch[2])
        )

        self.encoder_3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DualDilatedBlock(ch_in=list_ch[2], ch_out=list_ch[3])
        )

        self.encoder_4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DualDilatedBlock(ch_in=list_ch[3], ch_out=list_ch[4])
        )

        self.encoder_5 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DualDilatedBlock(ch_in=list_ch[4], ch_out=list_ch[5])
        )

    def forward(self, x):
        out_encoder_1 = self.encoder_1(x)
        out_encoder_2 = self.encoder_2(out_encoder_1)
        out_encoder_3 = self.encoder_3(out_encoder_2)
        out_encoder_4 = self.encoder_4(out_encoder_3)
        out_encoder_5 = self.encoder_5(out_encoder_4)

        return [out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5]


class Decoder(nn.Module):
    def __init__(self, list_ch):
        super(Decoder, self).__init__()

        self.upconv_4 = UpConv(list_ch[5], list_ch[4])
        self.decoder_conv_4 = nn.Sequential(
            SingleConv(2 * list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1)
        )
        self.upconv_3 = UpConv(list_ch[4], list_ch[3])
        self.decoder_conv_3 = nn.Sequential(
            SingleConv(2 * list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1)
        )
        self.upconv_2 = UpConv(list_ch[3], list_ch[2])
        self.decoder_conv_2 = nn.Sequential(
            SingleConv(2 * list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1)
        )
        self.upconv_1 = UpConv(list_ch[2], list_ch[1])
        self.decoder_conv_1 = nn.Sequential(
            SingleConv(2 * list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1)
        )

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

        return out_decoder_1


class AttDecoder(nn.Module):
    def __init__(self, list_ch):
        super(AttDecoder, self).__init__()

        self.upconv_4 = UpConv(list_ch[5], list_ch[4])
        self.att_gate4 = AttGate(list_ch[4])
        self.decoder_conv_4 = nn.Sequential(
            # MultiSingleConv(2 * list_ch[4], list_ch[4]),
            # MultiSingleConv(list_ch[4], list_ch[4])
            SingleConv(2 * list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1)
        )
        self.upconv_3 = UpConv(list_ch[4], list_ch[3])
        self.att_gate3 = AttGate(list_ch[3])
        self.decoder_conv_3 = nn.Sequential(
            # MultiSingleConv(2 * list_ch[3], list_ch[3]),
            # MultiSingleConv(list_ch[3], list_ch[3])
            SingleConv(2 * list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1)
        )
        self.upconv_2 = UpConv(list_ch[3], list_ch[2])
        self.att_gate2 = AttGate(list_ch[2])
        self.decoder_conv_2 = nn.Sequential(
            # MultiSingleConv(2 * list_ch[2], list_ch[2]),
            # MultiSingleConv(list_ch[2], list_ch[2])
            SingleConv(2 * list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1)
        )
        self.upconv_1 = UpConv(list_ch[2], list_ch[1])
        self.att_gate1 = AttGate(list_ch[1])
        self.decoder_conv_1 = nn.Sequential(
            SingleConv(2 * list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1)
        )

    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5 = out_encoder

        in_up4 = self.upconv_4(out_encoder_5)
        in_att4 = self.att_gate4(down_inp=out_encoder_4, sample_inp=in_up4)
        out_decoder_4 = self.decoder_conv_4(
            torch.cat((in_up4, in_att4), dim=1)
        )
        in_up3 = self.upconv_3(out_decoder_4)
        in_att3 = self.att_gate3(down_inp=out_encoder_3, sample_inp=in_up3)
        out_decoder_3 = self.decoder_conv_3(
            torch.cat((in_up3, in_att3), dim=1)
        )
        in_up2 = self.upconv_2(out_decoder_3)
        in_att2 = self.att_gate2(down_inp=out_encoder_2, sample_inp=in_up2)
        out_decoder_2 = self.decoder_conv_2(
            torch.cat((in_up2, in_att2), dim=1)
        )
        in_up1 = self.upconv_1(out_decoder_2)
        in_att1 = self.att_gate1(down_inp=out_encoder_1, sample_inp=in_up1)
        out_decoder_1 = self.decoder_conv_1(
            torch.cat((in_up1, in_att1), dim=1)
        )

        return out_decoder_1


class PureAttDecoder(nn.Module):
    def __init__(self, list_ch):
        super(PureAttDecoder, self).__init__()

        self.upconv_4 = UpConv(list_ch[5], list_ch[4])
        self.att_gate4 = AttGate(list_ch[4])
        self.decoder_conv_4 = SingleConv(2 * list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1)

        self.upconv_3 = UpConv(list_ch[4], list_ch[3])
        self.att_gate3 = AttGate(list_ch[3])
        self.decoder_conv_3 = SingleConv(2 * list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1)

        self.upconv_2 = UpConv(list_ch[3], list_ch[2])
        self.att_gate2 = AttGate(list_ch[2])
        self.decoder_conv_2 = SingleConv(2 * list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1)

        self.upconv_1 = UpConv(list_ch[2], list_ch[1])
        self.att_gate1 = AttGate(list_ch[1])
        self.decoder_conv_1 = SingleConv(2 * list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1)

    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5 = out_encoder

        in_up4 = self.upconv_4(out_encoder_5)
        in_att4 = self.att_gate4(down_inp=out_encoder_4, sample_inp=in_up4)
        out_decoder_4 = self.decoder_conv_4(
            torch.cat((in_up4, in_att4), dim=1)
        )

        in_up3 = self.upconv_3(out_decoder_4)
        in_att3 = self.att_gate3(down_inp=out_encoder_3, sample_inp=in_up3)
        out_decoder_3 = self.decoder_conv_3(
            torch.cat((in_up3, in_att3), dim=1)
        )

        in_up2 = self.upconv_2(out_decoder_3)
        in_att2 = self.att_gate2(down_inp=out_encoder_2, sample_inp=in_up2)
        out_decoder_2 = self.decoder_conv_2(
            torch.cat((in_up2, in_att2), dim=1)
        )

        in_up1 = self.upconv_1(out_decoder_2)
        in_att1 = self.att_gate1(down_inp=out_encoder_1, sample_inp=in_up1)
        out_decoder_1 = self.decoder_conv_1(
            torch.cat((in_up1, in_att1), dim=1)
        )

        return out_decoder_1


class PureMultiAttDecoder(nn.Module):
    def __init__(self, list_ch):
        super(PureMultiAttDecoder, self).__init__()

        self.upconv_4 = UpConv(list_ch[5], list_ch[4])
        self.att_gate4 = MultiAttGate(list_ch[4])
        self.decoder_conv_4 = nn.Sequential(
            # MultiSingleConv(2 * list_ch[4], list_ch[4]),
            # MultiSingleConv(list_ch[4], list_ch[4])
            SingleConv(2 * list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1),
            # SingleConv(list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1)
        )
        self.upconv_3 = UpConv(list_ch[4], list_ch[3])
        self.att_gate3 = MultiAttGate(list_ch[3])
        self.decoder_conv_3 = nn.Sequential(
            # MultiSingleConv(2 * list_ch[3], list_ch[3]),
            # MultiSingleConv(list_ch[3], list_ch[3])
            SingleConv(2 * list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1),
            # SingleConv(list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1)
        )
        self.upconv_2 = UpConv(list_ch[3], list_ch[2])
        self.att_gate2 = MultiAttGate(list_ch[2])
        self.decoder_conv_2 = nn.Sequential(
            # MultiSingleConv(2 * list_ch[2], list_ch[2]),
            # MultiSingleConv(list_ch[2], list_ch[2])
            SingleConv(2 * list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1),
            # SingleConv(list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1)
        )
        self.upconv_1 = UpConv(list_ch[2], list_ch[1])
        self.att_gate1 = MultiAttGate(list_ch[1])
        self.decoder_conv_1 = nn.Sequential(
            SingleConv(2 * list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1)
        )

    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5 = out_encoder

        in_up4 = self.upconv_4(out_encoder_5)
        in_att4 = self.att_gate4(down_inp=out_encoder_4, sample_inp=in_up4)
        out_decoder_4 = self.decoder_conv_4(
            torch.cat((in_up4, in_att4), dim=1)
        )

        in_up3 = self.upconv_3(out_decoder_4)
        in_att3 = self.att_gate3(down_inp=out_encoder_3, sample_inp=in_up3)
        out_decoder_3 = self.decoder_conv_3(
            torch.cat((in_up3, in_att3), dim=1)
        )

        in_up2 = self.upconv_2(out_decoder_3)
        in_att2 = self.att_gate2(down_inp=out_encoder_2, sample_inp=in_up2)
        out_decoder_2 = self.decoder_conv_2(
            torch.cat((in_up2, in_att2), dim=1)
        )

        in_up1 = self.upconv_1(out_decoder_2)
        in_att1 = self.att_gate1(down_inp=out_encoder_1, sample_inp=in_up1)
        out_decoder_1 = self.decoder_conv_1(
            torch.cat((in_up1, in_att1), dim=1)
        )

        return out_decoder_1


class BaseUNet(nn.Module):
    def __init__(self, in_ch, list_ch, mode_decoder, mode_encoder):
        super(BaseUNet, self).__init__()

        # ----- Encoder part ----- #
        if mode_encoder == 1:
            self.encoder = Encoder(in_ch, list_ch)
        elif mode_encoder == 2:
            self.encoder = DilatedEncoder(in_ch, list_ch)

        # ----- Decoder part ----- #
        if mode_decoder == 1:
            self.decoder = Decoder(list_ch)
        # elif mode_decoder == 2:
        # self.decoder = MultiDecoder(list_ch)
        elif mode_decoder == 3:
            self.decoder = AttDecoder(list_ch)
        elif mode_decoder == 4:
            self.decoder = PureAttDecoder(list_ch)
        elif mode_decoder == 5:
            self.decoder = PureMultiAttDecoder(list_ch)

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
    def __init__(self, in_ch, out_ch,
                 list_ch_A, list_ch_B,
                 mode_decoder_A=1, mode_decoder_B=1,
                 mode_encoder_A=1, mode_encoder_B=1):
        super(Model, self).__init__()

        # list_ch records the number of channels in each stage, eg. [-1, 32, 64, 128, 256, 512]
        self.net_A = BaseUNet(in_ch, list_ch_A, mode_decoder_A, mode_encoder=mode_encoder_A)
        self.net_B = BaseUNet(in_ch + list_ch_A[1], list_ch_B, mode_decoder_B, mode_encoder=mode_encoder_B)

        self.conv_out_A = nn.Conv3d(list_ch_A[1], out_ch, kernel_size=1, padding=0, bias=True)
        self.conv_out_B = nn.Conv3d(list_ch_B[1], out_ch, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out_net_A = self.net_A(x)
        out_net_B = self.net_B(torch.cat((out_net_A, x), dim=1))

        output_A = self.conv_out_A(out_net_A)
        output_B = self.conv_out_B(out_net_B)
        return [output_A, output_B]



class ModelMonai(nn.Module):
    def __init__(self, in_ch, out_ch,
                 list_ch_A, list_ch_B,
                 mode_decoder_A=1, mode_decoder_B=1,
                 mode_encoder_A=1, mode_encoder_B=1):
        super(ModelMonai, self).__init__()

        # list_ch records the number of channels in each stage, eg. [-1, 32, 64, 128, 256, 512]
        self.net_A = UNet(
            spatial_dims=3,
            in_channels=in_ch,
            out_channels=list_ch_A[1],
            channels=list_ch_A,
            strides=(2, 2, 2, 2),
            # num_res_units=0
        )
        self.net_B = UNet(
            spatial_dims=3,
            in_channels=in_ch + list_ch_A[1],
            out_channels=1,
            channels=list_ch_B,
            strides=(2, 2, 2, 2),
            # num_res_units=0
        )

        self.conv_out_A = nn.Conv3d(list_ch_A[1], out_ch, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out_net_A = self.net_A(x)
        output_B = self.net_B(torch.cat((out_net_A, x), dim=1))

        output_A = self.conv_out_A(out_net_A)
        return [output_A, output_B]


##############################
#        Discriminator
##############################


class Discriminator1(nn.Module):
    def __init__(self, in_ch_a, in_ch_b):
        super(Discriminator1, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        in_ch_all = in_ch_a + in_ch_b
        self.model = nn.Sequential(
            *discriminator_block(in_ch_all, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            # nn.ZeroPad3d((1, 0, 1, 0)),
        )
        self.final = nn.Conv3d(512, 1, 4, padding=1, bias=False)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        intermediate = self.model(img_input)
        pad = nn.functional.pad(intermediate, pad=(1, 0, 1, 0, 1, 0))
        return self.final(pad)


# without sigmoid and attention gate
class Discriminator1_1(nn.Module):
    def __init__(self, in_ch_a, in_ch_b):
        super(Discriminator1_1, self).__init__()

        def sampling_layer(in_filters, out_filters, normalization=True, down=True):
            """Returns downsampling/sampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters,
                                kernel_size=4 if down else 3,
                                stride=2 if down else 1, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def discriminator_block(in_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = nn.Sequential(
                *sampling_layer(in_filters, in_filters, down=True),
                *sampling_layer(in_filters, 2 * in_filters, down=False)
            )
            return layers

        in_ch_all = in_ch_a + in_ch_b
        self.initial = nn.Sequential(
            *sampling_layer(in_ch_all, 64, down=False),
        )
        self.model = nn.Sequential(
            *discriminator_block(64),
            *discriminator_block(128),
            *discriminator_block(256),
            # *discriminator_block(512),
            # nn.ZeroPad3d((1, 0, 1, 0)),
        )
        self.final = nn.Conv3d(512, 1, 4, padding=1, bias=False)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        initial = self.initial(img_input)
        intermediate = self.model(initial)
        pad = nn.functional.pad(intermediate, pad=(1, 0, 1, 0, 1, 0))
        return self.final(pad)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_ch):
        super(DiscriminatorBlock, self).__init__()

        def sampling_layer(in_filters, out_filters, normalization=True, down=True):
            """Returns downsampling/sampling layers of each discriminator block"""
            layers = nn.Sequential(nn.Conv3d(in_filters, out_filters,
                                             kernel_size=4 if down else 3,
                                             stride=2 if down else 1, padding=1),
                                   nn.BatchNorm3d(out_filters) if normalization else None,
                                   nn.LeakyReLU(0.2, inplace=True)
                                   )
            return layers

        self.down_conv = sampling_layer(in_ch, in_ch, normalization=True, down=True)
        self.sample_conv = sampling_layer(in_ch, in_ch, normalization=True, down=False)
        self.att_gate = AttGate(in_ch)

    def forward(self, x):
        down_inp = self.down_conv(x)
        sample_inp = self.sample_conv(down_inp)
        out_gate = self.att_gate(down_inp=down_inp, sample_inp=sample_inp)
        out = torch.cat((sample_inp, out_gate), 1)
        return out


# without sigmoid and with attention gate
class Discriminator1_2(nn.Module):
    def __init__(self, in_ch_a, in_ch_b):
        super(Discriminator1_2, self).__init__()

        n_filter = 64
        in_ch_all = in_ch_a + in_ch_b
        self.initial = nn.Sequential(nn.Conv3d(in_ch_all, n_filter,
                                               kernel_size=3,
                                               stride=1, padding=1),
                                     nn.BatchNorm3d(n_filter),
                                     nn.LeakyReLU(0.2, inplace=True)
                                     )

        self.model = nn.Sequential(
            DiscriminatorBlock(n_filter),
            DiscriminatorBlock(2*n_filter),
            DiscriminatorBlock(4*n_filter),
            DiscriminatorBlock(8*n_filter),
            # nn.ZeroPad3d((1, 0, 1, 0)),
        )
        self.final = nn.Conv3d(16*n_filter, 1, 4, padding=1, bias=False)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        initial = self.initial(img_input)
        intermediate = self.model(initial)
        pad = nn.functional.pad(intermediate, pad=(1, 0, 1, 0, 1, 0))
        return self.final(pad)


class Discriminator2(nn.Module):
    def __init__(self, in_ch_a, in_ch_b):
        super(Discriminator2, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        in_ch_all = in_ch_a + in_ch_b
        self.model = nn.Sequential(
            *discriminator_block(in_ch_all, 32, normalization=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            # nn.ZeroPad3d((1, 0, 1, 0)),
        )
        self.final = nn.Sequential(
            nn.Conv3d(256, 1, 4, padding=1, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        intermediate = self.model(img_input)
        # pad = nn.functional.pad(intermediate, pad=(1, 0, 1, 0, 1, 0))
        return self.final(intermediate)
        
        
class Discriminator3(nn.Module):
    def __init__(
            self,
            in_channels,
            img_size,
            # 768
            hidden_size: int = 384,
            mlp_dim: int = 384*4,
            num_layers: int = 12,
            num_heads: int = 12,
            pos_embed: str = "conv",
            classification: bool = False,
            num_classes: int = 2,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
            post_activation="Tanh",
            qkv_bias: bool = False,
    ):
        super().__init__()
        spatial_dims = 3
        patch_size = ensure_tuple_rep(16, spatial_dims)
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            # 384 32x32
            hidden_size=768,
            mlp_dim=3072,
            num_layers=4,
            # 12 med
            num_heads=12,
            pos_embed="perceptron",
            classification=True,
            dropout_rate=0.0,
            spatial_dims=spatial_dims,
            num_classes=2,
            # post_activation=None
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        out, _ = self.vit(img_input)
        return out

