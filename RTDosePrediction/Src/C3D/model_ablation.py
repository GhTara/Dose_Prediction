from typing import Optional, Sequence, Union, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.dynunet_block import get_conv_layer, UnetOutBlock

from monai.networks.nets.vit import ViT
from monai.networks.nets import resnet34, ResNet, resnet10
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers.convutils import calculate_out_shape, same_padding
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import Reshape
from monai.utils import ensure_tuple, ensure_tuple_rep

from UNet.models.nets.base_blocks_ablation import ModifiedUnetrUpBlock
from RTDosePrediction.Src.C3D.baseline_model import BaseUNet



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


##############################
#        Encoder
##############################
class ViTSharedEncoder(nn.Module):

    def __init__(
            self,
            in_channels: int,
            img_size: Union[Sequence[int], int],
            feature_size: int = 16,
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_heads: int = 12,
            num_layers: int = 12,
            pos_embed: str = "conv",
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = True,
            res_block: bool = True,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
    ) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = num_layers
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        self.skip1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.skip2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.skip3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.skip4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )

        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        ##############################
        # model a
        ##############################
        i = self.num_layers // 4
        z12, hidden_states_out = self.vit(x_in)
        # 16 x 128 x 128 x 128
        out_encoder_1 = self.skip1(x_in)
        z3 = hidden_states_out[i]
        # 32 x 64 x 64 x 64
        out_encoder_2 = self.skip2(self.proj_feat(z3))
        z6 = hidden_states_out[i * 2]
        # 64 x 32 x 32 x 32
        out_encoder_3 = self.skip3(self.proj_feat(z6))
        z9 = hidden_states_out[i * 3]
        # 128 x 16 x 16 x 16
        out_encoder_4 = self.skip4(self.proj_feat(z9))
        # 786 x 8 x 8 x 8
        out_encoder_5 = self.proj_feat(z12)

        return [out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5]


class SharedEncoder(nn.Module):
    def __init__(self, in_ch, list_ch):
        super(SharedEncoder, self).__init__()
        self.encoder_1 = nn.Sequential(
            SingleConv(in_ch, list_ch[1], kernel_size=3, stride=1, padding=1),
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

    def forward(self, x):
        out_encoder_1 = self.encoder_1(x)
        out_encoder_2 = self.encoder_2(out_encoder_1)
        out_encoder_3 = self.encoder_3(out_encoder_2)
        out_encoder_4 = self.encoder_4(out_encoder_3)

        return [out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4]


class DilatedSharedEncoder(nn.Module):
    def __init__(self, in_ch, list_ch):
        super(DilatedSharedEncoder, self).__init__()
        self.encoder_1 = SingleConv(in_ch, list_ch[1], kernel_size=3, stride=1, padding=1)
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

    def forward(self, x):
        out_encoder_1 = self.encoder_1(x)
        out_encoder_2 = self.encoder_2(out_encoder_1)
        out_encoder_3 = self.encoder_3(out_encoder_2)
        out_encoder_4 = self.encoder_4(out_encoder_3)

        return [out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4]


class InitialEncoderA(nn.Module):
    def __init__(self, in_ch, mid_ch, down_ch, mode):
        super(InitialEncoderA, self).__init__()

        if mode == 1:
            self.conv = nn.Sequential(
                SingleConv(in_ch, mid_ch, kernel_size=3, stride=1, padding=1),
                SingleConv(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1),
            )

        elif mode == 2:
            self.conv = DualDilatedBlock(ch_in=in_ch, ch_out=mid_ch)

        self.down = SingleConv(mid_ch, down_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        conv_x = self.conv(x)
        down_x = self.down(conv_x)
        return conv_x, down_x


##############################
#        Decoder
##############################
class MonaiSharedDecoder(nn.Module):

    def __init__(
            self,
            feature_size: int = 16,
            hidden_size: int = 768,
            pos_embed: str = "conv",
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = True,
            res_block: bool = True,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
            mode_multi: bool = False,
            act='relu',
            multiS_conv=True,
    ) -> None:
        super().__init__()

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            upsample_kernel_size=2,
            kernel_size=3,
            norm_name=norm_name
        ) if not mode_multi else \
            ModifiedUnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size,
                out_channels=feature_size * 8,
                upsample_kernel_size=2,
                act=act,
                multiS_conv=multiS_conv,
            )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            upsample_kernel_size=2,
            kernel_size=3,
            norm_name=norm_name
        ) if not mode_multi else \
            ModifiedUnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 8,
                out_channels=feature_size * 4,
                upsample_kernel_size=2,
                act=act,
                multiS_conv=multiS_conv,
            )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            upsample_kernel_size=2,
            kernel_size=3,
            norm_name=norm_name
        ) if not mode_multi else \
            ModifiedUnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 4,
                out_channels=feature_size * 2,
                upsample_kernel_size=2,
                act=act,
                multiS_conv=multiS_conv,
            )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            upsample_kernel_size=2,
            kernel_size=3,
            norm_name=norm_name
        ) if not mode_multi else \
            ModifiedUnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 2,
                out_channels=feature_size,
                upsample_kernel_size=2,
                act=act,
                multiS_conv=multiS_conv,
            )

    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5 = out_encoder
        dec4 = self.decoder4(out_encoder_5, out_encoder_4)
        dec3 = self.decoder3(dec4, out_encoder_3)
        dec2 = self.decoder2(dec3, out_encoder_2)
        dec1 = self.decoder1(dec2, out_encoder_1)

        return [dec1, dec2, dec3, dec4]


class SharedDecoder(nn.Module):
    def __init__(self, list_ch):
        super(SharedDecoder, self).__init__()

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
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4 = out_encoder

        out_decoder_3 = self.decoder_conv_3(
            torch.cat((self.upconv_3(out_encoder_4), out_encoder_3), dim=1)
        )
        out_decoder_2 = self.decoder_conv_2(
            torch.cat((self.upconv_2(out_decoder_3), out_encoder_2), dim=1)
        )
        out_decoder_1 = self.decoder_conv_1(
            torch.cat((self.upconv_1(out_decoder_2), out_encoder_1), dim=1)
        )

        return out_decoder_1


class DilatedSharedDecoder(nn.Module):
    def __init__(self, list_ch):
        super(DilatedSharedDecoder, self).__init__()

        self.upconv_3 = UpConv(list_ch[4], list_ch[3])
        self.decoder_conv_3 = conv_3_1(ch_in=2 * list_ch[3], ch_out=list_ch[3])
        self.upconv_2 = UpConv(list_ch[3], list_ch[2])
        self.decoder_conv_2 = conv_3_1(ch_in=2 * list_ch[2], ch_out=list_ch[2])
        self.upconv_1 = UpConv(list_ch[2], list_ch[1])
        self.decoder_conv_1 = SingleConv(2 * list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1)

    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4 = out_encoder

        out_decoder_3 = self.decoder_conv_3(
            torch.cat((self.upconv_3(out_encoder_4), out_encoder_3), dim=1)
        )
        out_decoder_2 = self.decoder_conv_2(
            torch.cat((self.upconv_2(out_decoder_3), out_encoder_2), dim=1)
        )
        out_decoder_1 = self.decoder_conv_1(
            torch.cat((self.upconv_1(out_decoder_2), out_encoder_1), dim=1)
        )

        return out_decoder_1


##############################
#        Model
##############################
class VitGenerator(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 mode_decoder,
                 mode_encoder,
                 img_size,
                 feature_size: int = 16,
                 hidden_size: int = 768,
                 mlp_dim: int = 3072,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 pos_embed: str = "conv",
                 norm_name: Union[Tuple, str] = "instance",
                 conv_block: bool = True,
                 res_block: bool = True,
                 dropout_rate: float = 0.0,
                 spatial_dims: int = 3,
                 mode_multi_dec=False,
                 act='relu',
                 multiS_conv=True):
        super().__init__()

        # ----- Encoder part ----- #
        self.encoder = ViTSharedEncoder(
            in_channels=in_ch,
            img_size=img_size,
            # 16 => 4
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=res_block,
            conv_block=conv_block,
            dropout_rate=dropout_rate,
        )

        # ----- Decoder part ----- #
        if True:
            self.decoder = MonaiSharedDecoder(
                feature_size=feature_size,
                hidden_size=hidden_size,
                mode_multi=mode_multi_dec,
                act=act,
                multiS_conv=multiS_conv
            )

        def to_out(in_feature):
            return nn.Sequential(
                nn.Conv3d(in_feature, out_ch, kernel_size=1, padding=0, bias=True),
                # nn.Tanh(),
                # nn.Sigmoid()
            )

        self.dose_convertors = nn.ModuleList([to_out(feature_size)])
        # depth: 4
        for i in range(1, 4):
            self.dose_convertors.append(to_out(feature_size * np.power(2, i)))
        self.out = nn.Sequential(
            nn.Conv3d(feature_size, out_ch, kernel_size=1, padding=0, bias=True),
            # nn.Tanh(),
            # nn.Sigmoid()
        )

    def update_config(self, config_hparam):
        self.encoder.hidden_size = config_hparam["hidden_size"]
        self.encoder.num_layers = config_hparam["hidden_size"]

    def forward(self, x):
        out_encoder = self.encoder(x)

        out_decoders = self.decoder(out_encoder)
        outputs = []
        for out_dec, convertor in zip(out_decoders, self.dose_convertors):
            outputs.append(convertor(out_dec))

        return outputs



class SharedEncoderModel(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 mode_decoder,
                 mode_encoder,
                 img_size,
                 feature_size_a: int = 16,
                 feature_size_b: int = 32,
                 hidden_size: int = 768,
                 mlp_dim: int = 3072,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 pos_embed: str = "conv",
                 norm_name: Union[Tuple, str] = "instance",
                 conv_block: bool = True,
                 res_block: bool = True,
                 dropout_rate: float = 0.0,
                 spatial_dims: int = 3, ):
        super().__init__()

        # ----- Encoder part ----- #
        self.encoder = ViTSharedEncoder(
            in_channels=in_ch,
            img_size=img_size,
            # 16 => 4
            feature_size=feature_size_a,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=res_block,
            conv_block=conv_block,
            dropout_rate=dropout_rate,
        )

        # ----- Decoder part ----- #
        if True:
            self.decoder_a = MonaiSharedDecoder(
                feature_size=feature_size_a,
                hidden_size=hidden_size,
            )
            self.decoder_b = MonaiSharedDecoder(
                feature_size=feature_size_b,
                hidden_size=hidden_size,
            )
        # self.out_a = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size_a, out_channels=out_ch)
        # self.out_b = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size_b, out_channels=out_ch)
        self.out_a = nn.Sequential(
            nn.Conv3d(feature_size_a, out_ch, kernel_size=1, padding=0, bias=True),
            nn.Tanh())
        self.out_b = nn.Sequential(
            nn.Conv3d(feature_size_b, out_ch, kernel_size=1, padding=0, bias=True),
            nn.Tanh())

    def forward(self, x):
        out_encoder = self.encoder(x)

        out_decoder_a = self.decoder_a(out_encoder)
        out_a = self.out_a(out_decoder_a[0])

        out_encoder_b = []
        for enc, dec_a in zip(out_encoder[:-1], out_decoder_a):
            out_encoder_b.append(torch.cat((enc, dec_a), 1))
        out_encoder_b.append(out_encoder[-1])

        out_decoder_b = self.decoder_b(out_encoder_b)
        out_b = self.out_b(out_decoder_b[0])

        # Output is a list: [Output]
        return out_a, out_b


class SharedUNetModel(nn.Module):
    def __init__(self, in_ch, out_ch, list_ch, mode_decoder, mode_encoder):
        super(SharedUNetModel, self).__init__()
        # list_ch : 16, 32, 64, 128, 256, 512
        # ----- Encoder part ----- #
        if mode_encoder == 1:
            self.shared_encoder = SharedEncoder(list_ch[2], [-1] + list_ch[2:-1])
        elif mode_encoder == 2:
            self.shared_encoder = DilatedSharedEncoder(list_ch[2], [-1] + list_ch[2:-1])

        # ----- Decoder part ----- #
        if mode_decoder == 1:
            self.shared_decoder = SharedDecoder([-1] + list_ch[2:-1])
        if mode_decoder == 2:
            self.shared_decoder = DilatedSharedDecoder([-1] + list_ch[2:-1])

        self.initial_encoder_a = InitialEncoderA(in_ch, list_ch[1], list_ch[2], mode=mode_encoder)

        self.initial_encoder_b = SingleConv(in_ch + list_ch[1], list_ch[2], kernel_size=3, stride=1, padding=1)

        self.decoder_a = nn.Sequential(
            SingleConv(list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1),
            UpConv(list_ch[2], list_ch[1])
        )
        self.out_decoder_a = SingleConv(2 * list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1)

        if mode_encoder == 1:
            self.bottle_neck_b = nn.Sequential(
                SingleConv(list_ch[5], list_ch[6], kernel_size=3, stride=2, padding=1),
                SingleConv(list_ch[6], list_ch[6], kernel_size=3, stride=1, padding=1),
                UpConv(list_ch[6], list_ch[5])
            )
        elif mode_encoder == 2:
            self.bottle_neck_b = nn.Sequential(
                nn.MaxPool3d(kernel_size=2),
                DualDilatedBlock(ch_in=list_ch[5], ch_out=list_ch[6]),
                UpConv(list_ch[6], list_ch[5])
            )
        if True:
            self.out_bottle_neck_b = nn.Sequential(
                SingleConv(2 * list_ch[5], list_ch[5], kernel_size=3, stride=1, padding=1),
                SingleConv(list_ch[5], list_ch[5], kernel_size=3, stride=1, padding=1)
            )
        # elif mode_encoder == 2:
        #     self.out_bottle_neck_b = DualDilatedBlock(ch_in=2 * list_ch[5], ch_out=list_ch[5])

        self.conv_out_a = nn.Sequential(
            nn.Conv3d(list_ch[1], out_ch, kernel_size=1, padding=0, bias=True),
            nn.Tanh())
        self.conv_out_b = nn.Sequential(
            nn.Conv3d(list_ch[2], out_ch, kernel_size=1, padding=0, bias=True),
            nn.Tanh())

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
        self.init_conv_IN(self.shared_encoder.modules)
        self.init_conv_IN(self.initial_encoder_a.modules)
        self.init_conv_IN(self.initial_encoder_b.modules)
        self.init_conv_IN(self.bottle_neck_b.modules)
        self.init_conv_IN(self.out_bottle_neck_b.modules)

        print('# random init decoder weight using nn.init.kaiming_uniform !')
        self.init_conv_IN(self.shared_decoder.modules)
        self.init_conv_IN(self.decoder_a.modules)
        self.init_conv_IN(self.out_decoder_a.modules)

    def forward(self, in_model):
        conv, down = self.initial_encoder_a(in_model)
        out_enc_a = self.shared_encoder(down)
        out_dec_a = self.shared_decoder(out_enc_a)
        out_dec_a = self.decoder_a(out_dec_a)
        out_dec_a = self.out_decoder_a(torch.cat((conv, out_dec_a), dim=1))

        x = self.initial_encoder_b(torch.cat((out_dec_a, in_model), dim=1))
        out_enc_b = self.shared_encoder(x)
        out_bot_b = self.bottle_neck_b(out_enc_b[-1])
        out_enc_b[-1] = self.out_bottle_neck_b(torch.cat((out_enc_b[-1], out_bot_b), dim=1))
        out_dec_b = self.shared_decoder(out_enc_b)

        out_a = self.conv_out_a(out_dec_a)
        out_b = self.conv_out_b(out_dec_b)

        # Output is a list: [Output]
        return [out_a, out_b]


class SharedUNetRModel(nn.Module):

    def __init__(
            self,
            in_channels_a: int,
            in_channels_b: int,
            out_channels: int,
            img_size: Union[Sequence[int], int],
            feature_size: int = 16,
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_heads: int = 12,
            pos_embed: str = "conv",
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = True,
            res_block: bool = True,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
    ) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        self.initial_a = get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=in_channels_a,
            out_channels=in_channels_b,
            kernel_size=1,
            stride=1,
            bias=True,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.vit = ViT(
            in_channels=in_channels_b,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        self.skip1 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size,
            num_layer=3,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.skip2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.skip2B = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size * 2,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.skip3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.skip3B = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 4,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.skip4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.skip4B = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.skip5 = get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 16,
            kernel_size=1,
            stride=1,
            bias=True,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.skip5B = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 16,
            out_channels=feature_size * 16,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.skip6B = get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 32,
            kernel_size=1,
            stride=1,
            bias=True,
            act=None,
            norm=None,
            conv_only=False,
        )

        self.decoder_b = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 32,
            out_channels=feature_size * 16,
            upsample_kernel_size=2,
            kernel_size=3,
            norm_name=norm_name
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            upsample_kernel_size=2,
            kernel_size=3,
            norm_name=norm_name
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            upsample_kernel_size=2,
            kernel_size=3,
            norm_name=norm_name
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            upsample_kernel_size=2,
            kernel_size=3,
            norm_name=norm_name
        )
        self.decoder_a = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            upsample_kernel_size=2,
            kernel_size=3,
            norm_name=norm_name
        )

        self.out_a = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.out_b = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size * 2, out_channels=out_channels)

        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        ##############################
        # model a
        ##############################
        x = self.initial_a(x_in)
        _, hidden_states_out = self.vit(x)
        z2 = hidden_states_out[2]
        x2a = self.skip1(self.proj_feat(z2))
        z4 = hidden_states_out[4]
        x4a = self.skip2(self.proj_feat(z4))
        z6 = hidden_states_out[6]
        x6a = self.skip3(self.proj_feat(z6))
        z8 = hidden_states_out[8]
        x8a = self.skip4(self.proj_feat(z8))
        z10 = hidden_states_out[10]
        x10a = self.skip5(self.proj_feat(z10))

        dec8a = self.decoder3(x10a, x8a)
        dec6a = self.decoder2(dec8a, x6a)
        dec4a = self.decoder1(dec6a, x4a)
        dec2a = self.decoder_a(dec4a, x2a)
        outA = self.out_a(dec2a)

        ##############################
        # model b
        ##############################
        x = torch.cat((dec2a, x_in), dim=1)
        z12, hidden_states_out = self.vit(x)
        z4 = hidden_states_out[4]
        x4b = self.skip2B(self.skip2(self.proj_feat(z4)))
        z6 = hidden_states_out[6]
        x6b = self.skip3B(self.skip3(self.proj_feat(z6)))
        z8 = hidden_states_out[8]
        x8b = self.skip4B(self.skip4(self.proj_feat(z8)))
        z10 = hidden_states_out[10]
        x10b = self.skip5B(self.skip5(self.proj_feat(z10)))
        x12b = self.skip6B(self.proj_feat(z12))

        dec10b = self.decoder_b(x12b, x10b)
        dec8b = self.decoder3(dec10b, x8b)
        dec6b = self.decoder2(dec8b, x6b)
        dec4b = self.decoder1(dec6b, x4b)
        outB = self.out_b(dec4b)

        return outA, outB


class SharedUNetRModelA(nn.Module):

    def __init__(
            self,
            in_channels_a: int,
            in_channels_b: int,
            out_channels: int,
            img_size: Union[Sequence[int], int],
            feature_size: int = 16,
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_heads: int = 12,
            pos_embed: str = "conv",
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = True,
            res_block: bool = True,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
    ) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels_b,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        self.skip1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels_a,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.skip2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.skip3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.skip4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            upsample_kernel_size=2,
            kernel_size=3,
            norm_name=norm_name
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            upsample_kernel_size=2,
            kernel_size=3,
            norm_name=norm_name
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            upsample_kernel_size=2,
            kernel_size=3,
            norm_name=norm_name
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            upsample_kernel_size=2,
            kernel_size=3,
            norm_name=norm_name
        )

        self.out_a = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        ##############################
        # model a
        ##############################
        z12, hidden_states_out = self.vit(x_in)
        x0a = self.skip1(x_in)
        z3 = hidden_states_out[3]
        x3a = self.skip2(self.proj_feat(z3))
        z6 = hidden_states_out[6]
        x6a = self.skip3(self.proj_feat(z6))
        z9 = hidden_states_out[9]
        x9a = self.skip4(self.proj_feat(z9))
        x12a = self.proj_feat(z12)

        dec4a = self.decoder4(x12a, x9a)
        dec3a = self.decoder3(dec4a, x6a)
        dec2a = self.decoder2(dec3a, x3a)
        dec1a = self.decoder1(dec2a, x0a)
        outA = self.out_a(dec1a)

        return outA, outA


def test():
    # generator = SharedUNetModel(in_ch=9, out_ch=1,
    #                             list_ch=[-1, 16, 32, 64, 128, 256, 512],
    #                             mode_decoder=2,
    #                             mode_encoder=2)
    # generator = SharedUNetRModelA(
    #     in_channels_a=9,
    #     in_channels_b=9,
    #     out_channels=1,
    #     img_size=(128, 128, 128),
    #     # 16 => 4
    #     feature_size=16,
    #     hidden_size=768,
    #     mlp_dim=3072,
    #     num_heads=12,
    #     pos_embed="perceptron",
    #     norm_name="instance",
    #     res_block=True,
    #     conv_block=True,
    #     dropout_rate=0.0,
    # )
    # encoder = ViTSharedEncoder(
    #     in_channels=9,
    #     img_size=(128, 128, 128),
    #     # 16 => 4
    #     feature_size=16,
    #     hidden_size=768,
    #     mlp_dim=3072,
    #     num_heads=12,
    #     num_layers=12,
    #     pos_embed="perceptron",
    #     norm_name="instance",
    #     res_block=True,
    #     conv_block=True,
    #     dropout_rate=0.0,
    # )
    # decoder = MonaiSharedDecoder(
    #     feature_size=16,
    #     hidden_size=768,
    # )

    # model = SharedEncoderModel(
    #     in_ch=9,
    #     out_ch=1,
    #     mode_decoder=1,
    #     mode_encoder=1,
    #     img_size=(128, 128, 128),
    #     num_layers=4,  # 4, 8, 12
    #     num_heads=6  # 3, 6, 12
    # )

    # generator = VitGenerator(
    #     in_ch=9,
    #     out_ch=1,
    #     mode_decoder=1,
    #     mode_encoder=1,
    #     feature_size=16,
    #     img_size=(128, 128, 128),
    #     num_layers=8,  # 4, 8, 12
    #     num_heads=6,  # 3, 6, 12
    #     act='mish',
    #     mode_multi_dec=False
    # )

    generator = create_pretrained_unet(
        in_ch=9, out_ch=1,
        list_ch_A=[-1, 16, 32, 64, 128, 256],
        list_ch_B=[-1, 32, 64, 128, 256, 512],
        ckpt_file='NetworkTrainer/C3D_bs4_iter80000.pkl',
        mode_decoder=1,
        mode_encoder=1,
        feature_size=16,
        img_size=(128, 128, 128),
        num_layers=8,  # 4, 8, 12
        num_heads=6,  # 3, 6, 12
        act='mish',
        mode_multi_dec=False,
    )

    # regressor = ResnetRegressor()
    # net, pretrained_params = create_pretrained_medical_resnet(
    #     'D:/python_code/thesis_final/dose_prediction/resnet_34_23dataset.pth', model_constructor=resnet34)
    # pretrained_params = set(pretrained_params) if pretrained_params else set()
    # for n, param in net.named_parameters():
    #     print(param.requires_grad, bool(n not in pretrained_params))
    #         # param.requires_grad = bool(n not in self.pretrained_params)
    inp = torch.randn((1, 9, 128, 128, 128))
    out = generator(inp)[1]
    # for o in out:
    print(out.shape)

    # inp = torch.randn((1, 9, 128, 128, 128))
    # out_gen = generator(inp)
    # print(out_gen.shape)

    # net = AttRegressor((10, 128, 128, 128),
    #                    channels=(16, 32, 64, 128, 1), strides=(1, 2, 2, 2, 1), padding=(1, 1, 1, 1, 1),
    #                    kernel_size=(3, 4, 4, 4, 3), num_res_units=0, std_noise=0.01)
    # init_weights(net, init_type='normal')

    # att_block = AttGate2022(in_ch=1, num_res_units=-1)
    # out = att_block(inp, inp)
    # print(out.shape)

    # [x2a, x4a, x6a, x8a, x10a] = encoder(inp)
    # [x2a, x4a, x6a, x8a] = decoder([x2a, x4a, x6a, x8a, x10a])
    # print(x2a.shape, x4a.shape, x6a.shape, x8a.shape)
    # print(x2a.shape, x4a.shape, x6a.shape, x8a.shape, x10a.shape)
    # out = net(out_gen, inp)
    # print(out[0].shape, out[1].shape, out[2].shape, out[3].shape)
    # print(len(out[1:]))
    # print(dec1.shape)


if __name__ == '__main__':
    test()
