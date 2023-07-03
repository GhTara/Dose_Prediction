"""
The implementation of transformer part is inspired by monai:
https://arxiv.org/abs/2211.02701
"""

from typing import Sequence, Union, Tuple

import numpy as np

from monai.networks.nets.vit import ViT
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep

from OARSegmentation.Models.Nets.base_blocks import ModifiedUnetrUpBlock
from NetworkTrainer.network_trainer import *
from DosePrediction.Models.Networks.c3d import BaseUNet


##############################
#        Encoder
##############################
class ViTEncoder(nn.Module):

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


##############################
#        Decoder
##############################
class PyMSCDecoder(nn.Module):

    def __init__(
            self,
            feature_size: int = 16,
            hidden_size: int = 768,
            norm_name: Union[Tuple, str] = "instance",
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


##############################
#        Generator
##############################
class MainSubsetModel(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 img_size,
                 feature_size: int = 16,
                 hidden_size: int = 768,
                 mlp_dim: int = 3072,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 conv_block: bool = True,
                 res_block: bool = True,
                 dropout_rate: float = 0.0,
                 mode_multi_dec=False,
                 act='relu',
                 multiS_conv=True):
        super().__init__()

        # ----- Encoder part ----- #
        self.encoder = ViTEncoder(
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
            self.decoder = PyMSCDecoder(
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


##############################
#        Model
##############################
class Model(nn.Module):
    def __init__(self, in_ch,
                 out_ch,
                 list_ch_A,
                 feature_size=16,
                 img_size=(128, 128, 128),
                 num_layers=8,  # 4, 8, 12
                 num_heads=6,  # 3, 6, 12
                 act='mish',
                 mode_multi_dec=True,
                 multiS_conv=True,
                 ):
        super(Model, self).__init__()

        # list_ch records the number of channels in each stage, eg. [-1, 32, 64, 128, 256, 512]
        self.net_A = BaseUNet(in_ch, list_ch_A)
        self.net_B = MainSubsetModel(
            in_ch=in_ch + list_ch_A[1],
            out_ch=out_ch,
            feature_size=feature_size,
            img_size=img_size,
            num_layers=num_layers,  # 4, 8, 12
            num_heads=num_heads,  # 3, 6, 12
            act=act,
            mode_multi_dec=mode_multi_dec,
            multiS_conv=multiS_conv,
        )

        self.conv_out_A = nn.Conv3d(list_ch_A[1], out_ch, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out_net_A = self.net_A(x)
        out_net_B = self.net_B(torch.cat((out_net_A, x), dim=1))

        output_A = self.conv_out_A(out_net_A)
        return [output_A, out_net_B]


def create_pretrained_unet(
        ckpt_file,
        in_ch,
        out_ch,
        list_ch_A,
        feature_size,
        img_size,
        num_layers=8,  # 4, 8, 12
        num_heads=6,  # 3, 6, 12
        act='mish',
        mode_multi_dec=True,
        multiS_conv=True,
):
    trainer = NetworkTrainer()
    pretrain = trainer.init_trainer(
        ckpt_file=ckpt_file,
        list_GPU_ids=[0],
        only_network=True)

    net = Model(in_ch,
                out_ch,
                list_ch_A,
                feature_size=feature_size,
                img_size=img_size,
                num_layers=num_layers,  # 4, 8, 12
                num_heads=num_heads,  # 3, 6, 12
                act=act,
                mode_multi_dec=mode_multi_dec,
                multiS_conv=multiS_conv,
                )

    net_dict = net.state_dict()
    # pretrain['network_state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
    missing = tuple({k for k in net_dict.keys() if k not in pretrain['network_state_dict']})
    print(f"missing in pretrained: {len(missing)}")
    inside = tuple({k for k in pretrain['network_state_dict'] if k in net_dict.keys()})
    print(f"inside pretrained: {len(inside)}")
    unused = tuple({k for k in pretrain['network_state_dict'] if k not in net_dict.keys()})
    print(f"unused pretrained: {len(unused)}")
    # assert len(inside) > len(missing)
    # assert len(inside) > len(unused)

    pretrain['network_state_dict'] = {k: v for k, v in pretrain['network_state_dict'].items() if k in net_dict.keys()}
    net.load_state_dict(pretrain['network_state_dict'], strict=False)
    return net, inside
