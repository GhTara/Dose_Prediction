from typing import Optional, Sequence, Tuple, Union

import torch.nn as nn

from monai.networks.blocks import ADN
from monai.networks.layers.convutils import same_padding
from monai.networks.layers.factories import Act, Norm
from monai.networks.blocks.dynunet_block import get_padding, get_output_padding

from OARSegmentation.Models.Nets.convs import MultiScaleConv


class MultiScaleConvolution(nn.Sequential):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            strides: Union[Sequence[int], int] = 1,
            kernel_size: Union[Sequence[int], int] = 3,
            adn_ordering: str = "NDA",
            act: Optional[Union[Tuple, str]] = "PRELU",
            norm: Optional[Union[Tuple, str]] = "INSTANCE",
            dropout: Optional[Union[Tuple, str, float]] = None,
            dropout_dim: Optional[int] = 1,
            dilation: Union[Sequence[int], int] = 1,
            groups: int = 1,
            bias: bool = True,
            conv_only: bool = False,
            is_transposed: bool = False,
            padding: Optional[Union[Sequence[int], int]] = None,
            output_padding: Optional[Union[Sequence[int], int]] = None,
            dimensions: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dimensions = spatial_dims if dimensions is None else dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_transposed = is_transposed

        if padding is None:
            padding = same_padding(kernel_size, dilation)

        conv: nn.Module
        conv = MultiScaleConv(
            ch_in=in_channels,
            ch_out=out_channels,
        )
        self.add_module("conv", conv)

        if conv_only:
            return
        if act is None and norm is None and dropout is None:
            return
        self.add_module(
            "adn",
            ADN(
                ordering=adn_ordering,
                in_channels=out_channels,
                act=act,
                norm=norm,
                norm_dim=self.dimensions,
                dropout=dropout,
                dropout_dim=dropout_dim,
            ),
        )


def get_multi_conv_layer(
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int] = 3,
        stride: Union[Sequence[int], int] = 1,
        act: Optional[Union[Tuple, str]] = Act.PRELU,
        norm: Optional[Union[Tuple, str]] = Norm.INSTANCE,
        dropout: Optional[Union[Tuple, str, float]] = None,
        bias: bool = False,
        conv_only: bool = True,
        is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return MultiScaleConvolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )
