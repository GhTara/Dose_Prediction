import functools

import torch
import torch.nn as nn
from torch.autograd import Variable


##############################################################################
# Classes
##############################################################################

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# 3D version of UnetGenerator
class UnetGenerator3d(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm3d, use_dropout=False, gpu_ids=[]):  # TODO
        super(UnetGenerator3d, self).__init__()
        self.gpu_ids = gpu_ids

        # currently, support only input_nc == output_nc
        # assert (input_nc == output_nc)

        self.initial_block = nn.Sequential(*[nn.Conv3d(input_nc, ngf, kernel_size=4, stride=1, padding=3, dilation=2),
                                             norm_layer(ngf),
                                             nn.LeakyReLU(0.2, )])
        # construct unet structure
        unet_block = UnetSkipConnectionBlock3d(in_nc=ngf * 8, down_nc=ngf * 8, up_nc=ngf * 8, norm_layer=norm_layer,
                                               innermost=True)

        unet_block = UnetSkipConnectionBlock3d(in_nc=ngf * 8, down_nc=ngf * 8, up_nc=ngf * 16, submodule=unet_block,
                                               norm_layer=norm_layer)

        unet_block = UnetSkipConnectionBlock3d(in_nc=ngf * 4, down_nc=ngf * 8, up_nc=ngf * 16, submodule=unet_block,
                                               norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock3d(in_nc=ngf * 2, down_nc=ngf * 4, up_nc=ngf * 8, submodule=unet_block,
                                               norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock3d(in_nc=ngf, down_nc=ngf * 2, up_nc=ngf * 4, submodule=unet_block,
                                               norm_layer=norm_layer)
        self.outer_block = nn.Sequential(
            *[nn.Conv3d(ngf * 2, output_nc, kernel_size=4, stride=1, padding=3, dilation=2),
              nn.Tanh()])

        self.model = unet_block

    def forward(self, input):
        input = self.initial_block(input)
        inner = self.model(input)
        out = self.outer_block(inner)
        return out


class AttGate(nn.Module):
    def __init__(self, in_ch):
        super(AttGate, self).__init__()

        self.initial_conv = nn.Conv3d(in_ch, in_ch,
                                      kernel_size=1, stride=1, padding='same')
        self.intermediate = nn.Sequential(
            nn.ReLU(),
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


# Defines the submodule with skip connection.
class UnetSkipConnectionBlock3d(nn.Module):
    def __init__(self, in_nc, down_nc, up_nc,
                 submodule=None, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False):
        super(UnetSkipConnectionBlock3d, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        downconv = nn.Conv3d(in_nc, down_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, )
        downnorm = norm_layer(down_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(in_nc)

        if innermost:
            upconv = nn.ConvTranspose3d(up_nc, in_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(up_nc, in_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)
        self.att_gate = AttGate(in_ch=in_nc)

    def forward(self, x):
        z = self.model(x)
        att_x = self.att_gate(z, x)
        return torch.cat([att_x, z], 1)


class BlockDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf, norm_layer=nn.BatchNorm3d):
        super(BlockDiscriminator, self).__init__()
        self.downsample = nn.Sequential(*[nn.Conv3d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                                          norm_layer(ndf),
                                          nn.LeakyReLU(0.2, )])
        self.pooling = nn.Sequential(*[nn.Conv3d(input_nc, ndf, kernel_size=4, stride=1, padding=3, dilation=2),
                                       norm_layer(ndf),
                                       nn.LeakyReLU(0.2, )])
        self.att_gate = AttGate(in_ch=ndf)

    def forward(self, x):
        z1 = self.downsample(x)
        z2 = self.pooling(z1)
        out = self.att_gate(z2, z1)
        return torch.cat([out, z2], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        # Initial layer
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=4, stride=1, padding=3, dilation=2),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, )
        ]

        # Inner layer
        n = 0
        for n in range(n_layers):
            sequence += [
                BlockDiscriminator(input_nc=(2 ** n) * ndf, ndf=(2 ** n) * ndf, norm_layer=norm_layer)
            ]
        # Last layer
        sequence += [
            nn.Conv3d((2 ** (n + 1)) * ndf, 1, kernel_size=4, stride=1, padding=3, dilation=2),
            norm_layer(1),
            nn.LeakyReLU(0.2, )
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

