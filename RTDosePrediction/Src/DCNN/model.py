import torch
import torch.nn as nn


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(SingleConv, self).__init__()

        self.single_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)


class DenseFeaureAggregation(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch):
        super(DenseFeaureAggregation, self).__init__()

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=1 * in_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, base_ch, dilation=2, kernel_size=3, padding=2, stride=1, bias=True),

        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_ch + base_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch + base_ch, base_ch, dilation=3, kernel_size=3, padding=3, stride=1, bias=True),

        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_ch + 2 * base_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch + 2 * base_ch, base_ch, dilation=5, kernel_size=3, padding=5, stride=1, bias=True),

        )
        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_ch + 3 * base_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch + 3 * base_ch, base_ch, dilation=7, kernel_size=3, padding=7, stride=1, bias=True),

        )
        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_ch + 4 * base_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch + 4 * base_ch, base_ch, dilation=9, kernel_size=3, padding=9, stride=1, bias=True),

        )

        self.conv_out = nn.Sequential(
            nn.BatchNorm2d(num_features=in_ch + 5 * base_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch + 5 * base_ch, out_ch, dilation=1, kernel_size=1, padding=0, stride=1, bias=True),
        )

    def forward(self, x):
        out_ = self.conv1(x)
        concat_ = torch.cat((out_, x), dim=1)
        out_ = self.conv2(concat_)
        concat_ = torch.cat((concat_, out_), dim=1)
        out_ = self.conv3(concat_)
        concat_ = torch.cat((concat_, out_), dim=1)
        out_ = self.conv4(concat_)
        concat_ = torch.cat((concat_, out_), dim=1)
        out_ = self.conv5(concat_)
        concat_ = torch.cat((concat_, out_), dim=1)
        out_ = self.conv_out(concat_)
        return out_


class Encoder(nn.Module):
    def __init__(self, in_ch, list_ch):
        super(Encoder, self).__init__()
        self.encoder_1 = nn.Sequential(
            SingleConv(in_ch, list_ch[1], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1)
        )
        self.encoder_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            SingleConv(list_ch[1], list_ch[2], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1)
        )
        self.encoder_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            SingleConv(list_ch[2], list_ch[3], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1)
        )
        self.encoder_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            SingleConv(list_ch[3], list_ch[4], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1)
        )

        self.DFA = DenseFeaureAggregation(list_ch[4], list_ch[4], list_ch[4])

    def forward(self, x):
        out_encoder_1 = self.encoder_1(x)
        out_encoder_2 = self.encoder_2(out_encoder_1)
        out_encoder_3 = self.encoder_3(out_encoder_2)
        out_encoder_4 = self.encoder_4(out_encoder_3)
        out_encoder_4 = self.DFA(out_encoder_4)
        return [out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4]


class Decoder(nn.Module):
    def __init__(self, out_ch, list_ch):
        super(Decoder, self).__init__()

        self.upconv_3_1 = nn.ConvTranspose2d(list_ch[4], list_ch[3], kernel_size=2, stride=2, bias=True)
        self.decoder_conv_3_1 = nn.Sequential(
            SingleConv(2 * list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1)
        )
        self.upconv_2_1 = nn.ConvTranspose2d(list_ch[3], list_ch[2], kernel_size=2, stride=2, bias=True)
        self.decoder_conv_2_1 = nn.Sequential(
            SingleConv(2 * list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1)
        )
        self.upconv_1_1 = nn.ConvTranspose2d(list_ch[2], list_ch[1], kernel_size=2, stride=2, bias=True)
        self.decoder_conv_1_1 = nn.Sequential(
            SingleConv(2 * list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(list_ch[1], out_ch, kernel_size=1, padding=0, bias=True)
        )

    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4 = out_encoder

        out_decoder_3_1 = self.decoder_conv_3_1(
            torch.cat((self.upconv_3_1(out_encoder_4), out_encoder_3), dim=1)
        )
        out_decoder_2_1 = self.decoder_conv_2_1(
            torch.cat((self.upconv_2_1(out_decoder_3_1), out_encoder_2), dim=1)
        )
        out_decoder_1_1 = self.decoder_conv_1_1(
            torch.cat((self.upconv_1_1(out_decoder_2_1), out_encoder_1), dim=1)
        )

        output = self.conv_out(out_decoder_1_1)
        return [output]


class Model(nn.Module):
    def __init__(self, in_ch, out_ch, list_ch):
        super(Model, self).__init__()
        self.encoder = Encoder(in_ch, list_ch)
        self.decoder = Decoder(out_ch, list_ch)

        # init
        self.initialize()

    @staticmethod
    def init_conv_deconv_BN(modules):
        for m in modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def initialize(self):
        print('# random init encoder weight using nn.init.kaiming_uniform !')
        self.init_conv_deconv_BN(self.decoder.modules)
        print('# random init decoder weight using nn.init.kaiming_uniform !')
        self.init_conv_deconv_BN(self.encoder.modules)

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)  # is a list

        return out_decoder

