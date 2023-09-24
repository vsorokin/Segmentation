from collections import OrderedDict

import torch.nn as nn
import torch
import logging


class UNet(nn.Module):
    def __init__(self, n=30):
        """
        :param n: number of filters in the first layer
        """
        super().__init__()

        # contracting path (left side)
        self.conv1_1 = self.conv(in_channels=4, out_channels=n)
        self.conv1_2 = self.conv(in_channels=n, out_channels=n)

        self.conv2_1 = self.conv(in_channels=n, out_channels=2 * n)
        self.conv2_2 = self.conv(in_channels=2 * n, out_channels=2 * n)

        self.conv3_1 = self.conv(in_channels=2 * n, out_channels=4 * n)
        self.conv3_2 = self.conv(in_channels=4 * n, out_channels=4 * n)

        self.conv4_1 = self.conv(in_channels=4 * n, out_channels=8 * n)
        self.conv4_2 = self.conv(in_channels=8 * n, out_channels=8 * n)

        self.conv5_1 = self.conv(in_channels=8 * n, out_channels=16 * n)
        self.conv5_2 = self.conv(in_channels=16 * n,
                                 out_channels=8 * n)  # reducing numbers of filters before upsampling, as the paper recommends

        self.max_pool = nn.MaxPool3d(kernel_size=2)

        # expansive path (right side)
        self.conv_right_4_1 = self.conv(in_channels=8 * n + 8 * n, out_channels=8 * n)
        self.conv_right_4_2 = self.conv(in_channels=8 * n, out_channels=4 * n)  # ditto

        self.conv_right_3_1 = self.conv(in_channels=4 * n + 4 * n, out_channels=4 * n)
        self.conv_right_3_2 = self.conv(in_channels=4 * n, out_channels=2 * n)  # ditto

        self.conv_right_2_1 = self.conv(in_channels=2 * n + 2 * n, out_channels=2 * n)
        self.conv_right_2_2 = self.conv(in_channels=2 * n, out_channels=n)  # ditto

        self.conv_right_1_1 = self.conv(in_channels=n + n, out_channels=n)
        self.conv_right_1_2 = self.conv(in_channels=n, out_channels=n)

        self.final_1x1x1_conv = self.conv(in_channels=n, out_channels=4, kernel_size=1, padding=0)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_in, evaluating=False):
        # contracting path (left side)
        logging.info("contracting path (left side)")
        self.print_shape(x_in, "x_in")
        x = self.conv1_1(x_in)
        self.print_shape(x, "conv1_1")
        x_1 = self.conv1_2(x)
        self.print_shape(x_1, "conv1_2")

        x = self.max_pool(x_1)
        self.print_shape(x, "max_pool")
        x = self.conv2_1(x)
        self.print_shape(x, "conv2_1")
        x_2 = self.conv2_2(x)
        self.print_shape(x_2, "conv2_2")

        x = self.max_pool(x_2)
        self.print_shape(x, "max_pool")
        x = self.conv3_1(x)
        self.print_shape(x, "conv3_1")
        x_3 = self.conv3_2(x)
        self.print_shape(x_3, "conv3_2")

        x = self.max_pool(x_3)
        self.print_shape(x, "max_pool")
        x = self.conv4_1(x)
        self.print_shape(x, "conv4_1")
        x_4 = self.conv4_2(x)
        self.print_shape(x_4, "conv4_2")

        x = self.max_pool(x_4)
        self.print_shape(x, "max_pool")
        x = self.conv5_1(x)
        self.print_shape(x, "conv5_1")
        x_5 = self.conv5_2(x)
        self.print_shape(x_5, "conv5_2")

        # expansive path (right side)
        logging.info("expansive path (right side)")
        x = self.upsample(x_5)
        self.print_shape(x, "upsample")
        x = torch.cat([x_4, x], dim=1)
        self.print_shape(x, "cat")
        x = self.conv_right_4_1(x)
        self.print_shape(x, "conv_right_4_1")
        x = self.conv_right_4_2(x)
        self.print_shape(x, "conv_right_4_2")

        x = self.upsample(x)
        self.print_shape(x, "upsample")
        x = torch.cat([x_3, x], dim=1)
        self.print_shape(x, "cat")
        x = self.conv_right_3_1(x)
        self.print_shape(x, "conv_right_3_1")
        x = self.conv_right_3_2(x)
        self.print_shape(x, "conv_right_3_2")

        x = self.upsample(x)
        self.print_shape(x, "upsample")
        x = torch.cat([x_2, x], dim=1)
        self.print_shape(x, "cat")
        x = self.conv_right_2_1(x)
        self.print_shape(x, "conv_right_2_1")
        x = self.conv_right_2_2(x)
        self.print_shape(x, "conv_right_2_2")

        x = self.upsample(x)
        self.print_shape(x, "upsample")
        x = torch.cat([x_1, x], dim=1)
        self.print_shape(x, "cat")
        x = self.conv_right_1_1(x)
        self.print_shape(x, "conv_right_1_1")
        x = self.conv_right_1_2(x)
        self.print_shape(x, "conv_right_1_2")

        x = self.final_1x1x1_conv(x)
        self.print_shape(x, "final_1x1x1_conv")

        result = self.softmax(x)
        self.print_shape(result, "softmax (result)")
        return result

    def conv(self, in_channels, out_channels, kernel_size=3, padding=1, bias=True):
        return nn.Sequential(OrderedDict({
            'conv': nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias),
            'relu': nn.ReLU(inplace=True),
            'batch_norm': nn.BatchNorm3d(out_channels)
        }))

    def print_shape(self, tensor, prefix=""):
        logging.info("%s%s", f"[{prefix}] " if prefix else "", tensor.shape)
