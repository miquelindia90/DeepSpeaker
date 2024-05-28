from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def getResnetOutputDimension(inputDimension, outputChannel=128):
    outputDimension = np.ceil(np.array(inputDimension, dtype=np.float32) / 2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32) / 2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32) / 2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32) / 2)
    return int(outputDimension) * outputChannel


class ResnetBlock(torch.nn.Module):
    def __init__(self, kernel_size_input, kernel_size, activation):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            kernel_size_input, kernel_size, 3, stride=1, padding=1
        )
        self.bn1 = torch.nn.BatchNorm2d(kernel_size)
        self.conv2 = torch.nn.Conv2d(kernel_size, kernel_size, 3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(kernel_size)
        self.activation1 = torch.nn.GELU() if activation=="Attention" else torch.nn.GELU()
        self.activation2 = torch.nn.GELU() if activation=="Attention" else torch.nn.GELU()
        self.residual_condition = True if kernel_size_input == kernel_size else False

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.residual_condition:
            out += x
        out = self.activation2(out)
        return out


class Resnet(torch.nn.Module):
    def __init__(self, kernel_size, group_blocks, activation="Relu"):
        super(Resnet, self).__init__()
        self.group_blocks = group_blocks
        self.group_kernels = [
            kernel_size // 8,
            kernel_size // 4,
            kernel_size // 2,
            kernel_size,
        ]
        self.__init_blocks(activation)

    def __init_resnet_group(self, input_kernel_size, kernel_size, num_blocks, activation):
        layers = OrderedDict()
        layers["block0"] = ResnetBlock(input_kernel_size, kernel_size, activation)
        for i in range(num_blocks - 1):
            layers["block" + str(i + 1)] = ResnetBlock(kernel_size, kernel_size, activation)
        return torch.nn.Sequential(layers)

    def __init_blocks(self, activation):
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.block1 = self.__init_resnet_group(
            1, self.group_kernels[0], self.group_blocks[0], activation
        )
        self.block2 = self.__init_resnet_group(
            self.group_kernels[0], self.group_kernels[1], self.group_blocks[1], activation
        )
        self.block3 = self.__init_resnet_group(
            self.group_kernels[1], self.group_kernels[2], self.group_blocks[2], activation
        )
        self.block4 = self.__init_resnet_group(
            self.group_kernels[2], self.group_kernels[3], self.group_blocks[3], activation
        )

    def forward(self, paddedInputTensor):
        paddedInputTensor = paddedInputTensor.view(
            paddedInputTensor.size(0),
            paddedInputTensor.size(1),
            1,
            paddedInputTensor.size(2),
        ).transpose(1, 2)

        block1_output = self.maxpool(self.block1(paddedInputTensor))
        block2_output = self.maxpool(self.block2(block1_output))
        block3_output = self.maxpool(self.block3(block2_output))
        block4_output = self.maxpool(self.block4(block3_output))

        outputTensor = block4_output.transpose(1, 2)
        outputTensor = outputTensor.contiguous().view(
            outputTensor.size(0),
            outputTensor.size(1),
            outputTensor.size(2) * outputTensor.size(3),
        )

        return outputTensor


class Resnet34(Resnet):
    def __init__(self, kernel_size, activation="Relu"):
        super(Resnet34, self).__init__(kernel_size, [3, 4, 6, 3], activation)


class Resnet101(Resnet):
    def __init__(self, kernel_size, activation="Relu"):
        super(Resnet101, self).__init__(kernel_size, [3, 4, 23, 3], activation)
