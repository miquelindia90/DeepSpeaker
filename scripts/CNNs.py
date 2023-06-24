from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def getVGG4LOutputDimension(inputDimension, outputChannel=128):
    outputDimension = np.ceil(np.array(inputDimension, dtype=np.float32) / 2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32) / 2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32) / 2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32) / 2)
    return int(outputDimension) * outputChannel


class VGG4L(torch.nn.Module):
    def __init__(self, kernel_size):
        super(VGG4L, self).__init__()

        self.conv11 = torch.nn.Conv2d(1, int(kernel_size / 8), 3, stride=1, padding=1)
        self.conv12 = torch.nn.Conv2d(
            int(kernel_size / 8), int(kernel_size / 8), 3, stride=1, padding=1
        )
        self.conv21 = torch.nn.Conv2d(
            int(kernel_size / 8), int(kernel_size / 4), 3, stride=1, padding=1
        )
        self.conv22 = torch.nn.Conv2d(
            int(kernel_size / 4), int(kernel_size / 4), 3, stride=1, padding=1
        )
        self.conv31 = torch.nn.Conv2d(
            int(kernel_size / 4), int(kernel_size / 2), 3, stride=1, padding=1
        )
        self.conv32 = torch.nn.Conv2d(
            int(kernel_size / 2), int(kernel_size / 2), 3, stride=1, padding=1
        )
        self.conv41 = torch.nn.Conv2d(
            int(kernel_size / 2), int(kernel_size), 3, stride=1, padding=1
        )
        self.conv42 = torch.nn.Conv2d(
            int(kernel_size), int(kernel_size), 3, stride=1, padding=1
        )

    def forward(self, paddedInputTensor):
        paddedInputTensor = paddedInputTensor.view(
            paddedInputTensor.size(0),
            paddedInputTensor.size(1),
            1,
            paddedInputTensor.size(2),
        ).transpose(1, 2)

        encodedTensorLayer1 = F.relu(self.conv11(paddedInputTensor))
        encodedTensorLayer1 = F.relu(self.conv12(encodedTensorLayer1))
        encodedTensorLayer1 = F.max_pool2d(
            encodedTensorLayer1, 2, stride=2, ceil_mode=True
        )

        encodedTensorLayer2 = F.relu(self.conv21(encodedTensorLayer1))
        encodedTensorLayer2 = F.relu(self.conv22(encodedTensorLayer2))
        encodedTensorLayer2 = F.max_pool2d(
            encodedTensorLayer2, 2, stride=2, ceil_mode=True
        )

        encodedTensorLayer3 = F.relu(self.conv31(encodedTensorLayer2))
        encodedTensorLayer3 = F.relu(self.conv32(encodedTensorLayer3))
        encodedTensorLayer3 = F.max_pool2d(
            encodedTensorLayer3, 2, stride=2, ceil_mode=True
        )

        encodedTensorLayer4 = F.relu(self.conv41(encodedTensorLayer3))
        encodedTensorLayer4 = F.relu(self.conv42(encodedTensorLayer4))
        encodedTensorLayer4 = F.max_pool2d(
            encodedTensorLayer4, 2, stride=2, ceil_mode=True
        )

        outputTensor = encodedTensorLayer4.transpose(1, 2)
        outputTensor = outputTensor.contiguous().view(
            outputTensor.size(0),
            outputTensor.size(1),
            outputTensor.size(2) * outputTensor.size(3),
        )

        return outputTensor


class ResnetBlock(torch.nn.Module):
    def __init__(self, kernel_size_input, kernel_size):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            kernel_size_input, kernel_size, 3, stride=1, padding=1
        )
        self.bn1 = torch.nn.BatchNorm2d(kernel_size)
        self.conv2 = torch.nn.Conv2d(kernel_size, kernel_size, 3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(kernel_size)
        self.gelu = torch.nn.GELU()
        if kernel_size_input == kernel_size:
            self.residual_condition = True

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.residual_condition:
            out += x
        out = self.gelu(out)
        return out


class Resnet34(torch.nn.Module):
    def __init__(self, kernel_size):
        super(Resnet34, self).__init__()
        self.group_blocks = [3, 4, 6, 3]
        self.group_kernels = [
            kernel_size // 8,
            kernel_size // 4,
            kernel_size // 2,
            kernel_size,
        ]
        self.__init_blocks()

    def __init_resnet_group(self, input_kernel_size, kernel_size, num_blocks):
        layers = OrderedDict()
        layers["block0"] = ResnetBlock(input_kernel_size, kernel_size)
        for i in range(num_blocks - 1):
            layers["block" + str(i + 1)] = ResnetBlock(kernel_size, kernel_size)
        return torch.nn.Sequential(layers)

    def __init_blocks(self):
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.block1 = self.__init_resnet_group(
            1, self.group_kernels[0], self.group_blocks[0]
        )
        self.block2 = self.__init_resnet_group(
            self.group_kernels[0], self.group_kernels[1], self.group_blocks[1]
        )
        self.block3 = self.__init_resnet_group(
            self.group_kernels[1], self.group_kernels[2], self.group_blocks[2]
        )
        self.block4 = self.__init_resnet_group(
            self.group_kernels[2], self.group_kernels[3], self.group_blocks[3]
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
