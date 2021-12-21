"""
Model C implementations for ...

Striving for Simplicity: The All Convolutional Net
https://arxiv.org/pdf/1412.6806.pdf
"""
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from wz_torch.vision import ConvBatchRelu


class StrivingSimplicityC(Module):
    """ Model C. """
    def __init__(self, num_classes: int) -> None:
        """ Constructor.

        :param num_classes: number of target classes
        :type  num_classes: int
        """
        super(StrivingSimplicityC, self).__init__()
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            ConvBatchRelu(in_channels=3, out_channels=96, padding=0),
            ConvBatchRelu(in_channels=96, out_channels=96, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),  # stage 1
            ConvBatchRelu(in_channels=96, out_channels=192, padding=0),
            ConvBatchRelu(in_channels=192, out_channels=192, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),  # stage 2
            ConvBatchRelu(in_channels=192, out_channels=192, padding=0),
            ConvBatchRelu(in_channels=192,
                          out_channels=192,
                          kernel_size=1,
                          padding=0),
            ConvBatchRelu(in_channels=192,
                          out_channels=self.num_classes,
                          kernel_size=1,
                          padding=0),  # neck
        )

    def __repr__(self) -> str:
        """ Representation.

        :return: instance representation
        :rtype:  str
        """
        return f'StrivingSimplicityC(num_classes={self.num_classes})'

    def forward(self, inp: Tensor) -> Tensor:
        """ Compute classification logits.

        :param inp: RGB images (N, 3, H, W)
        :type  inp: Tensor
        :return:    logits (N, num_classes)
        :rtype:     Tensor
        """
        featmap = self.backbone(inp)
        return featmap.mean(dim=-1).mean(dim=-1)  # global average pooling
