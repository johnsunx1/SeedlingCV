########################################################################
# This software is Copyright 2022 The Regents of Shandong Agricultural University.
# All Rights Reserved.
#
# Permission to copy, modify, and distribute this software and its
# documentation for educational, research and non-profit purposes, without fee,
# and without a written agreement is hereby granted, provided that the above
# copyright notice, this paragraph and the following two paragraphs appear
# in all copies.
#
# This software program and documentation are copyrighted by The Regents of
# Shandong Agricultural University. The software program and documentation are supplied
# "as is", without any accompanying services from The Regents. The Regents does
# not warrant that the operation of the program will be uninterrupted or
# error-free. The end-user understands that the program was developed for
# research purposes and is advised not to rely exclusively on the program for
# any reason.
#

#Author: Qikun Zhao, Chuanfeng Zhao (https://github.com/14385423,https://github.com/ZCFzhaochuanfei)
#For commermical usage, please contact corresponding author (johnsunx1@yahoo.com)
########################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F


# SiLU https://arxiv.org/pdf/1606.08415.pdf ----------------------------------------------------------------------------
class SiLU(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX


class MemoryEfficientSwish(nn.Module):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * torch.sigmoid(x)

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            return grad_output * (sx * (1 + x * (1 - sx)))

    def forward(self, x):
        return self.F.apply(x)


# Mish https://github.com/digantamisra98/Mish --------------------------------------------------------------------------
class Mish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()


class MemoryEfficientMish(nn.Module):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        return self.F.apply(x)


# FReLU https://arxiv.org/abs/2007.11824 -------------------------------------------------------------------------------
class FReLU(nn.Module):
    def __init__(self, c1, k=3):  # ch_in, kernel
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))
