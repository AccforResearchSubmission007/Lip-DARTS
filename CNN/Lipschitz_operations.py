import torch
import torch.nn as nn
import torch.nn.functional as F


class LipschitzBaseOp(nn.Module):
    def __init__(self):
        super().__init__()
        self.lipschitz_const = 1.0

    def lipschitz_bound(self):
        return self.lipschitz_const


OPS = {
    'none': lambda C, stride, affine: LipschitzZero(stride),
    'avg_pool_3x3': lambda C, stride, affine: LipschitzAvgPool(3, stride, padding=1),
    'max_pool_3x3': lambda C, stride, affine: LipschitzMaxPool(3, stride, padding=1),
    'skip_connect': lambda C, stride, affine: LipschitzIdentity() if stride == 1 else LipschitzFactorizedReduce(C, C,
                                                                                                                affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: LipschitzSepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: LipschitzSepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: LipschitzSepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: LipschitzDilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: LipschitzDilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: LipschitzSeqConv(C, C, (1, 7), (7, 1), stride, affine=affine),
}


class LipschitzReLUConvBN(LipschitzBaseOp):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.affine = affine

        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
        )

        if affine:
            self.gamma = nn.Parameter(torch.ones(1, C_out, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, C_out, 1, 1))
            self.gamma.data.clamp_(max=1.0)

    def forward(self, x):
        x = F.relu(x)
        x = self.conv(x)
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def lipschitz_bound(self):
        bound = 1.0
        if self.affine:
            bound *= self.gamma.data.abs().max().item()
        return bound


class LipschitzDilConv(LipschitzBaseOp):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.affine = affine

        self.depth_conv = nn.utils.spectral_norm(
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation,
                      groups=C_in, bias=False)
        )
        self.point_conv = nn.utils.spectral_norm(
            nn.Conv2d(C_in, C_out, 1, padding=0, bias=False)
        )
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, C_out, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, C_out, 1, 1))
            self.gamma.data.clamp_(max=1.0)

    def forward(self, x):
        x = F.relu(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def lipschitz_bound(self):
        bound = 1.0
        if self.affine:
            bound *= self.gamma.data.abs().max().item()
        return bound


class LipschitzSepConv(LipschitzBaseOp):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.affine = affine

        self.depth_conv1 = nn.utils.spectral_norm(
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride, padding=padding,
                      groups=C_in, bias=False)
        )
        self.point_conv1 = nn.utils.spectral_norm(
            nn.Conv2d(C_in, C_in, 1, padding=0, bias=False)
        )

        self.depth_conv2 = nn.utils.spectral_norm(
            nn.Conv2d(C_in, C_in, kernel_size, stride=1, padding=padding,
                      groups=C_in, bias=False)
        )
        self.point_conv2 = nn.utils.spectral_norm(
            nn.Conv2d(C_in, C_out, 1, padding=0, bias=False)
        )

        if affine:
            self.gamma1 = nn.Parameter(torch.ones(1, C_in, 1, 1))
            self.beta1 = nn.Parameter(torch.zeros(1, C_in, 1, 1))
            self.gamma2 = nn.Parameter(torch.ones(1, C_out, 1, 1))
            self.beta2 = nn.Parameter(torch.zeros(1, C_out, 1, 1))
            self.gamma1.data.clamp_(max=1.0)
            self.gamma2.data.clamp_(max=1.0)

    def forward(self, x):
        x = F.relu(x)

        x = self.depth_conv1(x)
        x = self.point_conv1(x)
        if self.affine:
            x = x * self.gamma1 + self.beta1

        x = F.relu(x)

        x = self.depth_conv2(x)
        x = self.point_conv2(x)
        if self.affine:
            x = x * self.gamma2 + self.beta2
        return x

    def lipschitz_bound(self):
        bound = 1.0 * 1.0
        if self.affine:
            bound *= self.gamma1.data.abs().max().item() * self.gamma2.data.abs().max().item()
        return bound


class LipschitzIdentity(LipschitzBaseOp):
    def __init__(self):
        super().__init__()
        self.lipschitz_const = 1.0

    def forward(self, x):
        return x


class LipschitzZero(LipschitzBaseOp):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
        self.epsilon = nn.Parameter(torch.tensor(1e-4))
        if stride != 1:
            self.downsample = nn.AvgPool2d(kernel_size=stride, stride=stride)
        self.lipschitz_const = self.epsilon.item()

    def forward(self, x):
        if self.stride == 1:
            return x * self.epsilon
        else:
            return self.downsample(x) * self.epsilon

    def lipschitz_bound(self):
        return self.epsilon.data.abs().item()


class LipschitzFactorizedReduce(LipschitzBaseOp):
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        assert C_out % 2 == 0
        self.affine = affine

        self.conv_1 = nn.utils.spectral_norm(
            nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False))
        self.conv_2 = nn.utils.spectral_norm(
            nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False))

        if affine:
            self.gamma = nn.Parameter(torch.ones(1, C_out, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, C_out, 1, 1))
            self.gamma.data.clamp_(max=1.0)

    def forward(self, x):
        x = F.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        if self.affine:
            out = out * self.gamma + self.beta
        return out

    def lipschitz_bound(self):
        bound = 1.0
        if self.affine:
            bound *= self.gamma.data.abs().max().item()
        return bound


class LipschitzAvgPool(LipschitzBaseOp):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding, count_include_pad=False)
        self.lipschitz_const = 1.0

    def forward(self, x):
        return self.pool(x)


class LipschitzMaxPool(LipschitzBaseOp):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.lipschitz_const = 1.0

    def forward(self, x):
        return self.pool(x)


class LipschitzSeqConv(LipschitzBaseOp):
    def __init__(self, C_in, C_out, kernel1, kernel2, stride, affine=True):
        super().__init__()
        self.affine = affine

        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(C_in, C_out, kernel1, stride=(1, stride), padding=(0, kernel1[1] // 2), bias=False))
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(C_out, C_out, kernel2, stride=(stride, 1), padding=(kernel2[0] // 2, 0), bias=False))

        if affine:
            self.gamma = nn.Parameter(torch.ones(1, C_out, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, C_out, 1, 1))
            self.gamma.data.clamp_(max=1.0)

    def forward(self, x):
        x = F.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def lipschitz_bound(self):
        bound = 1.0  # ReLU
        if self.affine:
            bound *= self.gamma.data.abs().max().item()
        return bound