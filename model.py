import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint


class ODENet(nn.Module):
    def __init__(self, in_ch, n_filters=64, downsample='residual', tol=1e-3, adjoint=False, t1=1):
        super(ODENet, self).__init__()

        if downsample == 'residual':
            self.downsample = ResDownsample(in_ch, out_ch=n_filters)
        elif downsample == 'convolution':
            self.downsample = ConvDownsample(in_ch, out_ch=n_filters)
        elif downsample == 'minimal':
            self.downsample = MinimalConvDownsample(in_ch, out_ch=n_filters)
        elif downsample == 'ode':
            self.downsample = ODEDownsample(in_ch, out_ch=n_filters, adjoint=adjoint)

        self.odeblock = ODEBlock(n_filters=n_filters, tol=tol, adjoint=adjoint, t1=t1)
        self.classifier = FCClassifier(in_ch=n_filters)

    def forward(self, x):
        x = self.downsample(x)
        x = self.odeblock(x)
        x = self.classifier(x)
        return x
    
    def features(self):  # ugly hack
        # remove last classification layer
        self.classifier.module[-1] = nn.Sequential()

    def nfe(self, reset=False):
        nfe = self.odeblock.nfe
        if reset:
            self.odeblock.nfe = 0
        return nfe


class ResNet(nn.Module):
    def __init__(self, in_ch, n_filters=64, downsample='residual'):
        super(ResNet, self).__init__()

        if downsample == 'residual':
            self.downsample = ResDownsample(in_ch, out_ch=n_filters)
        elif downsample == 'convolution':
            self.downsample = ConvDownsample(in_ch, out_ch=n_filters)
        elif downsample == 'minimal':
            self.downsample = MinimalConvDownsample(in_ch, out_ch=n_filters)

        self.features = nn.Sequential(*[ResBlock(n_filters, n_filters) for _ in range(6)])
        self.classifier = FCClassifier(n_filters)

    def forward(self, x):
        x = self.downsample(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def nfe(self, reset=False):
        return 0


"""
   Initial Downsample Blocks
"""


class MinimalConvDownsample(nn.Module):

    def __init__(self, in_ch, out_ch=64):
        super(MinimalConvDownsample, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_ch, 24, 3, 1),
            norm(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 4, 2, 1),
            norm(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, out_ch, 4, 2, 1)
        )

    def forward(self, *input):
        return self.module(*input)


class ConvDownsample(nn.Module):

    def __init__(self, in_ch, out_ch=64):
        super(ConvDownsample, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, 4, 2, 1)
        )

    def forward(self, *input):
        return self.module(*input)


class ResDownsample(nn.Module):

    def __init__(self, in_ch, out_ch=64):
        super(ResDownsample, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, out_ch, stride=2, downsample=conv1x1(64, out_ch, 2)),
        )

    def forward(self, *input):
        return self.module(*input)


class ODEDownsample(nn.Module):

    def __init__(self, in_ch, out_ch=64, adjoint=False):
        super(ODEDownsample, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 4, 2, 1)  # first downsample
        self.odeblock = ODEBlock(n_filters=out_ch, adjoint=adjoint)
        self.maxpool = nn.MaxPool2d(4, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.odeblock(x)
        x = self.maxpool(x)
        return x

"""
    Final FC Module
"""


class FCClassifier(nn.Module):

    def __init__(self, in_ch=64):
        super(FCClassifier, self).__init__()
        self.module = nn.Sequential(
            norm(in_ch),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # global average pooling
            Flatten(),
            nn.Linear(in_ch, 10)
        )

    def forward(self, *input):
        return self.module(*input)


"""
    Helper Modules
"""


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, transpose=False, **kwargs):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(dim_in + 1, dim_out, **kwargs)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, n_filters=64, tol=1e-3, adjoint=False, t1=1):
        super(ODEBlock, self).__init__()
        self.odefunc = ODEfunc(n_filters)
        self.integration_time = torch.tensor([0, t1]).float()
        self.tol = tol
        self.odeint = odeint_adjoint if adjoint else odeint

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = self.odeint(self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol)
        return out[1]  # dynamics @ t=t1

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

    @property
    def t1(self):
        return self.integration_time[1]

    @t1.setter
    def t1(self, value):
        self.integration_time = torch.tensor([0, value]).float()


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


if __name__ == '__main__':
    net = ODENet()
    print(net)
