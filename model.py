import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint


class ODENet(nn.Module):
    def __init__(self, in_ch, out=10, n_filters=64, downsample='residual', method='dopri5', tol=1e-3, adjoint=False, t1=1, dropout=0, norm='group'):
        super(ODENet, self).__init__()
        
        common = dict(out_ch=n_filters, norm=norm)
        if downsample == 'residual':
            self.downsample = ResDownsample(in_ch, **common)
        elif downsample == 'convolution':
            self.downsample = ConvDownsample(in_ch, **common)
        elif downsample == 'minimal':
            self.downsample = MinimalConvDownsample(in_ch, **common)
        elif downsample == 'one-shot':
            self.downsample = OneShotDownsample(in_ch, **common)
        elif downsample == 'ode':
            self.downsample = ODEDownsample(in_ch, adjoint=adjoint, t1=t1, tol=tol, method=method, **common)
        elif downsample == 'ode2':
            self.downsample = ODEDownsample2(in_ch, adjoint=adjoint, t1=t1, tol=tol, method=method, **common)

        self.odeblock = ODEBlock(n_filters=n_filters, tol=tol, adjoint=adjoint, t1=t1, method=method, norm=norm)
        self.classifier = FCClassifier(in_ch=n_filters, out=out, dropout=dropout, norm=norm)

    def forward(self, x):
        out = []
        x = self.downsample(x)
        if isinstance(x, (tuple, list)):
            f, x = x  # first elements are features, second is output to continue the forward
            if isinstance(self.classifier.module[-1], nn.Sequential):  # no classification to be performed, apply GAP and return
                f = torch.stack([fi.mean(-1).mean(-1) for fi in f])  # global avg pooling
            else:  # we want to apply the classifier to all features
                f = torch.stack([self.classifier(fi) for fi in f])
            out.append(f)

        x = self.odeblock(x)
        if x.dim() > 4:
            x = torch.stack([self.classifier(xi) for xi in x])
        else:
            x = self.classifier(x)

        out.append(x)
        out = torch.cat(out)
        return out

    def to_features_extractor(self, keep_pool=True):  # ugly hack
        if isinstance(self.downsample, (ODEDownsample, ODEDownsample2)):
            self.downsample.odeblock.return_last_only = False
        self.odeblock.return_last_only = False  # returns dynamic @ multiple timestamps
        if keep_pool:
            # remove last classification layer but maintain norm, relu and global avg pooling
            self.classifier.module[-1] = nn.Sequential()
        else:
            self.classifier = nn.Sequential(*list(self.classifier.module.children())[:2])

    def nfe(self, reset=False):
        nfe = self.odeblock.nfe
        if reset:
            self.odeblock.nfe = 0
        return nfe


class ResNet(nn.Module):
    def __init__(self, in_ch, out=10, n_filters=64, downsample='residual', dropout=0, norm='group'):
        super(ResNet, self).__init__()

        common = dict(out_ch=n_filters, norm=norm)
        if downsample == 'residual':
            self.downsample = ResDownsample(in_ch, **common)
        elif downsample == 'convolution':
            self.downsample = ConvDownsample(in_ch, **common)
        elif downsample == 'minimal':
            self.downsample = MinimalConvDownsample(in_ch, **common)
        elif downsample == 'one-shot':
            self.downsample = OneShotDownsample(in_ch, **common)

        self.features = nn.Sequential(*[ResBlock(n_filters, n_filters) for _ in range(6)])
        self.classifier = FCClassifier(n_filters, out=out, dropout=dropout, norm=norm)
        self._extract_features = False

    def to_features_extractor(self, keep_pool=True):  # ugly hack
        if keep_pool:
            # remove last classification layer but maintain norm, relu and global avg pooling
            self.classifier.module[-1] = nn.Sequential()
        else:
            self.classifier = nn.Sequential(*list(self.classifier.module.children())[:2])
        self._extract_features = True
        self._tmp_features = [None, ] * (len(self.features) + 1)  # we keep also the first input
        
        def hooks(idx):
            def __hook(m, i, o):
                self._tmp_features[idx] = o.data
            return __hook
        
        self.downsample.register_forward_hook(hooks(0))
        for n, block in enumerate(self.features):
            block.register_forward_hook(hooks(n + 1))

    def forward(self, x):
        x = self.downsample(x)
        x = self.features(x)
        if self._extract_features:
            x = torch.stack([self.classifier(xi) for xi in self._tmp_features])
        else:
            x = self.classifier(x)
        return x

    def nfe(self, reset=False):
        return 0


"""
   Initial Downsample Blocks
"""


class OneShotDownsample(nn.Module):

    def __init__(self, in_ch, out_ch=64, **kwargs):
        super(OneShotDownsample, self).__init__()
        self.module = nn.Conv2d(in_ch, out_ch, 4, 2, 1)

    def forward(self, *input):
        return self.module(*input)


class MinimalConvDownsample(nn.Module):

    def __init__(self, in_ch, out_ch=64, norm='group'):
        super(MinimalConvDownsample, self).__init__()
        norm = normalization(norm)
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

    def __init__(self, in_ch, out_ch=64, norm='group'):
        super(ConvDownsample, self).__init__()
        norm = normalization(norm)
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

    def __init__(self, in_ch, out_ch=64, norm='group'):
        super(ResDownsample, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2), norm=norm),
            ResBlock(64, out_ch, stride=2, downsample=conv1x1(64, out_ch, 2), norm=norm),
        )

    def forward(self, *input):
        return self.module(*input)


class ODEDownsample(nn.Module):

    def __init__(self, in_ch, out_ch=64, method='dopri5', adjoint=False, t1=1, tol=1e-3, norm='group'):
        super(ODEDownsample, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 4, 2, 1)  # first downsample
        self.odeblock = ODEBlock(n_filters=out_ch, adjoint=adjoint, t1=t1, tol=tol, method=method, norm=norm)
        self.maxpool = nn.MaxPool2d(4, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.odeblock(x)
        if x.dim() > 4:
            return x, self.maxpool(x[-1])

        x = self.maxpool(x)
        return x


class ODEDownsample2(nn.Module):

    def __init__(self, in_ch, out_ch=64, method='dopri5', adjoint=False, t1=1, tol=1e-3, norm='group'):
        super(ODEDownsample2, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 4, 2, 1)  # first downsample
        self.odeblock = ODEBlock(n_filters=out_ch, adjoint=adjoint, t1=t1, tol=tol, method=method, norm=norm)
        self.norm = nn.Sequential(normalization(norm)(out_ch), nn.ReLU(inplace=True))
        self.conv2 = nn.Conv2d(out_ch, out_ch, 4, 2, 1)  # downsample for successive ode
        
        self.apply_conv = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.odeblock(x)
        if x.dim() > 4:
            x = torch.stack([self.norm(xi) for xi in x])
            if self.apply_conv:
                x = torch.stack([self.conv2(xi) for xi in x])
                return x, x[-1]
            # otherwise apply conv2 only at the last
            return x, self.conv2(x[-1])

        x = self.norm(x)
        x = self.conv2(x)
        return x


"""
    Final FC Module
"""


class FCClassifier(nn.Module):

    def __init__(self, in_ch=64, out=10, dropout=0, norm='group'):
        super(FCClassifier, self).__init__()
        norm = normalization(norm)
        layers = [
                     norm(in_ch),
                     nn.ReLU(inplace=True),
                     nn.AdaptiveAvgPool2d((1, 1)),  # global average pooling
                 ] + (
                     [nn.Dropout(dropout), ] if dropout else []
                 ) + [
                     Flatten(),
                     nn.Linear(in_ch, out)
                 ]

        self.module = nn.Sequential(*layers)

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


def normalization(norm='group'):

    def _group_norm(dim):
        return nn.GroupNorm(min(32, dim), dim)
        
    def _batch_norm(dim):
        return nn.BatchNorm2d(dim, track_running_stats=False)

    if norm == 'group':
        return _group_norm
    elif norm == 'batch':
        return _batch_norm
        
    raise NotImplementedError('Normalization layer not implemented: {}'.format(norm))


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm='group'):
        super(ResBlock, self).__init__()
        norm = normalization(norm)
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

    def __init__(self, dim, norm='group'):
        super(ODEfunc, self).__init__()
        norm = normalization(norm)
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

    def __init__(self, n_filters=64, tol=1e-3, method='dopri5', adjoint=False, t1=1, norm='group'):
        super(ODEBlock, self).__init__()
        self.odefunc = ODEfunc(n_filters, norm=norm)
        self.t1 = t1
        self.tol = tol
        self.method = method
        self.odeint = odeint_adjoint if adjoint else odeint
        self.return_last_only = True

    def forward(self, x):
        if self.integration_time is None:
            return x
    
        self.integration_time = self.integration_time.type_as(x)
        out = self.odeint(self.odefunc, x, self.integration_time, method=self.method, rtol=self.tol, atol=self.tol)
        if self.return_last_only:
            out = out[-1]  # dynamics @ t=t1
        return out

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
        if isinstance(value, (int, float)):
            if value == 0:
                self.integration_time = None
            else:
                self.integration_time = torch.tensor([0, value], dtype=torch.float32)

        elif isinstance(value, (list, tuple, torch.Tensor)):
            if isinstance(value, tuple):
                value = list(value)
            if isinstance(value, torch.Tensor):
                value = value.tolist()
            if value[0] != 0:
                print(value[0])
                value = [0, ] + value

            self.integration_time = torch.tensor(value, dtype=torch.float32)
        else:
            raise ValueError('Argument must be a scalar, a list, or a tensor')


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


if __name__ == '__main__':
    net = ODENet(3, downsample='ode', t1=[.1, .2, .3, 1]).to('cuda')
    net.to_features_extractor()
    # print(net)
    a = torch.rand(7, 3, 32, 32).to('cuda')
    print(net(a).shape)
