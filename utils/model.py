import torch
import torch.nn as nn
import torch.nn.functional as F


def softplus(x):
    return torch.log(torch.exp(x) + 1)


class CSoftplus(nn.Module):
    def forward(self, x):
        return softplus(x)


def get_nonlinear_func(nonlinearity_type='silu'):
    """MLP와 ConvBlock 양쪽에서 쓸 수 있는 activation module을 반환한다."""
    if nonlinearity_type == 'relu':
        return nn.ReLU(inplace=True)
    elif nonlinearity_type == 'elu':
        return nn.ELU(inplace=True)
    elif nonlinearity_type == 'tanh':
        return nn.Tanh()
    elif nonlinearity_type == 'softplus':
        return nn.Softplus()
    elif nonlinearity_type == 'csoftplus':
        return CSoftplus()
    elif nonlinearity_type == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif nonlinearity_type == "silu" or nonlinearity_type == "swish":
        return nn.SiLU(inplace=True)
    else:
        raise NotImplementedError
    
def get_num_groups(num_channels, max_groups=8):
    """
    GroupNorm에서 num_channels를 나누어떨어지게 하는 group 수를 고른다.
    예: channels=64 -> groups=8
        channels=12 -> groups=6 or 4 etc.
    """
    for groups in reversed(range(1, max_groups + 1)):
        if num_channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinearity="silu", use_norm=True):
        super().__init__()

        act1 = get_nonlinear_func(nonlinearity)
        act2 = get_nonlinear_func(nonlinearity)

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        ]

        if use_norm:
            layers.append(nn.GroupNorm(get_num_groups(out_channels), out_channels))

        layers.append(act1)

        layers.append(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if use_norm:
            layers.append(nn.GroupNorm(get_num_groups(out_channels), out_channels))

        layers.append(act2)

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinearity="silu", use_norm=True):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = ConvBlock(
            in_channels,
            out_channels,
            nonlinearity=nonlinearity,
            use_norm=use_norm,
        )

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, nonlinearity="silu", use_norm=True):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = ConvBlock(
            out_channels + skip_channels,
            out_channels,
            nonlinearity=nonlinearity,
            use_norm=use_norm,
        )

    def forward(self, x, skip):
        x = self.up(x)

        # 입력 이미지 크기가 2의 거듭제곱이 아닐 때도 skip과 크기를 맞추기 위한 padding
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)

        if diff_h != 0 or diff_w != 0:
            x = F.pad(
                x,
                [
                    diff_w // 2,
                    diff_w - diff_w // 2,
                    diff_h // 2,
                    diff_h - diff_h // 2,
                ],
            )

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x
