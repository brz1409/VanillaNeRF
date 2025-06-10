import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name: str):
    if name == 'relu':
        return nn.ReLU(True)
    elif name == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError(f'Unknown activation {name}')


class PositionalEncoding(nn.Module):
    """Applies positional encoding as described in the NeRF paper."""

    def __init__(self, num_freqs: int, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.freq_bands = 2.0 ** torch.arange(num_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        if self.include_input:
            out.append(x)
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)


class NeRF(nn.Module):
    """Minimal NeRF MLP."""

    def __init__(self, depth: int = 8, width: int = 256, skips=(4,),
                 input_ch: int = 3 * 10 * 2 + 3, input_ch_dir: int = 3 * 4 * 2 + 3,
                 activation: str = 'relu'):
        super().__init__()
        self.depth = depth
        self.width = width
        self.skips = skips

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, width)] +
            [nn.Linear(width + (input_ch if i in skips else 0), width) for i in range(depth - 1)]
        )
        self.feature_linear = nn.Linear(width, width)
        self.alpha_linear = nn.Linear(width, 1)

        self.dir_linear = nn.Linear(input_ch_dir + width, width // 2)
        self.rgb_linear = nn.Linear(width // 2, 3)

        self.act = get_activation(activation)

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.act(l(h))
            if i in self.skips:
                h = torch.cat([x, h], -1)
        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)

        h = torch.cat([feature, d], -1)
        h = self.act(self.dir_linear(h))
        rgb = torch.sigmoid(self.rgb_linear(h))
        return torch.cat([rgb, F.relu(alpha)], -1)
