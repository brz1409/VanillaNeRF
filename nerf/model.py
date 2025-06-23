"""Implementation of the neural network architecture used for NeRF."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name: str):
    """Map a string to a PyTorch activation module."""

    # ``ReLU`` is used by the original NeRF implementation and generally works
    # well for most tasks.
    if name == "relu":
        return nn.ReLU(True)

    # ``Softplus`` can be used as a smooth alternative which guarantees
    # positive outputs.
    if name == "softplus":
        return nn.Softplus()

    # Provide a clear error if the caller passes an unknown name.
    raise ValueError(f"Unknown activation {name}")


class PositionalEncoding(nn.Module):
    """Embed coordinates using sinusoidal functions at multiple frequencies."""

    def __init__(self, num_freqs: int, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input

        # Create a vector of frequency bands. Higher frequencies allow the
        # network to capture finer detail. ``register_buffer`` ensures that the
        # tensor is moved to the correct device together with the module.
        self.register_buffer("freq_bands", 2.0 ** torch.arange(num_freqs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` with sine and cosine functions."""

        out = []
        if self.include_input:
            # Optionally include the raw coordinates in the embedding.
            out.append(x)
        for freq in self.freq_bands:
            # For each frequency add sine and cosine components.
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)


class NeRF(nn.Module):
    """Minimal multi-layer perceptron used to parameterize a NeRF."""

    def __init__(self, depth: int = 8, width: int = 256, skips=(4,),
                 input_ch: int = 3 * 10 * 2 + 3, input_ch_dir: int = 3 * 4 * 2 + 3,
                 activation: str = 'relu'):
        super().__init__()

        # Network architecture parameters. ``depth`` controls the number of
        # layers while ``width`` sets the hidden feature size. ``skips``
        # specifies at which layers the input is concatenated to the
        # activations (skip connections).
        self.depth = depth
        self.width = width
        self.skips = skips

        # Layers that process 3D point coordinates. Skip connections are
        # inserted after certain layers as in the original paper.
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, width)] +
            [nn.Linear(width + (input_ch if i in skips else 0), width) for i in range(depth - 1)]
        )
        self.feature_linear = nn.Linear(width, width)
        self.alpha_linear = nn.Linear(width, 1)

        # Viewing direction is processed together with point features to
        # predict RGB values.
        self.dir_linear = nn.Linear(input_ch_dir + width, width // 2)
        self.rgb_linear = nn.Linear(width // 2, 3)

        # Activation function used throughout the network.
        self.act = get_activation(activation)

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Evaluate the network at 3D locations ``x`` and viewing directions ``d``."""

        # ``x`` comes in already embedded. We progressively transform it with
        # fully connected layers. Skip connections insert the original input
        # halfway through the network which helps with learning high-frequency
        # details.
        h = x
        for i, layer in enumerate(self.pts_linears):
            h = self.act(layer(h))
            if i in self.skips:
                h = torch.cat([x, h], -1)

        # From the final hidden representation predict density (alpha) and a
        # feature vector that will later be combined with view direction.
        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)

        # Combine learned features with viewing direction before predicting color
        h = torch.cat([feature, d], -1)
        h = self.act(self.dir_linear(h))
        rgb = torch.sigmoid(self.rgb_linear(h))

        # Concatenate color and density. ``relu`` ensures densities are
        # non-negative.
        return torch.cat([rgb, F.relu(alpha)], -1)
