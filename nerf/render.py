"""Utilities for volumetric rendering of NeRF models."""

import torch
import torch.nn.functional as F
from typing import Callable


def render_rays(
    network_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    num_samples: int,
    rand: bool = True,
) -> torch.Tensor:
    """Render colors for a batch of rays.

    Parameters
    ----------
    network_fn : callable
        Function that maps 3D points and view directions to ``(rgb, sigma)``
        predictions.
    rays_o : torch.Tensor
        Origins of the rays, shape ``(N_rays, 3)``.
    rays_d : torch.Tensor
        Directions of the rays, shape ``(N_rays, 3)``.
    near, far : float
        Bounds of sampling along each ray.
    num_samples : int
        Number of sample points per ray.
    rand : bool, optional
        If ``True``, apply stratified sampling for anti-aliasing.

    Returns
    -------
    torch.Tensor
        Concatenated RGB color, depth and accumulated opacity with shape
        ``(N_rays, 5)``.
    """

    device = rays_o.device

    # ``t_vals`` are uniformly spaced sample positions between 0 and 1 that will
    # later be scaled to the ``[near, far]`` range.
    t_vals = torch.linspace(0.0, 1.0, steps=num_samples, device=device)
    if rand:
        # Stratified sampling in depth to reduce aliasing artifacts. We jitter
        # each bin by a random amount within its interval.
        mids = 0.5 * (t_vals[:-1] + t_vals[1:])
        upper = torch.cat([mids, t_vals[-1:]], -1)
        lower = torch.cat([t_vals[:1], mids], -1)
        t_vals = lower + (upper - lower) * torch.rand_like(t_vals)

    # Sample 3D points uniformly between near and far bounds
    # Convert normalized ``t_vals`` to actual depth values.
    z_vals = near * (1.0 - t_vals) + far * t_vals

    # Compute the 3D sample positions along each ray.
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    # Viewing directions are constant along a ray but we expand them to match
    # the sampled points so we can feed them to the network.
    dirs = rays_d[..., None, :].expand_as(pts)

    # Query the NeRF network
    pts_flat = pts.reshape(-1, 3)
    dirs_flat = dirs.reshape(-1, 3)
    raw = network_fn(pts_flat, dirs_flat)
    raw = raw.view(*pts.shape[:-1], 4)

    # ``network_fn`` is expected to return RGB values in ``[0, 1]`` and
    # non-negative densities. Avoid applying activations a second time.
    rgb = raw[..., :3]
    sigma = raw[..., 3]
    delta = z_vals[..., 1:] - z_vals[..., :-1]
    delta = torch.cat([delta, 1e10 * torch.ones_like(delta[..., :1])], -1)
    alpha = 1.0 - torch.exp(-sigma * delta)

    # Accumulate colors along each ray using alpha compositing. ``weights``
    # correspond to the contribution of each sample to the final pixel color.
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], -1), -1
    )[..., :-1]

    # Weighted sums yield the rendered RGB values, depth and total opacity.
    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * z_vals).sum(dim=-1)
    acc_map = weights.sum(dim=-1)

    # Return a tensor of shape ``(N_rays, 5)`` containing RGB color, depth and
    # accumulated opacity for each input ray.
    return torch.cat([rgb_map, depth_map[..., None], acc_map[..., None]], -1)
