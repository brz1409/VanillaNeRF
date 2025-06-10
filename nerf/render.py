import torch
import torch.nn.functional as F
from typing import Callable


def render_rays(network_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                rays_o: torch.Tensor,
                rays_d: torch.Tensor,
                near: float,
                far: float,
                num_samples: int,
                rand: bool = True) -> torch.Tensor:
    """Render pixels by marching rays and querying network."""
    device = rays_o.device
    t_vals = torch.linspace(0., 1., steps=num_samples, device=device)
    if rand:
        mids = .5 * (t_vals[:-1] + t_vals[1:])
        upper = torch.cat([mids, t_vals[-1:]], -1)
        lower = torch.cat([t_vals[:1], mids], -1)
        t_vals = lower + (upper - lower) * torch.rand_like(t_vals)

    z_vals = near * (1. - t_vals) + far * t_vals
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    dirs = rays_d[..., None, :].expand_as(pts)
    pts_flat = pts.reshape(-1, 3)
    dirs_flat = dirs.reshape(-1, 3)
    raw = network_fn(pts_flat, dirs_flat)
    raw = raw.view(*pts.shape[:-1], 4)

    rgb = torch.sigmoid(raw[..., :3])
    alpha = 1. - torch.exp(-F.relu(raw[..., 3]))

    weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], -1), -1)[..., :-1]
    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * z_vals).sum(dim=-1)
    acc_map = weights.sum(dim=-1)

    return torch.cat([rgb_map, depth_map[..., None], acc_map[..., None]], -1)
