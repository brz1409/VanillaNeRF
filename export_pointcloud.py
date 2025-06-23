#!/usr/bin/env python3
"""Export a point cloud from a trained Vanilla NeRF model.

This script samples rays from the training cameras, keeps the points with
high density contribution and writes them to a binary PLY file.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

import numpy as np
import torch

from data.dataset import load_llff_data
from nerf.model import NeRF, PositionalEncoding


device = torch.device("cpu")


def get_rays(H: int, W: int, focal: float, c2w: torch.Tensor):
    """Generate ray origins and directions for all pixels of an image."""
    i, j = torch.meshgrid(
        torch.arange(W, device=c2w.device),
        torch.arange(H, device=c2w.device),
        indexing="xy",
    )
    dirs = torch.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1
    )
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a binary PLY file with vertex colors."""
    verts = np.empty(
        len(xyz),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    verts["x"], verts["y"], verts["z"] = xyz.T.astype(np.float32)
    cols = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    verts["red"], verts["green"], verts["blue"] = cols.T

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(verts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    with open(path, "wb") as f:
        f.write(header)
        verts.tofile(f)


def render_samples(network, rays_o, rays_d, near, far, num_samples):
    """Evaluate the network along rays and return RGB, weights and points."""
    device = rays_o.device
    t_vals = torch.linspace(0.0, 1.0, steps=num_samples, device=device)
    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals[None, :].expand(rays_o.shape[0], num_samples)

    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]
    dirs = rays_d[:, None, :].expand_as(pts)

    pts_flat = pts.reshape(-1, 3)
    dirs_flat = dirs.reshape(-1, 3)
    raw = network(pts_flat, dirs_flat).view(rays_o.shape[0], num_samples, 4)
    rgb = raw[..., :3]
    sigma = raw[..., 3]

    delta = z_vals[:, 1:] - z_vals[:, :-1]
    delta = torch.cat([delta, 1e10 * torch.ones_like(delta[:, :1])], dim=1)
    alpha = 1.0 - torch.exp(-sigma * delta)
    trans = torch.cumprod(
        torch.cat([torch.ones((alpha.size(0), 1), device=device), 1.0 - alpha + 1e-10], dim=1),
        dim=1,
    )[:, :-1]
    weights = alpha * trans
    return rgb, weights, pts


@torch.no_grad()
def extract_pointcloud(
    network,
    poses: np.ndarray,
    hwf,
    near: float,
    far: float,
    *,
    n_rays: int,
    num_samples: int,
    weight_threshold: float,
    cam_ids: Optional[Sequence[int]],
):
    H, W, focal = [int(x) for x in hwf]
    if cam_ids is not None:
        poses = poses[cam_ids]
    rays_per_image = int(np.ceil(n_rays / poses.shape[0]))

    logger.info("Sampling %d rays per image from %d poses", rays_per_image, poses.shape[0])

    all_xyz = []
    all_rgb = []

    for idx, c2w in enumerate(poses):
        logger.info("Processing camera %d/%d", idx + 1, poses.shape[0])
        c2w = torch.from_numpy(c2w).to(device)
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        sel = torch.randperm(rays_o.shape[0], device=device)[:rays_per_image]
        rays_o = rays_o[sel]
        rays_d = rays_d[sel]

        rgb, weights, pts = render_samples(network, rays_o, rays_d, near, far, num_samples)
        rgb = rgb.reshape(-1, 3)
        w = weights.reshape(-1)
        pts = pts.reshape(-1, 3)

        mask = w > weight_threshold
        if mask.any():
            all_xyz.append(pts[mask].cpu().numpy())
            all_rgb.append(rgb[mask].cpu().numpy())
        logger.info(
            "  Kept %d/%d points from this view",
            mask.sum().item(),
            mask.numel(),
        )

    if not all_xyz:
        return np.zeros((0, 3)), np.zeros((0, 3))

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    logger.info("Total points collected: %d", xyz.shape[0])
    return xyz, rgb


def main(args):
    global device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    logger.info("Loading data from %s", args.data_dir)
    images, poses, hwf, near, far = load_llff_data(
        args.data_dir, downsample=args.downsample, save_downsampled=True
    )

    logger.info("Loading model checkpoint %s", args.checkpoint)
    model = NeRF()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()
    logger.info("Model loaded. Extracting point cloud ...")

    pos_enc = PositionalEncoding(10).to(device)
    dir_enc = PositionalEncoding(4).to(device)

    def network(pts, dirs):
        return model(pos_enc(pts), dir_enc(dirs))

    xyz, rgb = extract_pointcloud(
        network,
        poses,
        hwf,
        near,
        far,
        n_rays=args.n_rays,
        num_samples=args.num_samples,
        weight_threshold=args.weight_threshold,
        cam_ids=args.cam_ids,
    )
    logger.info("Writing %d points to %s", xyz.shape[0], args.output)
    write_ply(Path(args.output), xyz, rgb)
    logger.info("Point cloud export finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a colored point cloud from a trained NeRF model."
    )
    parser.add_argument("--data_dir", required=True, help="Path to dataset")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--output", default="pointcloud.ply", help="Output PLY file")
    parser.add_argument("--downsample", type=int, default=4, help="Image downsampling factor")
    parser.add_argument("--num_samples", type=int, default=64, help="Samples per ray")
    parser.add_argument("--n_rays", type=int, default=500000, help="Total rays to sample")
    parser.add_argument("--weight_threshold", type=float, default=0.1, help="Keep points with weight > threshold")
    parser.add_argument("--cam_ids", type=int, nargs="*", default=None, help="Optional list of camera indices")
    parser.add_argument("--device", default=None, help="Torch device to use (e.g. 'cuda' or 'cpu')")
    main(parser.parse_args())
