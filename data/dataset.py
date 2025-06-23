"""LLFF dataset utilities.

This module provides simple helpers to load LLFF style datasets and
wrap them in a :class:`torch.utils.data.Dataset` for training.
"""

from __future__ import annotations

import os
from typing import Tuple, List

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class SimpleDataset(Dataset):
    """Wrap images and poses for ``DataLoader``."""

    def __init__(self, images: np.ndarray, poses: np.ndarray, hwf: Tuple[int, int, float]):
        # Store the images as ``float32`` tensors scaled to ``[0, 1]`` so they
        # can be fed directly into the network.
        self.images = torch.from_numpy(images).float() / 255.0

        # Camera extrinsics for each image. ``poses`` is expected to have shape
        # ``(N, 3, 4)``.
        self.poses = torch.from_numpy(poses).float()

        # Height, width and focal length are kept separately for convenience.
        self.h, self.w, self.focal = hwf

    def __len__(self) -> int:  # pragma: no cover - trivial
        # The dataset length is simply the number of loaded images.
        return self.images.shape[0]

    def __getitem__(self, idx: int):  # pragma: no cover - trivial
        # ``DataLoader`` calls this to retrieve a training sample. We return
        # the image tensor together with its corresponding camera pose.
        return self.images[idx], self.poses[idx]


def _find_images(image_dir: str) -> List[str]:
    """Return absolute paths to all image files in ``image_dir``."""

    # Accepted file extensions. The dataset typically contains PNG or JPEG
    # files but we keep the check general.
    exts = {"png", "jpg", "jpeg"}

    # Filter directory contents to only keep image files.
    files = [
        f for f in sorted(os.listdir(image_dir)) if f.split(".")[-1].lower() in exts
    ]

    # Join each filename with ``image_dir`` to obtain an absolute path.
    return [os.path.join(image_dir, f) for f in files]


def load_llff_data(basedir: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, float], float, float]:
    """Load images and camera data from an LLFF dataset.

    Parameters
    ----------
    basedir: str
        Directory containing ``poses_bounds.npy`` and an ``images`` folder.

    Returns
    -------
    images: np.ndarray
        Array of shape ``(N, H, W, 3)``.
    poses: np.ndarray
        Array of shape ``(N, 3, 4)`` containing camera extrinsics.
    hwf: Tuple[int, int, float]
        ``(height, width, focal)`` extracted from ``poses_bounds.npy``.
    near, far: float
        Near and far bounds suggested by the dataset.
    """

    # ``poses_bounds.npy`` stores both camera extrinsics and the recommended
    # near/far bounds.  We reshape the array into ``(N, 3, 5)`` where each row
    # contains a 3x5 matrix [R|t|hwf].
    poses_bounds = np.load(os.path.join(basedir, "poses_bounds.npy"))
    poses_arr = poses_bounds[:, :-2].reshape([-1, 3, 5])
    bds = poses_bounds[:, -2:]

    # Height, width and focal length are taken from the first pose matrix.
    hwf = poses_arr[0, :, 4].astype(np.float32)

    # Extract the camera matrices (3x4). LLFF uses a different coordinate
    # system than PyTorch3D, so we re-arrange the axes to match the NeRF
    # convention.
    poses = poses_arr[..., :4].astype(np.float32)
    poses = np.stack(
        [poses[:, :, 1], -poses[:, :, 0], poses[:, :, 2], poses[:, :, 3]],
        axis=-1,
    )

    img_dir = os.path.join(basedir, "images")
    if not os.path.exists(img_dir):
        # Some datasets ship only downsampled image folders (e.g. ``images_4``).
        for name in sorted(os.listdir(basedir)):
            if name.startswith("images_"):
                img_dir = os.path.join(basedir, name)
                break

    # Read all images into a single ``(N, H, W, 3)`` array.
    img_files = _find_images(img_dir)
    images = np.stack([imageio.v2.imread(f) for f in img_files], axis=0)

    # Provide slightly tighter near/far bounds to avoid numerical issues.
    near = float(bds.min() * 0.9)
    far = float(bds.max() * 1.0)

    return images, poses, tuple(hwf), near, far


def downsample_data(images: np.ndarray, hwf: Tuple[int, int, float], factor: int) -> Tuple[np.ndarray, Tuple[int, int, float]]:
    """Downsample images and adjust ``hwf`` accordingly."""

    # Early exit if no downsampling is requested.
    if factor is None or factor <= 1:
        return images, hwf

    # Convert ``(N, H, W, C)`` array to ``(N, C, H, W)`` tensor for interpolation.
    img_t = torch.from_numpy(images).permute(0, 3, 1, 2).float()

    # Compute new spatial resolution using integer division.
    h = int(img_t.shape[2] / factor)
    w = int(img_t.shape[3] / factor)

    # ``mode='area'`` performs average pooling which is suited for downsampling
    # photographic images.
    img_t = F.interpolate(img_t, size=(h, w), mode="area")

    # Convert back to NumPy with original dtype in ``(N, H, W, C)`` order.
    img_t = img_t.permute(0, 2, 3, 1).cpu().numpy().astype(images.dtype)

    # Adjust focal length along with height and width.
    new_hwf = (h, w, hwf[2] / factor)
    return img_t, new_hwf

