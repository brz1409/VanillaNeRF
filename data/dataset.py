"""Dataset utilities for loading Blender-style scenes used in NeRF experiments."""

import json
import os
from typing import Tuple

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset


def load_blender_data(basedir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load images and camera poses from a Blender synthetic dataset.

    The returned arrays are NumPy tensors ready to be wrapped in a
    :class:`torch.utils.data.Dataset`.

    Parameters
    ----------
    basedir : str
        Path to the dataset directory containing the ``transforms_*.json`` files.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        ``images``: array of shape ``(N, H, W, 3)`` with uint8 values.
        ``poses``: array of shape ``(N, 4, 4)`` containing camera extrinsics.
        ``(H, W, focal)``: height, width and focal length of the images.
    """

    # NeRF's Blender datasets store camera information inside JSON files.
    # We default to the training split (``transforms_train.json``) but also
    # support a single ``transforms.json`` file when experimenting.
    json_path = os.path.join(basedir, "transforms_train.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(basedir, "transforms.json")
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    all_images = []
    all_poses = []
    for frame in meta["frames"]:
        # Each frame contains a path to the corresponding image and a
        # 4x4 camera-to-world transformation matrix.
        fpath = os.path.join(basedir, f"{frame['file_path']}.png")
        img = imageio.v2.imread(fpath)
        all_images.append(img)
        all_poses.append(np.array(frame["transform_matrix"], dtype=np.float32))

    images = np.stack(all_images, axis=0)
    poses = np.stack(all_poses, axis=0)

    # The JSON file specifies the camera field of view in radians.  We
    # convert this to a focal length for convenience.
    h, w = images[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    return images, poses, (h, w, focal)


class SimpleDataset(Dataset):
    """Wrap pre-loaded image and pose arrays for use with ``DataLoader``."""

    def __init__(self, images: np.ndarray, poses: np.ndarray, hwf: Tuple[int, int, float]):
        # Store images as floating point tensors normalized to [0, 1]
        self.images = torch.from_numpy(images).float() / 255.0
        # Camera-to-world transformation matrices for each image
        self.poses = torch.from_numpy(poses).float()
        # Height, width and focal length are needed when generating rays
        self.h, self.w, self.focal = hwf

    def __len__(self):
        # Number of images (and poses) in the dataset
        return self.images.shape[0]

    def __getitem__(self, idx):
        # Return image tensor and corresponding camera pose
        return self.images[idx], self.poses[idx]
