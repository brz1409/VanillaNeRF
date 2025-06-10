import os
from typing import Tuple
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset


def load_blender_data(basedir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load images and poses from a Blender synthetic dataset directory."""
    images = imageio.v2.imread(os.path.join(basedir, 'images', 'train', '000.png'))
    raise NotImplementedError('Dataset loading should be implemented for your data')


class SimpleDataset(Dataset):
    def __init__(self, images: np.ndarray, poses: np.ndarray, hwf: Tuple[int, int, float]):
        self.images = torch.from_numpy(images).float() / 255.0
        self.poses = torch.from_numpy(poses).float()
        self.h, self.w, self.focal = hwf

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.poses[idx]
