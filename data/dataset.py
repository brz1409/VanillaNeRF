"""Dataset utilities for loading datasets in Blender and Metashape formats."""

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
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


def _find_param(calib_xml: ET.Element, name: str) -> float:
    """Utility to read calibration parameters from XML."""

    elem = calib_xml.find(name)
    return float(elem.text) if elem is not None else 0.0


def _convert_metashape(xml_file: Path, out_file: Path, image_map: Dict[str, Path]) -> None:
    """Convert ``cameras.xml`` from Metashape to ``transforms.json``.

    This is a lightweight re-implementation of the conversion used by
    ``nerfstudio`` so that the ``nerfstudio`` package is not required.
    Only basic perspective/fisheye/equirectangular cameras are supported.
    """

    tree = ET.parse(xml_file)
    root = tree.getroot()
    chunk = root[0]

    sensors = chunk.find("sensors")
    if sensors is None:
        raise ValueError("No sensors found in cameras.xml")

    calibrated = [s for s in sensors.iter("sensor") if s.get("type") == "spherical" or s.find("calibration")]
    if not calibrated:
        raise ValueError("No calibrated sensors in cameras.xml")

    sensor_type = [s.get("type") for s in calibrated]
    if sensor_type.count(sensor_type[0]) != len(sensor_type):
        raise ValueError("Mixed sensor types are not supported")

    if sensor_type[0] == "frame":
        cam_model = "OPENCV"
    elif sensor_type[0] == "fisheye":
        cam_model = "OPENCV_FISHEYE"
    elif sensor_type[0] == "spherical":
        cam_model = "EQUIRECTANGULAR"
    else:
        raise ValueError(f"Unsupported sensor type {sensor_type[0]}")

    sensor_dict = {}
    for sensor in calibrated:
        s = {}
        res = sensor.find("resolution")
        if res is None:
            raise ValueError("Resolution missing in cameras.xml")
        s["w"] = int(res.get("width"))
        s["h"] = int(res.get("height"))

        calib = sensor.find("calibration")
        if calib is None:
            # Spherical cameras have no intrinsics
            s["fl_x"] = s["w"] / 2.0
            s["fl_y"] = s["h"]
            s["cx"] = s["w"] / 2.0
            s["cy"] = s["h"] / 2.0
        else:
            f = calib.find("f")
            if f is None:
                raise ValueError("Focal length not found in calibration")
            s["fl_x"] = s["fl_y"] = float(f.text)
            s["cx"] = _find_param(calib, "cx") + s["w"] / 2.0
            s["cy"] = _find_param(calib, "cy") + s["h"] / 2.0
        sensor_dict[sensor.get("id")] = s

    component_dict = {}
    components = chunk.find("components")
    if components is not None:
        for comp in components.iter("component"):
            t_elem = comp.find("transform")
            if t_elem is None:
                continue
            rot = t_elem.find("rotation")
            if rot is None or rot.text is None:
                R = np.eye(3)
            else:
                R = np.array([float(x) for x in rot.text.split()]).reshape(3, 3)
            trans = t_elem.find("translation")
            if trans is None or trans.text is None:
                T = np.zeros(3)
            else:
                T = np.array([float(x) for x in trans.text.split()])
            scale = t_elem.find("scale")
            s = 1.0 if scale is None or scale.text is None else float(scale.text)
            m = np.eye(4)
            m[:3, :3] = R
            m[:3, 3] = T / s
            component_dict[comp.get("id")] = m

    frames = []
    cameras = chunk.find("cameras")
    if cameras is None:
        raise ValueError("No cameras section in cameras.xml")

    for cam in cameras.iter("camera"):
        frame = {}
        label = cam.get("label")
        if label not in image_map:
            label = label.split(".")[0]
            if label not in image_map:
                # skip images not present
                continue
        frame["file_path"] = image_map[label].name

        sensor_id = cam.get("sensor_id")
        if sensor_id not in sensor_dict:
            continue
        frame.update(sensor_dict[sensor_id])

        t_elem = cam.find("transform")
        if t_elem is None or t_elem.text is None:
            continue
        mat = np.array([float(x) for x in t_elem.text.split()]).reshape(4, 4)

        comp_id = cam.get("component_id")
        if comp_id in component_dict:
            mat = component_dict[comp_id] @ mat

        mat = mat[[2, 0, 1, 3], :]
        mat[:, 1:3] *= -1
        frame["transform_matrix"] = mat.tolist()
        frames.append(frame)

    applied = np.eye(4)[:3, :]
    applied = applied[[2, 0, 1], :]

    data = {
        "camera_model": cam_model,
        "frames": frames,
        "applied_transform": applied.tolist(),
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_metashape_data(basedir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a dataset exported from Agisoft Metashape.

    If ``transforms.json`` is missing this function converts the ``cameras.xml``
    file found in ``basedir`` into that format using an embedded converter
    adapted from the `nerfstudio` project. The JSON file is then loaded in the
    same way as Blender datasets.

    Parameters
    ----------
    basedir : str
        Directory containing the images and ``cameras.xml`` file.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        ``images``: array of shape ``(N, H, W, 3)`` with uint8 values.
        ``poses``: array of shape ``(N, 4, 4)`` containing camera extrinsics.
        ``(H, W, focal)``: height, width and focal length of the images.
    """

    json_path = os.path.join(basedir, "transforms.json")
    if not os.path.exists(json_path):
        xml_path = Path(basedir) / "cameras.xml"
        if not xml_path.exists():
            raise FileNotFoundError(
                "Dataset directory must contain either transforms.json or cameras.xml"
            )

        images = {
            os.path.splitext(f)[0]: Path(basedir) / f
            for f in os.listdir(basedir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        }

        _convert_metashape(xml_path, Path(json_path), images)

    # With the JSON file present we can rely on the regular loader
    return load_blender_data(basedir)

def load_llff_data(basedir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load images and poses from an LLFF dataset using ``poses_bounds.npy``.

    Parameters
    ----------
    basedir : str
        Directory containing ``poses_bounds.npy`` and an ``images`` folder.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        ``images``: array of shape ``(N, H, W, 3)`` with uint8 values.
        ``poses``: array of shape ``(N, 4, 4)`` containing camera extrinsics.
        ``(H, W, focal)``: height, width and focal length of the images.
    """
    poses_bounds = np.load(os.path.join(basedir, "poses_bounds.npy"))
    poses = poses_bounds[:, :-2].reshape([-1, 3, 5])

    img_dir = None
    for d in ["images_4", "images"]:
        cand = os.path.join(basedir, d)
        if os.path.isdir(cand):
            img_dir = cand
            break
    if img_dir is None:
        raise FileNotFoundError("LLFF dataset is missing an images directory")

    img_files = [f for f in sorted(os.listdir(img_dir))
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if len(img_files) == 0:
        raise FileNotFoundError("No images found in LLFF dataset")

    images = [imageio.v2.imread(os.path.join(img_dir, f)) for f in img_files]
    images = np.stack(images, axis=0)
    h, w = images[0].shape[:2]

    poses[:, :2, 4] = np.array([h, w])[:, None]
    poses = np.concatenate([poses[:, 1:2], -poses[:, 0:1], poses[:, 2:]], axis=1)
    poses = poses.astype(np.float32)

    hwf = poses[0, :, 4]
    poses = poses[:, :, :4]
    bottom = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (poses.shape[0], 1)).reshape(poses.shape[0], 1, 4)
    poses = np.concatenate([poses, bottom], axis=1)

    return images, poses, (int(hwf[0]), int(hwf[1]), float(hwf[2]))


def downsample_data(
    images: np.ndarray, hwf: Tuple[int, int, float], factor: int
) -> Tuple[np.ndarray, Tuple[int, int, float]]:
    """Downsample images and adjust focal length by ``factor``.

    Parameters
    ----------
    images : np.ndarray
        Image array of shape ``(N, H, W, 3)``.
    hwf : Tuple[int, int, float]
        ``(H, W, focal)`` describing the original resolution.
    factor : int
        Desired downsampling factor. Values ``<=1`` return the input unchanged.

    Returns
    -------
    Tuple[np.ndarray, Tuple[int, int, float]]
        The downsampled images and updated ``(H, W, focal)`` tuple.
    """

    if factor <= 1:
        return images, hwf

    h, w, focal = hwf
    new_h = max(1, int(h / factor))
    new_w = max(1, int(w / factor))

    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    tensor = F.interpolate(tensor, size=(new_h, new_w), mode="area")
    images_ds = tensor.byte().permute(0, 2, 3, 1).numpy()

    return images_ds, (new_h, new_w, focal / factor)

def load_llff_data(basedir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load images and poses from an LLFF dataset using ``poses_bounds.npy``.

    Parameters
    ----------
    basedir : str
        Directory containing ``poses_bounds.npy`` and an ``images`` folder.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        ``images``: array of shape ``(N, H, W, 3)`` with uint8 values.
        ``poses``: array of shape ``(N, 4, 4)`` containing camera extrinsics.
        ``(H, W, focal)``: height, width and focal length of the images.
    """
    poses_bounds = np.load(os.path.join(basedir, "poses_bounds.npy"))
    poses = poses_bounds[:, :-2].reshape([-1, 3, 5])

    img_dir = None
    for d in ["images_4", "images"]:
        cand = os.path.join(basedir, d)
        if os.path.isdir(cand):
            img_dir = cand
            break
    if img_dir is None:
        raise FileNotFoundError("LLFF dataset is missing an images directory")

    img_files = [f for f in sorted(os.listdir(img_dir))
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if len(img_files) == 0:
        raise FileNotFoundError("No images found in LLFF dataset")

    images = [imageio.v2.imread(os.path.join(img_dir, f)) for f in img_files]
    images = np.stack(images, axis=0)
    h, w = images[0].shape[:2]

    poses[:, :2, 4] = np.array([h, w])[:, None]
    poses = np.concatenate([poses[:, 1:2], -poses[:, 0:1], poses[:, 2:]], axis=1)
    poses = poses.astype(np.float32)

    hwf = poses[0, :, 4]
    poses = poses[:, :, :4]
    bottom = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (poses.shape[0], 1)).reshape(poses.shape[0], 1, 4)
    poses = np.concatenate([poses, bottom], axis=1)

    return images, poses, (int(hwf[0]), int(hwf[1]), float(hwf[2]))
