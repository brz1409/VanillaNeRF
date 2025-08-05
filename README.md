# Vanilla NeRF

![NeRF Batch Size Comparison](nerf_simulation_parallel.gif)

- **Left**: Batch Size 1  
- **Middle**: Batch Size 10  
- **Right**: Batch Size 20  

This repository implements a compact version of **Neural Radiance Fields** (NeRF) in PyTorch.  It is intended as an educational work that exposes all parts of a basic NeRF setup with clear code and extensive comments.

The project covers the full pipeline from loading an LLFF dataset with `poses_bounds.npy` to training a network that synthesises new views of the scene.

## Installation

Create a virtual environment (optional) and install the Python dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The code only depends on PyTorch and a few utility libraries.  GPU support is recommended but not required. If TensorBoard fails to start with a `MessageToJson` error, upgrade `protobuf`:

```bash
pip install -U protobuf>=3.20
```

## Dataset format

`train.py` expects a directory containing an `images` folder and a
`poses_bounds.npy` file in the style of the LLFF datasets. Masks are ignored.
If you have multiple downsampled image folders (`images_4`, `images_8`, ...)
the loader picks the first one that exists.

To train on a dataset simply point the script to the directory:

```bash
python train.py --data_dir ./my_scene
```

The near and far bounds used for ray sampling are automatically extracted from
`poses_bounds.npy`. You can override them on the command line if needed.

## Running a training session

The simplest way to train is to rely on the default settings from
`configs/default.json`. You can override any parameter on the command line.
The following command trains the model on a dataset located in `./my_scene`
and stores checkpoints in `./outputs`:

```bash
python train.py --data_dir ./my_scene --out_dir ./outputs \
    --num_epochs 2 --batch_size 2
```

If you maintain your own configuration file pass it via `--config`:

```bash
python train.py --config my_config.json --data_dir ./my_scene
```

Large images can be downsampled on the fly using `--downsample`. For example
`--downsample 4` loads the images at quarter resolution which can speed up
experimentation.

The near and far bounds are read from `poses_bounds.npy` but can be overridden on
the command line or in a custom config file.

Important options include:

- `--num_epochs` – number of passes over the training set.
- `--batch_size` – how many images to load per iteration (default 1).
- `--lr` – learning rate for the Adam optimiser.
- `--near` / `--far` – near and far bounds for ray sampling.
- `--num_samples` – number of points sampled along each ray.
- `--num_rays` – how many rays to sample from each image.
- `--save_every` – how often (in epochs) to write a checkpoint.
- `--eval_index` – which training image to render for monitoring.
- `--log_dir` – base directory for TensorBoard logs.
- `--use_refraction` – set to `1` for refraction-aware training or `0` for a vanilla NeRF.

By default the provided configuration trains a refraction-aware model. To train
a standard NeRF instead, pass `--use_refraction 0` on the command line or set
`"use_refraction": false` in a custom config file.

During training the script prints the MSE loss for each batch.  Logs are also written for TensorBoard under `--log_dir` including PSNR and learning rate.  Start TensorBoard with:

```bash
tensorboard --logdir runs
```

The dashboard shows the training loss, PSNR, learning rate curves and a rendered
example image for each epoch. Each training run creates a new subdirectory in
`--log_dir` and `--out_dir` named after the dataset and a timestamp. The image
is rendered from the camera pose at `--eval_index` (0 by default) and saved as
`render_0000.png`, `render_0001.png`, ... in the output directory so you can
track progress frame by frame.
TensorBoard prints a URL (typically `http://localhost:6006`). Open it in a browser to monitor progress.

Checkpoints are saved every `--save_every` epochs as `model_0000.pt`, `model_0010.pt`, ... inside the run folder under `--out_dir`.

## Inspecting results

A saved checkpoint contains the weights of the `NeRF` network.  You can load it in Python and render novel views of the scene using the functions in `nerf/render.py`.  A minimal example is shown below:

```python
import torch
from nerf.model import NeRF, PositionalEncoding
from nerf.render import render_rays

model = NeRF()
model.load_state_dict(torch.load('outputs/model_0001.pt'))
model.eval()

# generate some rays (origins `rays_o` and directions `rays_d`)
# then call render_rays to obtain RGB values
colour_depth_acc = render_rays(
    lambda pts, dirs: model(PositionalEncoding(10)(pts), PositionalEncoding(4)(dirs)),
    rays_o, rays_d, near=dataset_near, far=dataset_far, num_samples=64,
)
```

The dataset loader and training script are intentionally simple, so you can easily modify them for your own experiments.

## Refraction-aware training and point cloud export

The repository can model scenes that contain a planar water surface. Rays
crossing the interface between air and water are bent according to Snell's
law and routed through separate NeRF networks for each medium. The height of
the water surface is treated as a learnable parameter and is optimised
jointly with the networks.

### Training with refraction

Refraction-aware training is enabled when `use_refraction` is set to `1`.
Specify the refractive indices of air and water and an initial guess for the
water level. The script will optimise both NeRFs and the water surface height:

```bash
python train.py --data_dir ./my_scene --use_refraction 1 \\
    --water_level 0.0 --n_air 1.0 --n_water 1.333
```

During training, checkpoints are written for the two networks
(`air_model_XXXX.pt` and `water_model_XXXX.pt`) as well as the estimated water
level (`water_level_XXXX.pt`). The saved water level can be loaded via
`torch.load(checkpoint)['water_level']`.

### Exporting a refraction-corrected point cloud

After training, a point cloud that accounts for refraction can be extracted by
providing the two network checkpoints and the learned water level:

```bash
python export_pointcloud.py --data_dir ./my_scene \\
    --air_checkpoint outputs/<run>/air_model_0099.pt \\
    --water_checkpoint outputs/<run>/water_model_0099.pt \\
    --water_level <value> --output corrected.ply
```

Adjust `--n_rays`, `--num_samples` and `--weight_threshold` to control the
density and quality of the exported cloud. The resulting `corrected.ply` can be
viewed in standard 3D software.

## Repository structure

- `data/` – dataset utilities including `load_llff_data`, `downsample_data` and `SimpleDataset`.
- `nerf/` – implementation of the `NeRF` model and the volumetric rendering code.
- `train.py` – command-line interface for training the network on an LLFF dataset.

## Further reading

To dive deeper into Neural Radiance Fields consult the original publication:

- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)

This repository is deliberately minimal, providing a concise starting point for custom research or teaching purposes.


![NeRF Batch Size Comparison](nerf_simulation_parallel.gif)

- **Left**: Batch Size 1  
- **Middle**: Batch Size 10  
- **Right**: Batch Size 20  
