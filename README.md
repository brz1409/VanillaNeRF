# Vanilla NeRF

![NeRF Batch Size Comparison](nerf_simulation_parallel.gif)

This repository implements a compact version of **Neural Radiance Fields** (NeRF) in PyTorch.  It is intended as an educational resource that exposes all moving parts of a basic NeRF setup with clear code and extensive comments.

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

## Repository structure

- `data/` – dataset utilities including `load_llff_data`, `downsample_data` and `SimpleDataset`.
- `nerf/` – implementation of the `NeRF` model and the volumetric rendering code.
- `train.py` – command-line interface for training the network on an LLFF dataset.

## Further reading

To dive deeper into Neural Radiance Fields consult the original publication:

- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)

This repository is deliberately minimal, providing a concise starting point for custom research or teaching purposes.
