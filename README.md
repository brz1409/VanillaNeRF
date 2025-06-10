# Vanilla NeRF

This repository implements a compact version of **Neural Radiance Fields** (NeRF) in PyTorch.  It is intended as an educational resource that exposes all moving parts of a basic NeRF setup with clear code and extensive comments.

The project covers the full pipeline from loading a small Blender-style dataset to training a network that synthesises new views of the scene.

## Installation

Create a virtual environment (optional) and install the Python dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The code only depends on PyTorch and a few utility libraries.  GPU support is recommended but not required.
If you plan to process `cameras.xml` files from Metashape no extra software is
needed. This repository includes a minimal converter inspired by
[`nerfstudio`](https://github.com/nerfstudio-project/nerfstudio) that will
generate the required `transforms.json` automatically.

## Dataset format

`train.py` expects a dataset exported from Blender or a similar tool.  The directory should contain image files and a `transforms_train.json` file describing the camera parameters.  Alternatively you can use an LLFF dataset containing a `poses_bounds.npy` file.  A minimal Blender-style layout looks like this:

```text
my_scene/
├─ transforms_train.json
├─ 000.png
├─ 001.png
└─ ...
```

The JSON file follows the structure used in the original NeRF paper.  Each entry lists the path to an image (relative to the dataset directory) and the `4×4` camera-to-world transformation matrix.  `data/dataset.py` contains the `load_blender_data` function that parses this format and returns NumPy arrays of images and poses plus the image resolution and focal length.

If you already have a dataset in this format simply pass its directory to the training script.

LLFF datasets are also supported. In this case the directory must contain an
`images` folder (optionally downsampled versions like `images_4`) and a
`poses_bounds.npy` file. Masks are ignored.

### Using `cameras.xml` from Agisoft Metashape

Place your Metashape export (`cameras.xml` and the image files) in a directory
of your choice. When `train.py` is pointed at this directory it will run a small
converter that reads the XML file and writes a compatible `transforms.json`. No
external dependencies are necessary. Subsequent runs will reuse the generated
JSON file.

## Running a training session

The following command trains the model on a dataset located in `./my_scene` and stores checkpoints in `./outputs`:

```bash
python train.py --data_dir ./my_scene --out_dir ./outputs \
    --num_epochs 2 --batch_size 2
```

Important options include:

- `--num_epochs` – number of passes over the training set.
- `--batch_size` – how many images to load per iteration.
- `--lr` – learning rate for the Adam optimiser.
- `--near` / `--far` – near and far bounds for ray sampling.
- `--num_samples` – number of points sampled along each ray.

During training the script prints the MSE loss for each batch.  After every epoch a checkpoint named `model_0000.pt`, `model_0001.pt`, ... is written to the output directory.

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
    rays_o, rays_d, near=2.0, far=6.0, num_samples=64,
)
```

The dataset loader and training script are intentionally simple, so you can easily modify them for your own experiments.

## Repository structure

- `data/` – dataset utilities including `load_blender_data`, `load_metashape_data`,
  `load_llff_data` and `SimpleDataset`.
- `nerf/` – implementation of the `NeRF` model and the volumetric rendering code.
- `train.py` – command-line interface for training the network on a Blender dataset.

## Further reading

To dive deeper into Neural Radiance Fields consult the original publication:

- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)

This repository is deliberately minimal, providing a concise starting point for custom research or teaching purposes.
