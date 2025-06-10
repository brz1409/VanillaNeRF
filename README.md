# Vanilla NeRF

This repository contains a minimal implementation of a Vanilla NeRF model in PyTorch. The goal is to provide an accessible starting point for experimenting with neural radiance fields and view synthesis.

## Features
- Positional encoding for input coordinates and view directions
- Fully connected MLP architecture for predicting color and density
- Volume rendering to synthesize images
- Training script with PSNR evaluation

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

## Dataset
The code expects a dataset of images, camera intrinsics and poses similar to the [LLFF](https://github.com/Fyusion/LLFF) or Blender synthetic datasets. Update paths in `train.py` accordingly.

## Usage
Training can be started with:
```bash
python train.py --data_dir /path/to/dataset
```
Rendered images and checkpoints will be saved to the `outputs` folder.

## Reference
- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)
