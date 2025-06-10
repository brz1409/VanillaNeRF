import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nerf.model import NeRF, PositionalEncoding
from nerf.render import render_rays
from data.dataset import SimpleDataset, load_blender_data


def create_network() -> NeRF:
    """Instantiate the NeRF model and positional encoders."""

    # Positional encoders increase the capacity of the MLP to represent
    # high-frequency details. ``10`` and ``4`` are common choices for the
    # number of frequencies used for spatial coordinates and view directions
    # respectively.
    pos_enc = PositionalEncoding(10)
    dir_enc = PositionalEncoding(4)

    # Input dimensions depend on the output of the encoders. We include the
    # raw coordinates themselves as well.
    model = NeRF(input_ch=3 * 10 * 2 + 3, input_ch_dir=3 * 4 * 2 + 3)
    return model, pos_enc, dir_enc


def train(args):
    """Main training loop."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset from disk
    images, poses, hwf = load_blender_data(args.data_dir)
    dataset = SimpleDataset(images, poses, hwf)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create network and optimizer
    model, pos_enc, dir_enc = create_network()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for imgs, poses in pbar:
            imgs = imgs.to(device)

            # The Blender dataset stores camera directions in the rotation
            # matrix columns. Here we extract ray origins and directions for
            # all pixels in the batch.
            rays_o = poses[:, :3, 3]
            rays_d = poses[:, :3, 2]
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            target = imgs.reshape(-1, 3)

            enc_pts = pos_enc(rays_o)
            enc_dirs = dir_enc(rays_d)

            def network(pts, dirs):
                """Small wrapper so ``render_rays`` can query the model."""

                enc_p = pos_enc(pts)
                enc_d = dir_enc(dirs)
                return model(enc_p, enc_d)

            outputs = render_rays(
                network, rays_o, rays_d, args.near, args.far, args.num_samples
            )
            pred_rgb = outputs[:, :3]
            loss = torch.mean((pred_rgb - target) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})

        ckpt_path = os.path.join(args.out_dir, f"model_{epoch:04d}.pt")
        torch.save(model.state_dict(), ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Path to dataset')
    parser.add_argument('--out_dir', default='outputs', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--near', type=float, default=2.0)
    parser.add_argument('--far', type=float, default=6.0)
    parser.add_argument('--num_samples', type=int, default=64)
    args = parser.parse_args()
    train(args)
