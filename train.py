import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nerf.model import NeRF, PositionalEncoding
from nerf.render import render_rays
from data.dataset import SimpleDataset, load_llff_data, downsample_data


DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "default.json")


def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing="ij"
    )
    dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)  # Weltkoordinaten
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # Ursprung überall gleich
    return rays_o, rays_d

def sample_random_rays(img, pose, H, W, focal, N_rand):
    rays_o, rays_d = get_rays(H, W, focal, pose)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    target_rgb = img.reshape(-1, 3)

    indices = torch.randint(0, rays_o.shape[0], (N_rand,))
    return rays_o[indices], rays_d[indices], target_rgb[indices]


def load_config(path: str) -> dict:
    """Load configuration from JSON file."""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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

    images, poses, hwf, near, far = load_llff_data(args.data_dir)

    if args.near is None:
        args.near = near
    if args.far is None:
        args.far = far

    images, hwf = downsample_data(images, hwf, args.downsample)
    H, W, focal = [int(hw) for hw in hwf]

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

            i = torch.randint(0, imgs.shape[0], (1,)).item()
            img = imgs[i]
            pose = poses[i]

            N_rand = args.batch_size  # kommt aus deiner Config oder CLI
            rays_o, rays_d, target = sample_random_rays(img, pose, H, W, focal, N_rand=N_rand)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            target = target.to(device)

            def network(pts, dirs):
                """Encode inputs and evaluate the NeRF model."""
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
    parser.add_argument("--config", help="Path to a JSON config file")
    parser.add_argument("--data_dir", required=True, help="Path to dataset")
    parser.add_argument("--out_dir", default="outputs", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--near", type=float, default=None)
    parser.add_argument("--far", type=float, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--downsample", type=int, default=None,
                        help="Downsample factor for input images")

    args = parser.parse_args()

    # Load configuration defaults and merge with CLI arguments
    config = load_config(DEFAULT_CONFIG)
    if args.config:
        config.update(load_config(args.config))
    for key, val in config.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    train(args)
