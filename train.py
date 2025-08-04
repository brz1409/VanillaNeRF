"""Training script for Vanilla NeRF."""

import argparse
import json
import math
import os
import torch
import imageio
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
from tqdm import tqdm


from nerf.model import NeRF, PositionalEncoding
from nerf.render import render_rays
from nerf.refraction import refract_rays
from data.dataset import SimpleDataset, load_llff_data


DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "default.json")


def get_rays(H, W, focal, c2w):
    """Generate ray origins and directions for all pixels in an image."""

    device = c2w.device  # automatically use the pose's device

    i, j = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing="xy",
    )
    dirs = torch.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1
    )
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d

def sample_random_rays(img, pose, H, W, focal, N_rand):
    """Sample ``N_rand`` random pixels from ``img`` and return their rays."""

    rays_o, rays_d = get_rays(H, W, focal, pose)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    target_rgb = img.reshape(-1, 3)

    indices = torch.randint(0, rays_o.shape[0], (N_rand,))
    return rays_o[indices], rays_d[indices], target_rgb[indices]


def load_config(path: str) -> dict:
    """Load configuration from JSON file."""

    # Using JSON keeps the configuration human readable and easy to edit.
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

    run_name = args.run_name

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Dataset directory '{args.data_dir}' not found")

    images, poses, hwf, near, far = load_llff_data(
        args.data_dir, downsample=args.downsample, save_downsampled=True
    )
    orig_hw = (int(hwf[0] * (args.downsample or 1)), int(hwf[1] * (args.downsample or 1)))

    if args.near is None:
        args.near = near
    if args.far is None:
        args.far = far

    H, W, focal = [int(hw) for hw in hwf]

    if args.downsample and args.downsample > 1:
        print(
            f"Downsampled images from {orig_hw[0]}x{orig_hw[1]} to {H}x{W} (factor {args.downsample})"
        )
    else:
        print(f"Using images at {H}x{W} (no downsampling)")

    dataset = SimpleDataset(images, poses, hwf)
    if args.eval_index < 0 or args.eval_index >= len(dataset):
        raise ValueError(f"eval_index {args.eval_index} is out of range")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(
        f"TensorBoard logs at {log_dir}. "
        f"Run `tensorboard --logdir {args.log_dir}` to view."
    )
    print(f"Rendering evaluation image from index {args.eval_index}")

    # Create network and optimizer
    air_model, air_pos_enc, air_dir_enc = create_network()
    water_model, water_pos_enc, water_dir_enc = create_network()
    air_model.to(device)
    water_model.to(device)
    air_pos_enc.to(device)
    water_pos_enc.to(device)
    air_dir_enc.to(device)
    water_dir_enc.to(device)
    water_level = torch.nn.Parameter(torch.tensor(args.water_level, device=device))
    params = list(air_model.parameters()) + list(water_model.parameters()) + [water_level]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    def network_air(pts, dirs):
        enc_p = air_pos_enc(pts)
        enc_d = air_dir_enc(dirs)
        return air_model(enc_p, enc_d)

    def network_water(pts, dirs):
        enc_p = water_pos_enc(pts)
        enc_d = water_dir_enc(dirs)
        return water_model(enc_p, enc_d)

    epoch_bar = tqdm(range(args.num_epochs), desc="Training", unit="epoch")
    global_step = 0
    for epoch in epoch_bar:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
        epoch_loss = 0.0
        for imgs, poses in pbar:
            imgs = imgs.to(device)

            i = torch.randint(0, imgs.shape[0], (1,)).item()
            img = imgs[i]
            pose = poses[i]

            N_rand = args.num_rays
            rays_o, rays_d, target = sample_random_rays(
                img, pose, H, W, focal, N_rand=N_rand
            )
            # Diagnostic printout to keep track of progress and randomness.
            print(f"Sampled {N_rand} random rays for epoch {epoch + 1}, batch step {global_step}")

            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            target = target.to(device)

            mask, refr_o, refr_d = refract_rays(
                rays_o, rays_d, water_level, args.n_air, args.n_water
            )
            outputs = torch.zeros(rays_o.shape[0], 5, device=device)
            if (~mask).any():
                out_air = render_rays(
                    network_air,
                    rays_o[~mask],
                    rays_d[~mask],
                    args.near,
                    args.far,
                    args.num_samples,
                )
                outputs[~mask] = out_air
            if mask.any():
                out_water = render_rays(
                    network_water,
                    refr_o,
                    refr_d,
                    args.near,
                    args.far,
                    args.num_samples,
                )
                outputs[mask] = out_water
            pred_rgb = outputs[:, :3]
            loss = torch.mean((pred_rgb - target) ** 2)
            psnr = -10.0 * torch.log10(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/psnr", psnr.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item(), "psnr": psnr.item()})

        with torch.no_grad():
            pose = dataset.poses[args.eval_index].to(device)
            rays_o_full, rays_d_full = get_rays(H, W, focal, pose)
            rays_o_full = rays_o_full.reshape(-1, 3)
            rays_d_full = rays_d_full.reshape(-1, 3)
            mask, refr_o_full, refr_d_full = refract_rays(
                rays_o_full, rays_d_full, water_level, args.n_air, args.n_water
            )
            rgb_depth = torch.zeros(rays_o_full.shape[0], 5, device=device)
            if (~mask).any():
                rgb_depth[~mask] = render_rays(
                    network_air,
                    rays_o_full[~mask],
                    rays_d_full[~mask],
                    args.near,
                    args.far,
                    args.num_samples,
                    rand=False,
                )
            if mask.any():
                rgb_depth[mask] = render_rays(
                    network_water,
                    refr_o_full,
                    refr_d_full,
                    args.near,
                    args.far,
                    args.num_samples,
                    rand=False,
                )
            img_pred = rgb_depth[:, :3].reshape(H, W, 3)
            writer.add_image(
                "train/render", img_pred.permute(2, 0, 1).cpu(), epoch
            )
            if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
                img_path = os.path.join(out_dir, f"render_{epoch:04d}.png")
                imageio.imwrite(
                    img_path, (img_pred.cpu().numpy() * 255).astype("uint8")
                )
                # Save a rendered image for visual inspection of progress.
                print(f"Saved evaluation image to {img_path}")

            target_img = dataset.images[args.eval_index].to(device)
            mse_img = torch.mean((img_pred - target_img) ** 2)
            psnr_eval = (-10.0 * torch.log10(mse_img))
            writer.add_scalar("eval/psnr", psnr_eval.item(), epoch)

            avg_loss = epoch_loss / len(dataloader)
            writer.add_scalar("epoch/loss", avg_loss, epoch)
            writer.add_histogram(
                "params/alpha_air", air_model.alpha_linear.weight, epoch
            )
            writer.add_histogram(
                "params/alpha_water", water_model.alpha_linear.weight, epoch
            )

            writer.flush()

            if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
                ckpt_air = os.path.join(out_dir, f"air_model_{epoch:04d}.pt")
                ckpt_water = os.path.join(out_dir, f"water_model_{epoch:04d}.pt")
                torch.save(air_model.state_dict(), ckpt_air)
                torch.save(water_model.state_dict(), ckpt_water)
                torch.save({'water_level': water_level.detach().cpu()},
                           os.path.join(out_dir, f"water_level_{epoch:04d}.pt"))
                # Checkpoints allow resuming training or rendering later on.
                print(f"Saved checkpoints to {ckpt_air} and {ckpt_water}")
            epoch_bar.set_postfix({"loss": avg_loss, "psnr": psnr_eval.item()})

    writer.close()


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
    parser.add_argument(
        "--num_rays",
        type=int,
        default=None,
        help="Number of random rays per image",
    )
    parser.add_argument("--downsample", type=int, default=None,
                        help="Downsample factor for input images")
    parser.add_argument("--log_dir", default="runs",
                        help="Directory for TensorBoard logs")
    parser.add_argument("--save_every", type=int, default=None,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--eval_index", type=int, default=None,
                        help="Index of the training image used for TensorBoard renders")
    parser.add_argument("--water_level", type=float, default=None,
                        help="Initial height of the water surface")
    parser.add_argument("--n_air", type=float, default=None,
                        help="Refractive index of air")
    parser.add_argument("--n_water", type=float, default=None,
                        help="Refractive index of water")

    args = parser.parse_args()

    # Load configuration defaults and merge with CLI arguments
    config = load_config(DEFAULT_CONFIG)
    if args.config:
        config.update(load_config(args.config))
    for key, val in config.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    # Create a unique name for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.run_name = f"{os.path.basename(args.data_dir.rstrip(os.sep))}_{timestamp}"

    train(args)
