import argparse
import torch
import imageio

from data.dataset import load_llff_data, downsample_data
from nerf.model import NeRF, PositionalEncoding
from nerf.render import render_rays
from train import get_rays


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, poses, hwf, near, far = load_llff_data(args.data_dir)
    images, hwf = downsample_data(images, hwf, args.downsample)
    H, W, focal = [int(x) for x in hwf]

    model = NeRF()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    pos_enc = PositionalEncoding(10)
    dir_enc = PositionalEncoding(4)

    pose = torch.from_numpy(poses[0]).to(device)
    rays_o, rays_d = get_rays(H, W, focal, pose)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    with torch.no_grad():
        rgb_depth_acc = render_rays(
            lambda pts, dirs: model(pos_enc(pts), dir_enc(dirs)),
            rays_o,
            rays_d,
            near=near,
            far=far,
            num_samples=args.num_samples,
            rand=False,
        )
    rgb = rgb_depth_acc[:, :3].reshape(H, W, 3).cpu().numpy()
    imageio.imwrite(args.output, (rgb * 255).astype("uint8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="render.png")
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=64)
    main(parser.parse_args())
