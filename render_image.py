import argparse
import torch
import imageio
import yaml
import os

from data.dataset import load_llff_data, downsample_data
from nerf.model import NeRF, PositionalEncoding
from nerf.render import render_rays
from train import get_rays

def load_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

def main(args):
    # Konfiguration laden
    config = load_config(os.path.join(args.data_dir, "config.yaml"))
    # Downsample aus config übernehmen, falls nicht per Argument gesetzt
    downsample = args.downsample if args.downsample != 1 else config.get("downsample", 1)
    print(f"Verwende Downsampling-Faktor: {downsample}")

    print(f"Starte Rendering mit Daten aus: {args.data_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Verwende Gerät: {device}")

    print("Lade Daten ...")
    images, poses, hwf, near, far = load_llff_data(args.data_dir)
    print("Daten geladen.")

    print(f"Downsampling mit Faktor {downsample} ...")
    images, hwf = downsample_data(images, hwf, downsample)
    print("Downsampling abgeschlossen.")

    H, W, focal = [int(x) for x in hwf]
    print(f"Bildgröße: {H}x{W}, Fokal: {focal}")

    print(f"Lade Modell-Checkpoint: {args.checkpoint}")
    model = NeRF()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()
    print("Modell geladen und auf eval gesetzt.")

    print("Initialisiere Positional Encodings ...")
    pos_enc = PositionalEncoding(10)
    dir_enc = PositionalEncoding(4)
    print("Positional Encodings initialisiert.")


    print(f"Renderbild-Index: 0, Pose-Shape: {poses.shape}")
    pose = torch.from_numpy(poses[0]).to(device)
    print("Berechne Strahlen ...")
    rays_o, rays_d = get_rays(H, W, focal, pose)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    print("Strahlen berechnet.")

    print("Starte Rendering ...")
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
    print("Rendering abgeschlossen.")

    rgb = rgb_depth_acc[:, :3].reshape(H, W, 3).cpu().numpy()
    print(f"Speichere Bild unter: {args.output}")
    imageio.imwrite(args.output, (rgb * 255).astype("uint8"))
    print("Fertig.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="render.png")
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=64)
    main(parser.parse_args())