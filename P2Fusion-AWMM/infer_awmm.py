"""Inference from adverse IR/VIS only; no clean inputs or training teachers."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from awmm.runtime import load_config, make_model, load_awmm_checkpoint


def image_index(folder):
    result = {}
    for path in sorted(Path(folder).iterdir()):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            if path.stem in result:
                raise ValueError("Duplicate sample stem: {}".format(path.stem))
            result[path.stem] = path
    return result


def image_tensor(path):
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="Normally read architecture from checkpoint")
    parser.add_argument("--ir-dir", required=True)
    parser.add_argument("--vis-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = load_config(args.config) if args.config else checkpoint.get("config")
    if not config:
        raise ValueError("Checkpoint has no architecture config; supply --config")
    device = torch.device(args.device)
    model = make_model(config["network"]).to(device)
    load_awmm_checkpoint(model, args.checkpoint)
    model.eval()
    ir_paths, vis_paths = image_index(args.ir_dir), image_index(args.vis_dir)
    if not ir_paths or set(ir_paths) != set(vis_paths):
        raise ValueError("IR/VIS folders must contain identical, nonempty filename-stem sets")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for stem in sorted(ir_paths):
            ir, vis = image_tensor(ir_paths[stem]), image_tensor(vis_paths[stem])
            if ir.shape != vis.shape:
                raise ValueError("Mismatched image dimensions: {}".format(stem))
            result = model(ir.to(device), vis.to(device), return_aux=False)["fused"]
            image = (result[0, 0].float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(image).save(output / (stem + ".png"))
    print(json.dumps({"images": len(ir_paths), "output": str(output), "format": "grayscale fusion"}))


if __name__ == "__main__":
    main()
