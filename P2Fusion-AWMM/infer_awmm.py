"""Inference from adverse IR/VIS only; no clean inputs or training teachers."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from awmm.runtime import load_config, make_model, load_awmm_checkpoint
from awmm.preprocessing import input_mode, read_model_image, original_color_restore, fused_y_uint8


def image_index(folder):
    result = {}
    for path in sorted(Path(folder).iterdir()):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            if path.stem in result:
                raise ValueError("Duplicate sample stem: {}".format(path.stem))
            result[path.stem] = path
    return result


def image_tensor(path, mode="rgb"):
    array = read_model_image(path, mode).astype(np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)


def restore_color(fused, color_reference, reference_mode):
    """Postprocess color; this is not a jointly trained RGB fusion decoder.

    Adding an achromatic offset preserves reference chroma before clipping.
    Anchor the offset to the actual training convention: first RGB channel
    (legacy default) or luminance. Never silently interpret red as Y.
    """
    if (fused.ndim != 4 or fused.shape[1] != 1 or color_reference.ndim != 4
            or color_reference.shape[1] != 3
            or fused.shape[0] != color_reference.shape[0]
            or fused.shape[-2:] != color_reference.shape[-2:]):
        raise ValueError("Expected aligned B1HW fusion and B3HW color reference")
    if reference_mode == "first_channel":
        anchor = color_reference[:, :1]
    elif reference_mode == "luminance":
        coefficients = color_reference.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        anchor = (color_reference * coefficients).sum(1, keepdim=True)
    else:
        raise ValueError("Unknown fusion_reference: " + str(reference_mode))
    return (color_reference.float() + fused.float() - anchor.float()).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="Normally read architecture from checkpoint")
    parser.add_argument("--ir-dir", required=True)
    parser.add_argument("--vis-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-y-dir", default=None,
                        help="Also save decoder Y as 8-bit grayscale PNG; original_y checkpoints only")
    parser.add_argument("--color-source", choices=("none", "input", "reconstructed", "original"), default=None,
                        help="original_y defaults to original OpenCV Y+observed Cr/Cb; RGB mode keeps legacy options")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = load_config(args.config) if args.config else checkpoint.get("config")
    if not config:
        raise ValueError("Checkpoint has no architecture config; supply --config")
    mode = input_mode(config)
    if args.save_y_dir and mode != "original_y":
        raise ValueError("Decoder output can be called Y only for an original_y-trained checkpoint")
    if checkpoint.get("config") and input_mode(checkpoint["config"]) != mode:
        raise ValueError("Input mode override differs from checkpoint training metadata")
    if args.color_source is None:
        args.color_source = "original" if mode == "original_y" else "none"
    if args.color_source == "original" and mode != "original_y":
        raise ValueError("Original Y/Cr/Cb restoration requires an original_y-trained checkpoint; "
                         "legacy RGB/R checkpoints need a separate Y fine-tuning/training stage")
    if mode == "original_y" and args.color_source not in ("none", "original"):
        raise ValueError("original_y supports only original observed Cr/Cb or grayscale Y output")
    device = torch.device(args.device)
    model = make_model(config["network"], config.get("model_variant", "awmm")).to(device)
    load_awmm_checkpoint(model, args.checkpoint)
    model.eval()
    ir_paths, vis_paths = image_index(args.ir_dir), image_index(args.vis_dir)
    if not ir_paths or set(ir_paths) != set(vis_paths):
        raise ValueError("IR/VIS folders must contain identical, nonempty filename-stem sets")
    output = Path(args.output)
    y_output = Path(args.save_y_dir) if args.save_y_dir else None
    if y_output is not None:
        if y_output.resolve() == output.resolve():
            raise ValueError("--save-y-dir must differ from --output")
        y_output.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for stem in sorted(ir_paths):
            ir, vis = image_tensor(ir_paths[stem], mode), image_tensor(vis_paths[stem], mode)
            if ir.shape != vis.shape:
                raise ValueError("Mismatched image dimensions: {}".format(stem))
            ir, vis = ir.to(device), vis.to(device)
            outputs = model(ir, vis, return_aux=args.color_source == "reconstructed")
            result = outputs["fused"]
            if y_output is not None:
                Image.fromarray(fused_y_uint8(result)).save(y_output / (stem + ".png"))
            if args.color_source == "original":
                Image.fromarray(original_color_restore(result, vis_paths[stem])).save(output / (stem + ".png"))
                continue
            if args.color_source == "none":
                array = result[0, 0].float().clamp(0, 1).cpu().numpy()
            else:
                reference = (vis if args.color_source == "input"
                             else outputs["aux"]["vis"]["clean_recon"])
                convention = config.get("loss", {}).get("fusion_reference", "first_channel")
                rgb = restore_color(result, reference, convention)
                array = rgb[0].permute(1, 2, 0).cpu().numpy()
            image = (array * 255).round().astype(np.uint8)
            Image.fromarray(image).save(output / (stem + ".png"))
    print(json.dumps({"images": len(ir_paths), "output": str(output),
                      "format": "grayscale fusion" if args.color_source == "none" else "RGB color postprocess",
                      "input_mode": mode,
                      "y_output": str(y_output) if y_output is not None else None,
                      "color_source": args.color_source}))


if __name__ == "__main__":
    main()
