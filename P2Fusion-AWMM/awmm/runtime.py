"""Shared configuration, strict checkpoint handling and clean-reference teachers."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    for key in ("dataset", "network", "train", "loss", "teachers"):
        if key not in config:
            raise ValueError("Configuration is missing {!r}".format(key))
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(network_config):
    from models.awmm_network import AWMMP2Fusion
    return AWMMP2Fusion(**network_config)


def manifest_digest(paths):
    """Bind a resumed run to the actual manifest bytes, not absolute paths."""
    result = {}
    for path in paths:
        path = Path(path)
        result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def extract_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint/state dictionary")
    for key in ("model", "params_ema", "params", "state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    result = {}
    for key, value in checkpoint.items():
        while key.startswith("module."):
            key = key[len("module."):]
        if key in result:
            raise ValueError("Duplicate checkpoint key after removing module prefix: " + key)
        result[key] = value
    return result


def load_awmm_checkpoint(model, path):
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    model.load_state_dict(extract_state(checkpoint), strict=True)
    return checkpoint


class CleanReferenceTeachers(nn.Module):
    """Real, frozen IST/VQT. No fallback to random or heuristic teachers."""

    def __init__(self, config, device):
        super().__init__()
        from nets.segformer import SegFormer
        teacher_config = config["teachers"]
        path = Path(teacher_config["ir_checkpoint"]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                "Missing pretrained IR saliency teacher: {}. Copy the original "
                "nets/best_epoch_weights.pth or set teachers.ir_checkpoint.".format(path)
            )
        self.ir = SegFormer(num_classes=2, phi="b2", pretrained=False)
        state = extract_state(torch.load(str(path), map_location="cpu", weights_only=False))
        self.ir.load_state_dict(state, strict=True)
        try:
            import pyiqa
        except ImportError as error:
            raise RuntimeError("Real training requires the original pyiqa BRISQUE dependency.") from error
        self.vis = pyiqa.create_metric("brisque", device=device)
        self.to(device).eval()
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, clean_ir, clean_vis):
        # Inputs follow dataset.input_mode: original_y is uint8 OpenCV Y
        # repeated across channels; legacy rgb is actual RGB. No extra normalization.
        # Run outside mixed precision for saliency probabilities and BRISQUE.
        self.eval()
        with torch.autocast(device_type=clean_ir.device.type, enabled=False):
            logits = self.ir(clean_ir.float())
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite clean IR teacher logits")
            saliency = logits.softmax(dim=1)[:, 1:2]
            raw_quality = self.vis(clean_vis.float()).float().reshape(-1, 1)
            if not torch.isfinite(raw_quality).all():
                raise FloatingPointError("Non-finite clean BRISQUE score")
            quality = ((100.0 - raw_quality) / 100.0).clamp(0.0, 1.0)
        return {"ir": saliency, "vis": quality}


def build_teachers(config, device):
    return CleanReferenceTeachers(config, device)
