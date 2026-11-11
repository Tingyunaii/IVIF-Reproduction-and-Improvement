"""Explicit legacy RGB versus uploaded-original uint8 OpenCV Y conventions."""
import numpy as np
from PIL import Image

INPUT_MODES = ("rgb", "original_y")


def input_mode(config):
    mode = config.get("dataset", {}).get("input_mode", "rgb")
    if mode not in INPUT_MODES:
        raise ValueError("Unknown dataset.input_mode: " + str(mode))
    return mode


def read_model_image(path, mode="rgb"):
    if mode == "rgb":
        with Image.open(path) as image:
            return np.array(image.convert("RGB"), dtype=np.uint8)
    if mode != "original_y":
        raise ValueError("Unknown input mode: " + str(mode))
    import cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Cannot read image: " + str(path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    y = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0:1]
    # Both IR and VIS use n_channels=3 in the supplied original train/test loader.
    return np.repeat(y, 3, axis=2)


def original_color_restore(fused_y, visible_path):
    """Match original test.py: uint8 fused Y + observed VIS Cr/Cb -> RGB."""
    import cv2
    if fused_y.ndim != 4 or fused_y.shape[:2] != (1, 1):
        raise ValueError("Expected one fused Y image [1,1,H,W]")
    y = (fused_y[0, 0].detach().float().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    visible = cv2.imread(str(visible_path), cv2.IMREAD_COLOR)
    if visible is None:
        raise ValueError("Cannot read visible image: " + str(visible_path))
    y_vis, cr, cb = cv2.split(cv2.cvtColor(visible, cv2.COLOR_BGR2YCrCb))
    if y.shape != y_vis.shape:
        y = cv2.resize(y, (y_vis.shape[1], y_vis.shape[0]), interpolation=cv2.INTER_LINEAR)
    bgr = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
