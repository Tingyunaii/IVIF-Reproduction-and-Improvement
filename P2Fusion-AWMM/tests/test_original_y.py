"""Audit against the actual uploaded original functions/postprocessing block."""
import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
import torch

from awmm.preprocessing import read_model_image, original_color_restore, input_mode
from train_awmm import resume_semantics

ROOT = Path(__file__).resolve().parents[1]


def original_functions():
    tree = ast.parse((ROOT / "utils/utils_image.py").read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in ("imread_uint", "tensor2uint")]
    namespace = {"cv2": cv2, "np": np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "original_utils", "exec"), namespace)
    return namespace


class OriginalYTests(unittest.TestCase):
    def test_bit_exact_original_loader_png_and_jpeg(self):
        original = original_functions()["imread_uint"]
        rng = np.random.RandomState(99)
        with tempfile.TemporaryDirectory() as tmp:
            for extension in ("png", "jpg"):
                path = Path(tmp) / ("colored_ir_or_vis." + extension)
                cv2.imwrite(str(path), rng.randint(0, 256, (31, 37, 3), dtype=np.uint8))
                expected = original(str(path), n_channels=3)
                actual = read_model_image(path, "original_y")
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(actual[:, :, 0], actual[:, :, 2])

    def test_bit_exact_original_test_color_block(self):
        original = original_functions()
        tree = ast.parse((ROOT / "test.py").read_text(encoding="utf-8"))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        loop = next(n for n in main.body if isinstance(n, ast.For))
        def assigned(node, name):
            return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
        start = next(i for i, n in enumerate(loop.body) if assigned(n, "fused_Y"))
        end = next(i for i, n in enumerate(loop.body) if assigned(n, "fused_rgb"))
        block = ast.Module(body=loop.body[start:end + 1], type_ignores=[])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vis.png"
            cv2.imwrite(str(path), np.random.RandomState(7).randint(0, 256, (29, 31, 3), dtype=np.uint8))
            for height, width in ((29, 31), (13, 19)):
                fused = torch.linspace(-.1, 1.1, height * width).view(1, 1, height, width)
                context = {"output": fused[0].clone(), "vis_path": str(path), "cv2": cv2,
                           "util": SimpleNamespace(tensor2uint=original["tensor2uint"])}
                exec(compile(block, "original_test_color", "exec"), context)
                np.testing.assert_array_equal(original_color_restore(fused, path), context["fused_rgb"])

    def test_resume_mode_is_bound_and_legacy_is_rgb(self):
        import copy
        import json
        config = json.loads((ROOT / "configs/awmm.json").read_text())
        self.assertEqual(input_mode(config), "rgb")
        strict = copy.deepcopy(config)
        strict["dataset"]["input_mode"] = "original_y"
        self.assertNotEqual(resume_semantics(config), resume_semantics(strict))

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            input_mode({"dataset": {"input_mode": "unknown"}})


if __name__ == "__main__":
    unittest.main()
