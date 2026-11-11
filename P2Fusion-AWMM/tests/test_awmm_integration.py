"""End-to-end CPU wiring test. Mock teachers are confined to this test process."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from awmm.data import write_awmm_splits
from awmm.runtime import load_config
import train_awmm
import infer_awmm


class TestTeachers(torch.nn.Module):
    def forward(self, ir, vis):
        return {"ir": ir.mean(1, keepdim=True), "vis": vis.new_full((vis.shape[0], 1), 0.75)}


class EndToEndTests(unittest.TestCase):
    def test_manifest_train_resume_and_adverse_only_inference(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, manifests, run = base / "dataset", base / "splits", base / "run"
            rng = np.random.RandomState(3)
            for folder, count, visible_name in (
                (root / "Rain/Scene_A/Train", 5, "vi_Medium"),
                (root / "AWMM-Text/Test/Haze", 1, "vi"),
            ):
                for index in range(count):
                    clean_ir = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                    clean_vis = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                    arrays = {"gt_ir": clean_ir, "gt_vi": clean_vis,
                              "ir": (clean_ir * 0.8).astype(np.uint8),
                              visible_name: (clean_vis * 0.7 + 30).astype(np.uint8)}
                    for name, image in arrays.items():
                        destination = folder / name
                        destination.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(image).save(destination / (str(index) + ".png"))
            write_awmm_splits(root, manifests, val_ratio=0.2)
            config = load_config(Path(__file__).resolve().parents[1] / "configs/awmm.json")
            config["dataset"].update(root=str(root), manifest_dir=str(manifests), crop_size=64)
            config["network"].update(num_blocks=1, Ex_depths=[1], input_resolution=[16, 16])
            config["train"].update(batch_size=2, num_workers=0, epochs=2, freeze_encoder_steps=1,
                                    amp=False, log_every=1, val_every=1, save_every=1)
            config["run_dir"] = str(run)
            config_path = base / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with patch("awmm.runtime.build_teachers", return_value=TestTeachers()), contextlib.redirect_stdout(io.StringIO()):
                train_awmm.main(["--config", str(config_path), "--from-scratch", "--device", "cpu", "--max-steps", "1"])
                first = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
                self.assertEqual(first["step"], 1)
                self.assertTrue((run / "best.pt").is_file())
                train_awmm.main(["--config", str(config_path), "--resume", str(run / "last.pt"),
                                 "--device", "cpu", "--max-steps", "2"])
            second = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(second["step"], 2)
            rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
            self.assertTrue(any(row.get("resume") for row in rows if row["event"] == "start"))
            expected = {"intensity", "ssim", "gradient", "distill_ir", "distill_vis",
                        "consistency", "orthogonality", "self_supervision"}
            self.assertTrue(expected.issubset(next(row for row in rows if row["event"] == "validation")))
            arguments = ["infer_awmm.py", "--checkpoint", str(run / "best.pt"),
                         "--ir-dir", str(root / "AWMM-Text/Test/Haze/ir"),
                         "--vis-dir", str(root / "AWMM-Text/Test/Haze/vi"),
                         "--output", str(base / "output"), "--device", "cpu"]
            with patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                infer_awmm.main()
            with Image.open(base / "output/0.png") as image:
                self.assertEqual(image.size, (64, 64))
                self.assertEqual(image.mode, "L")
            for color_source in ("input", "reconstructed"):
                color_arguments = list(arguments)
                color_arguments[color_arguments.index("--output") + 1] = str(base / color_source)
                color_arguments += ["--color-source", color_source]
                with patch("sys.argv", color_arguments), contextlib.redirect_stdout(io.StringIO()):
                    infer_awmm.main()
                with Image.open(base / color_source / "0.png") as image:
                    self.assertEqual(image.size, (64, 64))
                    self.assertEqual(image.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
