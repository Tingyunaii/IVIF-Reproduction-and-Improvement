import json
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from train_awmm import (audit_split_manifests, atomic_save, capture_rng,
                        load_training_state, optimizer_groups, parse_args,
                        restore_rng, resume_semantics, set_encoder_frozen, validate)


class MiniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_first = nn.Sequential(nn.Linear(2, 2), nn.Dropout())
        self.patch_embed = nn.Linear(2, 2)
        self.rstb = nn.Linear(2, 2)
        self.norm_Ex_ir = nn.LayerNorm(2)
        self.fdn_ir = nn.Linear(2, 2)
        self.dwam_ir = nn.Linear(2, 2)


class TrainingTests(unittest.TestCase):
    def test_encoder_freeze_does_not_freeze_adapters(self):
        model = MiniModel().train()
        set_encoder_frozen(model, True)
        self.assertFalse(model.conv_first.training)
        self.assertTrue(all(not p.requires_grad for p in model.conv_first.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.fdn_ir.parameters()))
        set_encoder_frozen(model, False)
        self.assertTrue(model.conv_first.training)
        self.assertTrue(all(p.requires_grad for p in model.conv_first.parameters()))

    def test_optimizer_includes_frozen_encoder_and_adapter_lr(self):
        model = MiniModel()
        set_encoder_frozen(model, True)
        groups = optimizer_groups(model, 1e-4, 0.1)
        self.assertEqual(groups[0]["lr"], 1e-5)
        self.assertEqual(groups[1]["lr"], 1e-4)
        self.assertEqual(sum(len(group["params"]) for group in groups), len(list(model.parameters())))

    def test_manifest_audit_catches_individual_clean_reference_leak(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / name for name in ("train.json", "val.json")]
            for path, split, sample in zip(paths, ("train", "val"), ("a", "b")):
                path.write_text(json.dumps({"schema_version": 1, "split": split, "records": [
                    {"id": sample, "group": sample, "clean_ir_sha256": sample,
                     "clean_vis_sha256": "same-image"}]}))
            with self.assertRaisesRegex(ValueError, "clean_vis_sha256"):
                audit_split_manifests(*paths)
            value = json.loads(paths[1].read_text())
            value["records"][0]["clean_vis_sha256"] = "different"
            paths[1].write_text(json.dumps(value))
            audit_split_manifests(*paths)

    def test_rng_roundtrip(self):
        random.seed(3)
        np.random.seed(3)
        torch.manual_seed(3)
        state = capture_rng()
        expected = (random.random(), float(np.random.random()), torch.rand(2))
        restore_rng(state)
        self.assertEqual(expected[0], random.random())
        self.assertEqual(expected[1], float(np.random.random()))
        torch.testing.assert_close(expected[2], torch.rand(2))

    def test_resume_semantics_bind_loss_validation_and_training_but_allow_relocation(self):
        import copy
        configuration = {"network": {"drop_rate": 0.0}, "loss": {"ssim_data_range": 1},
                         "teachers": {"ir_checkpoint": "a.pth"}, "dataset": {"root": "old", "crop_size": 256},
                         "train": {"epochs": 10, "lr": 1e-4, "validation_batch_limit": 0}}
        relocated = copy.deepcopy(configuration)
        relocated["dataset"]["root"] = "new"
        relocated["train"]["epochs"] = 20
        self.assertEqual(resume_semantics(configuration), resume_semantics(relocated))
        changed = copy.deepcopy(configuration)
        changed["train"]["validation_batch_limit"] = 2
        self.assertNotEqual(resume_semantics(configuration), resume_semantics(changed))
        changed = copy.deepcopy(configuration)
        changed["loss"]["ssim_data_range"] = 255
        self.assertNotEqual(resume_semantics(configuration), resume_semantics(changed))
        changed = copy.deepcopy(configuration)
        changed["teachers"]["ir_checkpoint"] = "other.pth"
        self.assertNotEqual(resume_semantics(configuration), resume_semantics(changed))

    def test_complete_checkpoint_resume_and_manifest_rejection(self):
        model = MiniModel()
        optimizer = torch.optim.AdamW(optimizer_groups(model, 1e-4, 0.1))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2], gamma=0.5)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        model.fdn_ir(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        expected = {key: value.clone() for key, value in model.state_dict().items()}
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                   "epoch": 1, "batch_in_epoch": 2, "step": 3, "best_val": 4.0,
                   "manifest_digest": "abc", "rng": capture_rng()}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pt"
            atomic_save(payload, path)
            self.assertFalse(path.with_name("last.pt.tmp").exists())
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            state = load_training_state(path, model, optimizer, scheduler, scaler, "abc")
            self.assertEqual(state["step"], 3)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[key])
            with self.assertRaisesRegex(ValueError, "manifests differ"):
                load_training_state(path, model, optimizer, scheduler, scaler, "changed")

    def test_cli_source_modes_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            parse_args(["--resume", "last.pt", "--from-scratch"])

    def test_validation_is_sample_weighted_and_restores_training_mode(self):
        model = MiniModel().train()
        loader = [
            {key: torch.zeros(2, 3, 32, 32) for key in ("ir", "vis", "clean_ir", "clean_vis")},
            {key: torch.ones(1, 3, 32, 32) for key in ("ir", "vis", "clean_ir", "clean_vis")},
        ]
        def fake_losses(model, criterion, teachers, batch, amp):
            self.assertFalse(model.training)
            return {"total": batch["ir"].mean() + 1}
        with patch("train_awmm.calculate_losses", fake_losses):
            result = validate(model, None, None, loader, torch.device("cpu"), False)
            self.assertAlmostEqual(result["total"], 4 / 3)
            self.assertTrue(model.training)
            limited = validate(model, None, None, loader, torch.device("cpu"), False, max_batches=1)
            self.assertEqual(limited["total"], 1)

    def test_real_student_one_step_save_and_resume_without_external_teachers(self):
        """Integration only: fake labels are patched in this test, never production."""
        from PIL import Image
        import train_awmm

        torch.set_num_threads(2)
        root = Path(__file__).resolve().parents[1]
        configuration = json.loads((root / "configs" / "awmm.json").read_text())
        configuration["network"].update(num_blocks=1, Ex_depths=[1])
        configuration["train"].update(epochs=3, batch_size=2, num_workers=0,
                                       freeze_encoder_steps=1, amp=False)
        configuration["dataset"]["crop_size"] = 64

        class SyntheticTeachers:
            def __call__(self, clean_ir, clean_vis):
                return {"ir": clean_ir.mean(1, keepdim=True),
                        "vis": clean_vis.new_full((len(clean_vis), 1), 0.8)}

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rows = []
            for index in range(3):
                row = {"id": str(index), "group": str(index),
                       "clean_ir_sha256": "ir" + str(index),
                       "clean_vis_sha256": "vis" + str(index)}
                pixels = np.random.default_rng(index).integers(0, 256, (64, 64, 3), dtype=np.uint8)
                for key in ("ir", "vis", "clean_ir", "clean_vis"):
                    name = "{}_{}.png".format(index, key)
                    Image.fromarray(pixels).save(directory / name)
                    row[key] = name
                rows.append(row)
            manifests = directory / "splits"
            manifests.mkdir()
            for name, selected in (("train", rows[:2]), ("val", rows[2:])):
                (manifests / (name + ".json")).write_text(json.dumps(
                    {"schema_version": 1, "split": name, "records": selected}))
            configuration["dataset"].update(root=str(directory), manifest_dir=str(manifests))
            configuration["run_dir"] = str(directory / "run")
            config_path = directory / "config.json"
            config_path.write_text(json.dumps(configuration))
            arguments = ["--config", str(config_path), "--device", "cpu", "--val-batches", "1"]
            with patch("awmm.runtime.build_teachers", return_value=SyntheticTeachers()), redirect_stdout(StringIO()):
                train_awmm.main(arguments + ["--from-scratch", "--max-steps", "1"])
                last_path = directory / "run" / "last.pt"
                first = torch.load(last_path, weights_only=False)
                self.assertEqual(first["step"], 1)
                self.assertEqual(first["epoch"], 1)
                self.assertTrue((directory / "run" / "best.pt").is_file())
                train_awmm.main(arguments + ["--resume", str(last_path), "--max-steps", "2"])
                second = torch.load(last_path, weights_only=False)
            self.assertEqual(second["step"], 2)
            self.assertFalse(torch.equal(first["model"]["conv_first.0.weight"],
                                         second["model"]["conv_first.0.weight"]))
            for key in ("optimizer", "scheduler", "scaler", "rng", "manifest_digest"):
                self.assertIn(key, second)


if __name__ == "__main__":
    unittest.main()
