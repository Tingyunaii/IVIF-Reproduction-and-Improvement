import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from awmm.data import AWMMFourImageDataset, make_awmm_splits, write_awmm_splits


class AWMMPairingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_image(self, path, value, shape=(20, 24)):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((*shape, 3), value, dtype=np.uint8)).save(path)

    def pair(self, base, stem, clean_ir, clean_vis, levels=("Light", "Medium", "Heavy")):
        for folder, value in (("ir", 180), ("gt_ir", clean_ir), ("gt_vi", clean_vis)):
            self.write_image(base / folder / (stem + ".png"), value)
        for level in levels:
            folder = "vi" if level == "mixed" else "vi_" + level
            self.write_image(base / folder / (stem + ".png"), 160)

    def fixture(self):
        for index in range(6):
            self.pair(self.root / "Rain" / "Scene_A" / "Train", str(index), 10 + index, 40 + index)
        self.pair(self.root / "AWMM-Text" / "Test" / "Rain", "test", 90, 120, ("mixed",))

    def test_groups_and_transitive_test_exclusion(self):
        self.fixture()
        # Different names share IR with test, and a second record shares VIS
        # with that record. The complete transitive component must be excluded.
        self.pair(self.root / "Haze" / "Scene_B" / "Train", "different", 90, 70)
        self.pair(self.root / "Snow" / "Scene_C" / "Train", "another", 80, 70)
        manifests, report = make_awmm_splits(self.root, seed=7)
        self.assertEqual(len(manifests["excluded"]["records"]), 6)
        groups = {}
        for split in ("train", "val", "test", "excluded"):
            rows = manifests[split]["records"]
            groups[split] = {row["group"] for row in rows}
        self.assertFalse(groups["train"] & groups["val"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["val"] & groups["test"])
        self.assertEqual(groups["excluded"], groups["test"])
        # All three severities for one clean pair stay on the same side.
        members = {}
        for split in ("train", "val"):
            for row in manifests[split]["records"]:
                members.setdefault(row["clean_ir_sha256"], set()).add(split)
        self.assertTrue(all(len(splits) == 1 for splits in members.values()))
        second, _ = make_awmm_splits(self.root, seed=7)
        self.assertEqual(manifests, second)
        self.assertEqual(report["counts"]["train"]["groups"], 5)

    def test_missing_pair_is_not_silently_discarded(self):
        self.fixture()
        (self.root / "Rain" / "Scene_A" / "Train" / "vi_Light" / "0.png").unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete four-way"):
            make_awmm_splits(self.root)

    def test_duplicate_stem_is_rejected(self):
        self.fixture()
        self.write_image(self.root / "Rain" / "Scene_A" / "Train" / "ir" / "0.jpg", 20)
        with self.assertRaisesRegex(ValueError, "Duplicate image stem"):
            make_awmm_splits(self.root)

    def test_dimensions_are_checked(self):
        self.fixture()
        self.write_image(self.root / "Rain" / "Scene_A" / "Train" / "vi_Medium" / "0.png", 100, (21, 24))
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            make_awmm_splits(self.root)

    def test_preserves_existing_manifests(self):
        self.fixture()
        out = self.root / "splits"
        write_awmm_splits(self.root, out)
        before = (out / "train.json").read_bytes()
        with self.assertRaises(FileExistsError):
            write_awmm_splits(self.root, out, seed=1)
        self.assertEqual(before, (out / "train.json").read_bytes())

    def test_identical_reference_in_text_and_scene_is_grouped(self):
        self.fixture()
        self.pair(self.root / "AWMM-Text" / "Train", "renamed", 10, 40, ("mixed",))
        manifests, _ = make_awmm_splits(self.root, source="all")
        rows = manifests["train"]["records"] + manifests["val"]["records"]
        same = [row for row in rows if row["clean_ir"] in (
            "AWMM-Text/Train/gt_ir/renamed.png", "Rain/Scene_A/Train/gt_ir/0.png")]
        self.assertEqual(len({row["group"] for row in same}), 1)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
    def test_synchronized_geometry_and_tensors(self):
        self.fixture()
        manifests, _ = make_awmm_splits(self.root)
        row = manifests["train"]["records"][0]
        pattern = np.arange(20 * 24 * 3, dtype=np.uint8).reshape(20, 24, 3)
        for key in ("ir", "vis", "clean_ir", "clean_vis"):
            Image.fromarray(pattern).save(self.root / row[key])
        dataset = AWMMFourImageDataset(self.root, manifests["train"], crop_size=32)
        random.seed(1)
        batch = dataset[0]
        for key in ("ir", "vis", "clean_ir", "clean_vis"):
            self.assertEqual(tuple(batch[key].shape), (3, 32, 32))
            self.assertTrue(batch[key].equal(batch["ir"]))
        validation = AWMMFourImageDataset(self.root, manifests["val"], crop_size=16, training=False)
        self.assertTrue(validation[0]["ir"].equal(validation[0]["ir"]))

    def test_test_manifest_cannot_be_used_for_training(self):
        self.fixture()
        manifests, _ = make_awmm_splits(self.root)
        with self.assertRaisesRegex(ValueError, "Training dataset"):
            AWMMFourImageDataset(self.root, manifests["test"])


if __name__ == "__main__":
    unittest.main()
