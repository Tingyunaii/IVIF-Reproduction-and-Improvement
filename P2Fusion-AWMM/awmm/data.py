"""AWMM four-image loading and reproducible, clean-reference grouped splits.

Grouping matches exact decoded RGB clean images (including image dimensions).
It prevents exact-reference leakage across weather/strength/file-name variants;
it cannot identify shifted views, recompressed images, or nearby video frames as
the same scene. A manually reviewed scene/sequence split remains preferable.
"""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_KEYS = ("ir", "vis", "clean_ir", "clean_vis")
MANIFEST_VERSION = 1


def _image_index(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("Required image directory does not exist: {}".format(directory))
    result = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = path.stem
        if key in result:
            raise ValueError("Duplicate image stem {!r} in {}".format(key, directory))
        result[key] = path
    if not result:
        raise ValueError("No supported images in {}".format(directory))
    return result


def _paired_records(root, folder, vis_folder, weather, level, source):
    indices = {
        "ir": _image_index(folder / "ir"),
        "vis": _image_index(folder / vis_folder),
        "clean_ir": _image_index(folder / "gt_ir"),
        "clean_vis": _image_index(folder / "gt_vi"),
    }
    all_stems = set().union(*(set(index) for index in indices.values()))
    missing = {key: sorted(all_stems - set(index)) for key, index in indices.items()}
    missing = {key: values for key, values in missing.items() if values}
    if missing:
        short = {key: values[:10] for key, values in missing.items()}
        raise ValueError("Incomplete four-way pairing in {} ({}): {}".format(folder, vis_folder, short))
    records = []
    for stem in sorted(all_stems):
        row = {key: index[stem].relative_to(root).as_posix() for key, index in indices.items()}
        row.update({
            "id": "{}/{}/{}".format(folder.relative_to(root).as_posix(), level, stem),
            "weather": weather,
            "level": level,
            "source": source,
        })
        records.append(row)
    return records


def discover_awmm(root, source="scene"):
    """Return candidate training records and independently held-out test records."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("AWMM root does not exist: {}".format(root))
    if source not in {"scene", "text", "all"}:
        raise ValueError("source must be scene, text, or all")
    candidates = []
    if source in {"text", "all"}:
        folder = root / "AWMM-Text" / "Train"
        if folder.is_dir():
            candidates.extend(_paired_records(root, folder, "vi", "mixed", "mixed", "text"))
        elif source == "text":
            raise ValueError("AWMM-Text/Train was not found under {}".format(root))
    if source in {"scene", "all"}:
        for weather_folder in sorted(root.iterdir()):
            if not weather_folder.is_dir() or weather_folder.name.lower() not in {"haze", "rain", "snow"}:
                continue
            for scene in sorted(weather_folder.glob("Scene_*")):
                folder = scene / "Train"
                if not folder.is_dir():
                    continue
                vis_folders = sorted(path for path in folder.glob("vi_*") if path.is_dir())
                if not vis_folders:
                    raise ValueError("No vi_Light/vi_Medium/vi_Heavy directories in {}".format(folder))
                for vis_folder in vis_folders:
                    level = vis_folder.name[len("vi_"):]
                    if level.lower() not in {"light", "medium", "heavy"}:
                        continue
                    candidates.extend(_paired_records(
                        root, folder, vis_folder.name, weather_folder.name, level, "scene"))
    if not candidates:
        raise ValueError("No training candidates discovered for source={!r} under {}".format(source, root))

    tests = []
    test_root = root / "AWMM-Text" / "Test"
    if test_root.is_dir():
        for folder in sorted(test_root.iterdir()):
            if folder.is_dir() and folder.name.lower() in {"haze", "rain", "snow"}:
                tests.extend(_paired_records(root, folder, "vi", folder.name, "mixed", "test"))
    return candidates, tests


def _safe_image_path(root, relative):
    root = Path(root).expanduser().resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError("Manifest image path escapes AWMM root: {}".format(relative))
    return path


def _decoded_rgb_hash(path, cache):
    path = str(path)
    if path not in cache:
        with Image.open(path) as image:
            image = image.convert("RGB")
            value = hashlib.sha256()
            value.update("RGB:{}x{}:".format(*image.size).encode("ascii"))
            value.update(image.tobytes())
            cache[path] = (value.hexdigest(), image.size)
    return cache[path]


class _DisjointSet:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def make_awmm_splits(root, source="scene", val_ratio=0.1, seed=3407, require_test=True):
    """Build grouped manifests; no files are written by this function.

    Identical clean IR OR identical clean VIS connects records, transitively.
    Candidate groups connected to any official test record are excluded from
    both training and validation, even if only one clean modality matches.
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be strictly between 0 and 1")
    root = Path(root).expanduser().resolve()
    candidates, tests = discover_awmm(root, source)
    if require_test and not tests:
        raise ValueError("No AWMM-Text/Test weather pairs found; cannot screen official-test overlap")
    records = candidates + tests
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Duplicate manifest sample IDs")
    sets = _DisjointSet(len(records))
    hash_owner = {}
    cache = {}
    for index, row in enumerate(records):
        sizes = {}
        for key in IMAGE_KEYS:
            path = _safe_image_path(root, row[key])
            if key.startswith("clean_"):
                digest, size = _decoded_rgb_hash(path, cache)
                row[key + "_sha256"] = digest
                fingerprint = (key, digest)
                if fingerprint in hash_owner:
                    sets.union(index, hash_owner[fingerprint])
                else:
                    hash_owner[fingerprint] = index
                sizes[key] = size
            else:
                with Image.open(path) as image:
                    sizes[key] = image.size
        if len(set(sizes.values())) != 1:
            raise ValueError("Four-image dimensions differ for {}: {}".format(row["id"], sizes))
        row["width"], row["height"] = sizes["ir"]

    components = {}
    for index in range(len(records)):
        components.setdefault(sets.find(index), []).append(index)
    test_components = {sets.find(index) for index in range(len(candidates), len(records))}
    group_names = {}
    for component, indices in components.items():
        fingerprints = sorted({
            key + ":" + records[index][key + "_sha256"]
            for index in indices for key in ("clean_ir", "clean_vis")
        })
        group_names[component] = hashlib.sha256("\n".join(fingerprints).encode("ascii")).hexdigest()
        for index in indices:
            records[index]["group"] = group_names[component]

    usable_groups = sorted(group_names[key] for key in components if key not in test_components)
    if len(usable_groups) < 2:
        raise ValueError("Need at least two non-test clean-reference groups; found {}".format(len(usable_groups)))
    random.Random(seed).shuffle(usable_groups)
    val_count = min(len(usable_groups) - 1, max(1, int(round(len(usable_groups) * val_ratio))))
    val_groups = set(usable_groups[:val_count])
    splits = {"train": [], "val": [], "test": tests, "excluded": []}
    for index, row in enumerate(candidates):
        if sets.find(index) in test_components:
            row["exclusion_reason"] = "clean_reference_group_overlaps_official_test"
            splits["excluded"].append(row)
        elif row["group"] in val_groups:
            splits["val"].append(row)
        else:
            splits["train"].append(row)

    metadata = {
        "schema_version": MANIFEST_VERSION,
        "root_hint": str(root),
        "source": source,
        "seed": seed,
        "requested_val_group_ratio": val_ratio,
        "grouping": "transitive identical decoded RGB clean IR OR clean VIS (dimensions included)",
    }
    manifests = {}
    for split, rows in splits.items():
        manifests[split] = dict(metadata, split=split, records=rows)
    report = dict(metadata)
    report["counts"] = {
        split: {
            "samples": len(rows),
            "groups": len({row["group"] for row in rows}),
            "weather_level": dict(sorted(Counter(row["weather"] + "/" + row["level"] for row in rows).items())),
        }
        for split, rows in splits.items()
    }
    report["limitations"] = [
        "An experimental seeded split, not an official AWMM protocol.",
        "Exact RGB hashes do not detect different viewpoints, recompression, near-duplicate frames, or all same-scene images.",
        "Only selected candidate sources are scanned; text and scene pools may overlap and source=all can repeat variants.",
        "Official-test references are used only for overlap screening, never for training targets or model selection.",
    ]
    return manifests, report


def write_awmm_splits(root, out, source="scene", val_ratio=0.1, seed=3407, overwrite=False):
    out = Path(out).expanduser().resolve()
    names = ("train", "val", "test", "excluded", "report")
    existing = [str(out / (name + ".json")) for name in names if (out / (name + ".json")).exists()]
    if existing and not overwrite:
        raise FileExistsError("Existing manifests are preserved; use --overwrite only intentionally: {}".format(existing))
    manifests, report = make_awmm_splits(root, source=source, val_ratio=val_ratio, seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    for name, document in dict(manifests, report=report).items():
        destination = out / (name + ".json")
        temporary = out / (name + ".json.tmp")
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
    return report


class AWMMFourImageDataset:
    """RGB [0,1] tensors with synchronized geometry; torch is imported lazily.

    A fixed-size validation center crop is deterministic. crop_size=None loads
    whole images (use batch_size=1 when image dimensions differ).
    """
    def __init__(self, root, manifest, crop_size=256, training=True):
        self.root = Path(root).expanduser().resolve()
        if isinstance(manifest, (str, Path)):
            manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
        if manifest.get("schema_version") != MANIFEST_VERSION:
            raise ValueError("Unsupported AWMM manifest schema_version")
        if training and manifest.get("split") != "train":
            raise ValueError("Training dataset must use the train manifest")
        self.records = manifest["records"]
        self.crop_size = crop_size
        self.training = bool(training)
        if crop_size is not None and (not isinstance(crop_size, int) or crop_size < 1):
            raise ValueError("crop_size must be a positive integer or None")
        if not self.records:
            raise ValueError("AWMM manifest contains no records")
        for row in self.records:
            for key in IMAGE_KEYS:
                path = _safe_image_path(self.root, row[key])
                if not path.is_file():
                    raise FileNotFoundError(str(path))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch

        row = self.records[index]
        images = {}
        for key in IMAGE_KEYS:
            with Image.open(_safe_image_path(self.root, row[key])) as image:
                images[key] = np.array(image.convert("RGB"), dtype=np.uint8)
        shapes = {key: image.shape for key, image in images.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError("Four-image dimensions differ for {}: {}".format(row["id"], shapes))
        height, width = images["ir"].shape[:2]
        size = self.crop_size
        if size is not None:
            pad_bottom, pad_right = max(0, size - height), max(0, size - width)
            if pad_bottom or pad_right:
                images = {
                    key: np.pad(image, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="edge")
                    for key, image in images.items()
                }
                height, width = images["ir"].shape[:2]
            if self.training:
                top, left = random.randint(0, height - size), random.randint(0, width - size)
            else:
                top, left = (height - size) // 2, (width - size) // 2
            images = {key: image[top:top + size, left:left + size] for key, image in images.items()}
        if self.training:
            flip_horizontal, flip_vertical = random.random() < 0.5, random.random() < 0.5
            if flip_horizontal:
                images = {key: image[:, ::-1] for key, image in images.items()}
            if flip_vertical:
                images = {key: image[::-1] for key, image in images.items()}
        result = {
            key: torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0)
            for key, image in images.items()
        }
        result["id"] = row["id"]
        return result
