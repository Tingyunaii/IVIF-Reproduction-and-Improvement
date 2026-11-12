#!/usr/bin/env python3
"""Create reproducible AWMM experimental splits without a CUDA dependency."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awmm.data import write_awmm_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="AWMM-100k dataset root")
    parser.add_argument("--out", default="splits/awmm", help="Output manifest directory")
    parser.add_argument("--source", choices=("scene", "text", "all"), default="scene")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of clean-reference groups")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--overwrite", action="store_true", help="Intentionally replace existing split files")
    args = parser.parse_args()
    report = write_awmm_splits(args.root, args.out, args.source, args.val_ratio, args.seed, args.overwrite)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Saved fixed manifests to {}".format(Path(args.out).resolve()))


if __name__ == "__main__":
    main()
