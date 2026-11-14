"""Audit MSRS paired data and split training candidates; never train a model or fabricate weather pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from awmm.data import _image_index, _decoded_rgb_hash, _DisjointSet


def prepare(root, output, val_ratio=.1, seed=3407, expected_train=1083, expected_test=361):
    root, output = Path(root).expanduser().resolve(), Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Use a NEW manifest directory: ' + str(output))
    if not 0 < val_ratio < 1:
        raise ValueError('val_ratio must be between 0 and 1')
    records, candidates = [], []
    for split, folder, expected in [('train_candidate', root / 'trainsets/MSRS', expected_train),
                                     ('test', root / 'valsets/MSRS', expected_test)]:
        indices = {key: _image_index(folder / key) for key in ('ir', 'vi')}
        if set(indices['ir']) != set(indices['vi']) or len(indices['ir']) != expected:
            raise ValueError('{} needs {} exactly matched pairs'.format(split, expected))
        for name in sorted(indices['ir']):
            row = dict(id=split + '/' + name, source_split=split,
                       ir=str(indices['ir'][name].resolve()), vis=str(indices['vi'][name].resolve()))
            records.append(row)
            if split == 'train_candidate':
                candidates.append(row)
    union = _DisjointSet(len(records))
    owners, cache = {}, {}
    for index, row in enumerate(records):
        sizes = []
        for key in ('ir', 'vis'):
            digest, size = _decoded_rgb_hash(row[key], cache)
            row[key + '_sha256'] = digest
            sizes.append(size)
            fingerprint = (key, digest)
            if fingerprint in owners:
                union.union(index, owners[fingerprint])
            else:
                owners[fingerprint] = index
        if sizes[0] != sizes[1]:
            raise ValueError('IR/VIS sizes differ: ' + row['id'])
        row['width'], row['height'] = sizes[0]
    components = {}
    for index in range(len(records)):
        components.setdefault(union.find(index), []).append(index)
    test_components = {union.find(index) for index in range(len(candidates), len(records))}
    for indices in components.values():
        fingerprints = sorted({key + ':' + records[i][key + '_sha256'] for i in indices for key in ('ir', 'vis')})
        group = hashlib.sha256('\n'.join(fingerprints).encode()).hexdigest()
        for index in indices:
            records[index]['group'] = group
    usable = sorted({row['group'] for index, row in enumerate(candidates) if union.find(index) not in test_components})
    if len(usable) < 2:
        raise ValueError('Need at least two training groups outside the held-out test set')
    random.Random(seed).shuffle(usable)
    val_groups = set(usable[:min(len(usable)-1, max(1, round(len(usable)*val_ratio)))])
    splits = dict(train=[], val=[], test=records[len(candidates):], excluded=[])
    for index, row in enumerate(candidates):
        if union.find(index) in test_components:
            row['exclusion_reason'] = 'exact_reference_component_overlaps_test'
            splits['excluded'].append(row)
        else:
            splits['val' if row['group'] in val_groups else 'train'].append(row)
    metadata = dict(schema_version=1, task='normal_paired_fusion', dataset='MSRS', root=str(root),
                    seed=seed, requested_val_group_ratio=val_ratio,
                    pair_semantics='Original registered IR/VIS only; no degraded/clean weather pairs are claimed.',
                    limitations='Exact decoded RGB grouping cannot detect near frames, recompression or every same-scene image.')
    output.mkdir(parents=True)
    for split, rows in splits.items():
        (output / (split + '.json')).write_text(json.dumps(dict(metadata, split=split, records=rows), indent=2) + '\n')
    report = dict(metadata, counts={split: dict(samples=len(rows), groups=len({row['group'] for row in rows}))
                                   for split, rows in splits.items()})
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='/home/zj/miniconda3/envs/p2fusion/project/Dataset')
    p.add_argument('--out', default='data_splits/msrs_y')
    p.add_argument('--val-ratio', type=float, default=.1)
    p.add_argument('--seed', type=int, default=3407)
    args = p.parse_args()
    print(json.dumps(prepare(args.root, args.out, args.val_ratio, args.seed), indent=2))


if __name__ == '__main__':
    main()
