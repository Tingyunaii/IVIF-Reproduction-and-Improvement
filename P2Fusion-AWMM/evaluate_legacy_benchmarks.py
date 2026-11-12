"""Offline RGB/R legacy inference, color restoration and evaluation on four P2Fusion benchmarks."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

import evaluate_p2fusion as metrics
from awmm.preprocessing import input_mode
from compare_awmm import sha256, write_csv

COUNTS = dict(metrics.COUNTS)
SCORES = ('FQI_E', 'VIF', 'Qabf', 'MI')


def audit_dataset(root, dataset, expected_count):
    folder = Path(root) / dataset
    ir = metrics.image_files(folder / 'ir', match_stem=True)
    vi = metrics.image_files(folder / 'vi', match_stem=True)
    if not ir or set(ir) != set(vi):
        raise ValueError('IR/VIS sample IDs differ: ' + dataset)
    if len(ir) != expected_count:
        raise ValueError('{}: expected {} pairs, found {}'.format(dataset, expected_count, len(ir)))
    return ir, vi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='/home/zj/miniconda3/envs/p2fusion/project/IVIF-Reproduction-and-Improvement-awmm-adaptation/P2Fusion-AWMM/runs/awmm_bs32_300/best.pt')
    parser.add_argument('--test-root', default='/home/zj/miniconda3/envs/p2fusion/project/Dataset/valsets')
    parser.add_argument('--out-dir', default='evaluation/legacy_color_benchmarks')
    parser.add_argument('--datasets', nargs='+', choices=tuple(COUNTS), default=list(COUNTS))
    parser.add_argument('--vif-mode', choices=('mean', 'sum', 'fusion'), default='mean')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if len(args.datasets) != len(set(args.datasets)):
        parser.error('Do not repeat a dataset')
    project = Path(__file__).resolve().parent
    root = Path(args.test_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.out_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Use a NEW output directory to preserve previous results: ' + str(output))
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    pairs = {name: audit_dataset(root, name, COUNTS[name]) for name in args.datasets}
    import torch
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = payload.get('config')
    if (not config or input_mode(config) != 'rgb'
            or config.get('loss', {}).get('fusion_reference', 'first_channel') != 'first_channel'):
        raise ValueError('Use the first RGB/R experiment checkpoint, with first_channel fusion supervision')
    del payload
    output.mkdir(parents=True)
    run = dict(
        model='legacy_color', checkpoint=dict(path=str(checkpoint), sha256=sha256(checkpoint)),
        test_root=str(root), datasets={name: COUNTS[name] for name in args.datasets},
        input_mode='rgb', fusion_reference='first_channel', color_source='input',
        color='Observed VIS RGB + (predicted scalar - observed VIS R), clipped to [0,1], rounded to uint8.',
        gray='Pillow RGB-to-L uint8, matching previous evaluations', vif_mode=args.vif_mode,
        pairing='Exact filename stems; no intersection filtering, no resizing.',
        training=False, scripts_sha256={name: sha256(project / name) for name in
                                       ('evaluate_legacy_benchmarks.py', 'infer_awmm.py',
                                        'evaluate_p2fusion.py', 'awmm/preprocessing.py')},
        note='The first RGB/R experiment uses its own R-anchored color postprocessing, not original Y/CrCb. '
             'Same evaluator as prior runs. Dataset-specific paper values are references, not strict reproduction claims.')
    (output / 'run.json').write_text(json.dumps(run, indent=2) + '\n', encoding='utf-8')
    summaries, all_rows, reports = [], [], {}
    for dataset in args.datasets:
        folder = root / dataset
        images = output / 'images' / dataset
        report_dir = output / 'metrics' / dataset
        inference = [sys.executable, str(project / 'infer_awmm.py'), '--checkpoint', str(checkpoint),
                     '--ir-dir', str(folder / 'ir'), '--vis-dir', str(folder / 'vi'),
                     '--output', str(images), '--color-source', 'input', '--device', args.device]
        print('Inference + color restoration: ' + dataset, flush=True)
        subprocess.run(inference, cwd=str(project), check=True)
        predictions = metrics.image_files(images, match_stem=True)
        if set(predictions) != set(pairs[dataset][0]):
            raise ValueError('Fusion sample IDs differ: ' + dataset)
        for path in predictions.values():
            with Image.open(path) as image:
                if image.mode != 'RGB':
                    raise ValueError('Color restoration did not produce RGB: ' + str(path))
        evaluation = [sys.executable, str(project / 'evaluate_p2fusion.py'), '--dataset', dataset,
                      '--match-stem', '--ir-dir', str(folder / 'ir'), '--vi-dir', str(folder / 'vi'),
                      '--fused-dir', str(images), '--out-dir', str(report_dir),
                      '--expected-count', str(COUNTS[dataset]), '--vif-mode', args.vif_mode]
        print('Evaluation: ' + dataset, flush=True)
        subprocess.run(evaluation, cwd=str(project), check=True)
        report = json.loads((report_dir / 'summary.json').read_text(encoding='utf-8'))
        if report['dataset'] != dataset or report['count'] != COUNTS[dataset]:
            raise ValueError('Evaluation count/dataset mismatch: ' + dataset)
        if not all(np.isfinite(value) for value in report['means'].values()):
            raise ValueError('Nonfinite evaluation: ' + dataset)
        reports[dataset] = report
        summaries.append(dict(dataset=dataset, model='legacy_color', count=report['count'],
                              **{key: report['means'][key] for key in SCORES}))
        with (report_dir / 'per_image.csv').open(encoding='utf-8', newline='') as handle:
            dataset_rows = [dict(dataset=dataset, model='legacy_color', **row) for row in csv.DictReader(handle)]
        if len(dataset_rows) != COUNTS[dataset]:
            raise ValueError('Per-image report count mismatch: ' + dataset)
        all_rows.extend(dataset_rows)
        # Retain completed datasets if a later dataset fails.
        write_csv(output / 'summary.csv', summaries)
        write_csv(output / 'per_image.csv', all_rows)
        (output / 'summary.json').write_text(json.dumps(dict(protocol=run, results=reports), indent=2) + '\n', encoding='utf-8')
    print('\nDataset count FQI_E VIF Qabf MI')
    for row in summaries:
        print(row['dataset'], row['count'], ' '.join('{:.6f}'.format(row[key]) for key in SCORES))
    print('Saved:', output)


if __name__ == '__main__':
    main()
