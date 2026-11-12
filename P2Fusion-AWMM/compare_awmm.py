"""Paired AWMM author/adapted comparison; clean images are evaluation-only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import evaluate_p2fusion as metrics

WEATHERS = ('Haze', 'Rain', 'Snow')
MODELS = ('author', 'adapted')
METRICS = ('FQI_E', 'VIF', 'Qabf', 'MI')
REFERENCES = {'degraded_reference': ('ir', 'vi'), 'clean_reference': ('gt_ir', 'gt_vi')}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit_pairs(data_root, expected_count):
    result = {}
    for weather in WEATHERS:
        folder = Path(data_root) / 'AWMM-Text' / 'Test' / weather
        paths = {key: metrics.image_files(folder / key, match_stem=True)
                 for key in ('ir', 'vi', 'gt_ir', 'gt_vi')}
        keys = set(paths['ir'])
        if not keys or any(set(items) != keys for items in paths.values()):
            raise ValueError('Four-way test IDs differ for ' + weather)
        if len(keys) != expected_count:
            raise ValueError('{}: expected {} pairs, found {}'.format(weather, expected_count, len(keys)))
        result[weather] = paths
    return result


def evaluate_weather(weather, sources, predictions, vif_mode):
    names = sorted(sources['ir'])
    if any(set(items) != set(names) for items in predictions.values()):
        raise ValueError('Prediction IDs differ for ' + weather)
    rows, deltas = [], []
    for index, name in enumerate(names, 1):
        # No clean reference is passed to either inference process.
        images = {key: metrics.read_gray(items[name]) for key, items in sources.items()}
        fused = {model: metrics.read_gray(items[name]) for model, items in predictions.items()}
        for reference, source_keys in REFERENCES.items():
            scores = {}
            for model in MODELS:
                score = metrics.evaluate(images[source_keys[0]], images[source_keys[1]], fused[model], vif_mode)
                if not all(np.isfinite(value) for value in score.values()):
                    raise ValueError('Nonfinite metric: {}/{}/{}'.format(weather, name, model))
                scores[model] = score
                rows.append(dict(weather=weather, sample_id=name, reference_set=reference, model=model, **score))
            deltas.append(dict(weather=weather, sample_id=name, reference_set=reference,
                               **{key: scores['adapted'][key] - scores['author'][key] for key in METRICS}))
        if index == 1 or index % 10 == 0 or index == len(names):
            print('Evaluation {}/{}: {}/{}'.format(weather, len(names), index, name), flush=True)
    return rows, deltas


def summarize(rows):
    result = []
    for weather in (*WEATHERS, 'ALL'):
        for reference in REFERENCES:
            for model in MODELS:
                chosen = [r for r in rows if r['reference_set'] == reference and r['model'] == model
                          and (weather == 'ALL' or r['weather'] == weather)]
                if not chosen:
                    continue
                result.append(dict(weather=weather, reference_set=reference, model=model, count=len(chosen),
                                   **{key: float(np.mean([r[key] for r in chosen])) for key in METRICS}))
    return result


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='/home/zj/miniconda3/envs/p2fusion/project/Dataset/AWMM-100k')
    parser.add_argument('--author-checkpoint', default='/home/zj/miniconda3/envs/p2fusion/project/checkpoints/author_original/p2fusion_author_best_E.pth')
    parser.add_argument('--adapted-checkpoint', default='runs/awmm_original_y_bs32_300/best.pt')
    parser.add_argument('--out-dir', default='evaluation/awmm_author_vs_adapted')
    parser.add_argument('--expected-count', type=int, default=50, help='Pairs in each weather subset')
    parser.add_argument('--vif-mode', choices=('mean', 'sum', 'fusion'), default='mean')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.expected_count < 1:
        parser.error('--expected-count must be positive')
    project = Path(__file__).resolve().parent
    root = Path(args.data_root).expanduser().resolve()
    output = Path(args.out_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory to preserve previous images and reports: ' + str(output))
    checkpoints = {'author': Path(args.author_checkpoint).expanduser().resolve(),
                   'adapted': Path(args.adapted_checkpoint).expanduser().resolve()}
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(str(path))
    for filename in ('infer_original_p2fusion.py', 'infer_awmm.py'):
        if not (project / filename).is_file():
            raise FileNotFoundError(str(project / filename))
    pairs = audit_pairs(root, args.expected_count)
    import torch
    payload = torch.load(checkpoints['adapted'], map_location='cpu', weights_only=False)
    if payload.get('config', {}).get('dataset', {}).get('input_mode', 'rgb') != 'original_y':
        raise ValueError('Adapted checkpoint must come from the corrected original_y experiment')
    del payload
    output.mkdir(parents=True)
    provenance = {
        'dataset': 'AWMM-Text/Test', 'expected_count_per_weather': args.expected_count,
        'checkpoints': {key: {'path': str(path), 'sha256': sha256(path)} for key, path in checkpoints.items()},
        'vif_mode': args.vif_mode, 'gray': 'Pillow RGB-to-L uint8, matching evaluate_p2fusion.py',
        'reference_sets': REFERENCES, 'clean_inputs_used_in_inference': False,
        'color': 'fused Y + observed degraded VIS Cr/Cb, original OpenCV uint8 restoration',
        'scripts_sha256': {name: sha256(project / name) for name in
                           ('compare_awmm.py', 'evaluate_p2fusion.py', 'infer_awmm.py',
                            'infer_original_p2fusion.py', 'awmm/preprocessing.py')},
        'note': 'Reference metrics, not a verified official evaluator. No AWMM paper reference values. '
                'Degraded-reference fusion fidelity can reward retained weather artifacts. '
                'Clean-reference metrics measure fidelity to clean modalities, not fused ground truth.',
    }
    (output / 'run.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    all_rows, all_deltas = [], []
    for weather in WEATHERS:
        folder = root / 'AWMM-Text' / 'Test' / weather
        predictions = {}
        for model in MODELS:
            destination = output / 'images' / model / weather
            script = 'infer_original_p2fusion.py' if model == 'author' else 'infer_awmm.py'
            command = [sys.executable, str(project / script), '--checkpoint', str(checkpoints[model]),
                       '--ir-dir', str(folder / 'ir'), '--vis-dir', str(folder / 'vi'),
                       '--output', str(destination), '--device', args.device]
            if model == 'adapted':
                command += ['--color-source', 'original']
            print('Inference: {} {}'.format(model, weather), flush=True)
            subprocess.run(command, cwd=str(project), check=True)
            predictions[model] = metrics.image_files(destination, match_stem=True)
        rows, deltas = evaluate_weather(weather, pairs[weather], predictions, args.vif_mode)
        all_rows.extend(rows)
        all_deltas.extend(deltas)
        # Persist each completed weather even if a later weather fails.
        write_csv(output / (weather + '_per_image.csv'), rows)
    summary = summarize(all_rows)
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'per_image.csv', all_rows)
    write_csv(output / 'paired_deltas.csv', all_deltas)
    (output / 'summary.json').write_text(json.dumps(dict(protocol=provenance, results=summary), indent=2)+'\n')
    print('\nweather reference model count FQI_E VIF Qabf MI')
    for row in summary:
        print(row['weather'], row['reference_set'], row['model'], row['count'],
              ' '.join('{:.6f}'.format(row[key]) for key in METRICS))
    print('Saved:', output)


if __name__ == '__main__':
    main()
