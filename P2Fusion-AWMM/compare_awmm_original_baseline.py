"""Evaluate the AWMM-trained original network against author weights and the existing Y model."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import evaluate_p2fusion as metrics
from compare_awmm import WEATHERS, REFERENCES, audit_pairs, sha256, write_csv
from diagnose_awmm_y import SCORES, summarize

MODELS = ('author', 'original_awmm', 'current_y')
COMPARISONS = {'current_y_minus_original_awmm': ('current_y', 'original_awmm'),
               'original_awmm_minus_author': ('original_awmm', 'author')}


def evaluate_weather(weather, sources, predictions, vif_mode):
    names = sorted(sources['ir'])
    if set(predictions) != set(MODELS) or any(set(items) != set(names) for items in predictions.values()):
        raise ValueError('All three model predictions must match every test ID: ' + weather)
    rows, deltas = [], []
    for index, name in enumerate(names, 1):
        images = {key: metrics.read_gray(items[name]) for key, items in sources.items()}
        fused = {model: metrics.read_gray(items[name]) for model, items in predictions.items()}
        for reference, keys in REFERENCES.items():
            scores = {}
            for model in MODELS:
                score = metrics.evaluate(images[keys[0]], images[keys[1]], fused[model], vif_mode)
                if not all(np.isfinite(value) for value in score.values()):
                    raise ValueError('Nonfinite score: {}/{}/{}'.format(weather, model, name))
                scores[model] = score
                rows.append(dict(weather=weather, sample_id=name, reference_set=reference, model=model, **score))
            for comparison, (left, right) in COMPARISONS.items():
                deltas.append(dict(weather=weather, sample_id=name, reference_set=reference, comparison=comparison,
                                   **{key: scores[left][key] - scores[right][key] for key in SCORES}))
        if index == 1 or index % 10 == 0 or index == len(names):
            print('Evaluate {}: {}/{}'.format(weather, index, len(names)), flush=True)
    return rows, deltas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='runs/p2fusion_original_awmm_y_bs32_300/best.pt')
    p.add_argument('--comparison-dir', default='/home/zj/miniconda3/envs/p2fusion/project/P2Fusion-AWMM-originalY/evaluation/awmm_y_diagnostic')
    p.add_argument('--data-root', default='/home/zj/miniconda3/envs/p2fusion/project/Dataset/AWMM-100k')
    p.add_argument('--out-dir', default='evaluation/original_awmm_vs_current_y')
    p.add_argument('--expected-count', type=int, default=50)
    p.add_argument('--vif-mode', choices=('mean', 'sum', 'fusion'), default='mean')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.expected_count < 1:
        p.error('--expected-count must be positive')
    project = Path(__file__).resolve().parent
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    previous_dir = Path(args.comparison_dir).expanduser().resolve()
    root = Path(args.data_root).expanduser().resolve()
    output = Path(args.out_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Use a NEW output directory: ' + str(output))
    previous_run = json.loads((previous_dir / 'run.json').read_text(encoding='utf-8'))
    if (previous_run.get('dataset') != 'AWMM-Text/Test' or previous_run.get('clean_inputs_used_in_inference') is not False
            or previous_run.get('expected_count_per_weather') != args.expected_count
            or previous_run.get('vif_mode') != args.vif_mode or Path(previous_run['data_root']).resolve() != root):
        raise ValueError('Reuse the completed Y diagnostic with matching data/count/VIF and degraded-only inference')
    if previous_run['scripts_sha256']['evaluate_p2fusion.py'] != sha256(project / 'evaluate_p2fusion.py'):
        raise ValueError('Evaluator differs from the existing comparison')
    with (previous_dir / 'summary.csv').open(encoding='utf-8', newline='') as handle:
        previous_scores = {(row['weather'], row['reference_set'], row['model']): row
                           for row in csv.DictReader(handle) if row['protocol'] == 'rgb_l'}
    import torch
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = payload.get('config', {})
    if config.get('model_variant') != 'original_p2fusion' or config.get('dataset', {}).get('input_mode') != 'original_y':
        raise ValueError('Use the newly AWMM-trained original_p2fusion/original_y baseline checkpoint')
    reference = config.get('baseline_reference', {}).get('sha256')
    if reference and reference != previous_run['checkpoints']['adapted']['sha256']:
        raise ValueError('Current Y comparison weights differ from the baseline reference experiment')
    del payload
    pairs = audit_pairs(root, args.expected_count)
    cached = {}
    for weather in WEATHERS:
        cached[weather] = {}
        for model, old_name in (('author', 'author'), ('current_y', 'adapted')):
            items = metrics.image_files(previous_dir / 'images/rgb' / old_name / weather, match_stem=True)
            if set(items) != set(pairs[weather]['ir']):
                raise ValueError('Incomplete cached comparison: {}/{}'.format(weather, model))
            cached[weather][model] = items
    output.mkdir(parents=True)
    provenance = dict(
        baseline_checkpoint=dict(path=str(checkpoint), sha256=sha256(checkpoint)),
        baseline_config=config, comparison_dir=str(previous_dir),
        comparison_run_sha256=sha256(previous_dir / 'run.json'), reused_checkpoints=previous_run['checkpoints'],
        dataset='AWMM-Text/Test', data_root=str(root), expected_count_per_weather=args.expected_count,
        models=MODELS, reference_sets=REFERENCES, vif_mode=args.vif_mode,
        color='All three use original fused Y + observed degraded VIS Cr/Cb restoration',
        gray='Pillow RGB-to-L uint8, identical to preceding reference evaluation',
        clean_inputs_used_in_inference=False,
        comparisons={key: '{} minus {}'.format(*pair) for key, pair in COMPARISONS.items()},
        scripts_sha256={name: sha256(project / name) for name in
                        ('compare_awmm_original_baseline.py', 'infer_awmm.py', 'evaluate_p2fusion.py',
                         'awmm/runtime.py', 'awmm/preprocessing.py', 'models/original_awmm_baseline.py')},
        note='Author: original published weights; original_awmm: original architecture trained on AWMM; '
             'current_y: adapted model with FDN/DWAM. Clean references are not fused ground truth. '
             'Reference evaluator, not verified official metrics. Best checkpoint selection uses five vs eight validation loss terms.')
    (output / 'run.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    all_rows, all_deltas = [], []
    for weather in WEATHERS:
        folder = root / 'AWMM-Text/Test' / weather
        destination = output / 'images/original_awmm' / weather
        command = [sys.executable, str(project / 'infer_awmm.py'), '--checkpoint', str(checkpoint),
                   '--ir-dir', str(folder / 'ir'), '--vis-dir', str(folder / 'vi'), '--output', str(destination),
                   '--color-source', 'original', '--device', args.device]
        print('Original AWMM baseline inference: ' + weather, flush=True)
        subprocess.run(command, cwd=str(project), check=True)
        predictions = dict(cached[weather], original_awmm=metrics.image_files(destination, match_stem=True))
        rows, deltas = evaluate_weather(weather, pairs[weather], predictions, args.vif_mode)
        all_rows.extend(rows)
        all_deltas.extend(deltas)
        write_csv(output / (weather + '_per_image.csv'), rows)
    summary = summarize(all_rows, ('reference_set', 'model'))
    for row in summary:
        if row['model'] in ('author', 'current_y'):
            old_name = 'adapted' if row['model'] == 'current_y' else 'author'
            old = previous_scores[row['weather'], row['reference_set'], old_name]
            if int(old['count']) != row['count'] or any(not np.isclose(float(old[key]), row[key], rtol=1e-10, atol=1e-10) for key in SCORES):
                raise ValueError('Reused image/reference scores changed; verify data integrity: ' + str(row))
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'per_image.csv', all_rows)
    write_csv(output / 'paired_deltas.csv', all_deltas)
    write_csv(output / 'delta_summary.csv', summarize(all_deltas, ('reference_set', 'comparison')))
    print('\nClean-reference summary: weather model count FQI_E VIF Qabf MI MI_ir MI_vi')
    for row in summary:
        if row['reference_set'] == 'clean_reference':
            print(row['weather'], row['model'], row['count'], ' '.join('{:.6f}'.format(row[key]) for key in SCORES))
    print('Saved:', output)


if __name__ == '__main__':
    main()
