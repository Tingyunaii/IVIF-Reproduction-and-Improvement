"""Compare the first RGB/R scalar experiment before/after color restoration with existing AWMM baselines."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import evaluate_p2fusion as metrics
from awmm.preprocessing import input_mode
from compare_awmm import WEATHERS, REFERENCES, audit_pairs, sha256, write_csv
from diagnose_awmm_y import SCORES, summarize

MODELS = ('author', 'original_y', 'legacy_gray', 'legacy_color')
COMPARISONS = {
    'legacy_color_minus_legacy_gray': ('legacy_color', 'legacy_gray'),
    'legacy_color_minus_original_y': ('legacy_color', 'original_y'),
    'legacy_color_minus_author': ('legacy_color', 'author'),
}


def evaluate_weather(weather, sources, predictions, vif_mode):
    names = sorted(sources['ir'])
    if set(predictions) != set(MODELS) or any(set(items) != set(names) for items in predictions.values()):
        raise ValueError('Four prediction sets must match ALL test IDs: ' + weather)
    rows, deltas = [], []
    for index, name in enumerate(names, 1):
        images = {key: metrics.read_gray(items[name]) for key, items in sources.items()}
        fused = {model: metrics.read_gray(items[name]) for model, items in predictions.items()}
        for reference, source_keys in REFERENCES.items():
            scores = {}
            for model in MODELS:
                score = metrics.evaluate(images[source_keys[0]], images[source_keys[1]], fused[model], vif_mode)
                if not all(np.isfinite(value) for value in score.values()):
                    raise ValueError('Nonfinite score: {}/{}/{}'.format(weather, name, model))
                scores[model] = score
                rows.append(dict(weather=weather, sample_id=name, reference_set=reference, model=model, **score))
            for comparison, (left, right) in COMPARISONS.items():
                deltas.append(dict(weather=weather, sample_id=name, reference_set=reference, comparison=comparison,
                                   **{key: scores[left][key] - scores[right][key] for key in SCORES}))
        if index == 1 or index % 10 == 0 or index == len(names):
            print('Legacy comparison {}/{}: {}/{}'.format(weather, len(names), index, name), flush=True)
    return rows, deltas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='/home/zj/miniconda3/envs/p2fusion/project/Dataset/AWMM-100k')
    parser.add_argument('--legacy-checkpoint', default='/home/zj/miniconda3/envs/p2fusion/project/IVIF-Reproduction-and-Improvement-awmm-adaptation/P2Fusion-AWMM/runs/awmm_bs32_300/best.pt')
    parser.add_argument('--baseline-dir', default='evaluation/awmm_y_diagnostic',
                        help='Completed diagnose_awmm_y.py output, with author/adapted RGB images and run.json')
    parser.add_argument('--out-dir', default='evaluation/awmm_legacy_comparison')
    parser.add_argument('--expected-count', type=int, default=50)
    parser.add_argument('--vif-mode', choices=('mean', 'sum', 'fusion'), default='mean')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.expected_count < 1:
        parser.error('--expected-count must be positive')
    project = Path(__file__).resolve().parent
    root = Path(args.data_root).expanduser().resolve()
    checkpoint = Path(args.legacy_checkpoint).expanduser().resolve()
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    output = Path(args.out_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Use a NEW output directory to preserve old results: ' + str(output))
    if not checkpoint.is_file():
        raise FileNotFoundError('First RGB/R experiment checkpoint not found: ' + str(checkpoint))
    baseline_run_path = baseline_dir / 'run.json'
    baseline_run = json.loads(baseline_run_path.read_text(encoding='utf-8'))
    with (baseline_dir / 'summary.csv').open(encoding='utf-8', newline='') as handle:
        baseline_summary = {(row['weather'], row['reference_set'], row['model']): row
                            for row in csv.DictReader(handle) if row['protocol'] == 'rgb_l'}
    if (baseline_run.get('dataset') != 'AWMM-Text/Test'
            or baseline_run.get('vif_mode') != args.vif_mode
            or baseline_run.get('expected_count_per_weather') != args.expected_count
            or baseline_run.get('clean_inputs_used_in_inference') is not False
            or set(baseline_run.get('protocols', {})) != {'rgb_l', 'direct_y', 'direct_y_opencv_reference'}):
        raise ValueError('Baseline must be a completed Y diagnostic with matching count/VIF and degraded-only inference')
    if Path(baseline_run['data_root']).resolve() != root:
        raise ValueError('Baseline data root differs; use the SAME test data')
    if baseline_run['scripts_sha256']['evaluate_p2fusion.py'] != sha256(project / 'evaluate_p2fusion.py'):
        raise ValueError('Evaluator changed since baseline generation; compare with identical metric algorithms')
    pairs = audit_pairs(root, args.expected_count)
    baseline_predictions = {}
    baseline_image_hashes = {}
    for weather in WEATHERS:
        baseline_predictions[weather] = {}
        for model, previous_name in (('author', 'author'), ('original_y', 'adapted')):
            images = metrics.image_files(baseline_dir / 'images/rgb' / previous_name / weather, match_stem=True)
            if set(images) != set(pairs[weather]['ir']):
                raise ValueError('Incomplete baseline RGB predictions: {}/{}'.format(model, weather))
            baseline_predictions[weather][model] = images
            baseline_image_hashes[weather + '/' + model] = {name: sha256(path) for name, path in images.items()}
    import torch
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = payload.get('config')
    if not config or input_mode(config) != 'rgb':
        raise ValueError('Use the FIRST RGB/R checkpoint; an original_y checkpoint is a different experiment')
    fusion_reference = config.get('loss', {}).get('fusion_reference', 'first_channel')
    if fusion_reference != 'first_channel':
        raise ValueError('Expected the first RGB/R experiment with first_channel fusion supervision')
    del payload
    output.mkdir(parents=True)
    provenance = dict(
        dataset='AWMM-Text/Test', data_root=str(root), expected_count_per_weather=args.expected_count,
        legacy_checkpoint=dict(path=str(checkpoint), sha256=sha256(checkpoint),
                               input_mode='rgb', fusion_reference=fusion_reference),
        reused_baseline=dict(path=str(baseline_dir), run_sha256=sha256(baseline_run_path),
                             checkpoints=baseline_run['checkpoints'], image_sha256=baseline_image_hashes),
        models={
            'author': 'Existing author RGB: original Y + observed VIS Cr/Cb.',
            'original_y': 'Existing corrected AWMM RGB: original Y + observed VIS Cr/Cb.',
            'legacy_gray': 'First RGB/R-trained model, raw scalar output saved as grayscale; NOT decoder Y.',
            'legacy_color': 'Same first RGB/R model, observed VIS RGB + (predicted scalar - observed VIS R), clipped to [0,1].',
        },
        comparisons={key: '{} minus {}'.format(*value) for key, value in COMPARISONS.items()},
        reference_sets=REFERENCES, gray='Pillow RGB-to-L uint8, identical for all four prediction streams',
        vif_mode=args.vif_mode, clean_inputs_used_in_inference=False,
        scripts_sha256={name: sha256(project / name) for name in
                        ('compare_awmm_legacy.py', 'infer_awmm.py', 'evaluate_p2fusion.py',
                         'compare_awmm.py', 'diagnose_awmm_y.py', 'awmm/preprocessing.py')},
        note='No retraining or checkpoint changes. Color restoration follows each experiment input/target convention. '
             'RGB/R scalar is not relabelled as original Y. Color postprocessing can retain degraded input colors '
             'and clip RGB; this is a comparison of complete pipelines, not an isolated architecture ablation. '
             'Clean modalities are evaluation references, not fused ground truth. Reference evaluator only.')
    (output / 'run.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    all_rows, all_deltas = [], []
    for weather in WEATHERS:
        folder = root / 'AWMM-Text/Test' / weather
        predictions = dict(baseline_predictions[weather])
        for model, color_source in (('legacy_gray', 'none'), ('legacy_color', 'input')):
            destination = output / 'images' / model / weather
            command = [sys.executable, str(project / 'infer_awmm.py'), '--checkpoint', str(checkpoint),
                       '--ir-dir', str(folder / 'ir'), '--vis-dir', str(folder / 'vi'),
                       '--output', str(destination), '--color-source', color_source, '--device', args.device]
            print('Inference: {} {}'.format(model, weather), flush=True)
            subprocess.run(command, cwd=str(project), check=True)
            predictions[model] = metrics.image_files(destination, match_stem=True)
        rows, deltas = evaluate_weather(weather, pairs[weather], predictions, args.vif_mode)
        all_rows.extend(rows)
        all_deltas.extend(deltas)
        write_csv(output / (weather + '_per_image.csv'), rows)
    summary = summarize(all_rows, ('reference_set', 'model'))
    delta_summary = summarize(all_deltas, ('reference_set', 'comparison'))
    for row in summary:
        if row['model'] in ('author', 'original_y'):
            previous_name = 'adapted' if row['model'] == 'original_y' else 'author'
            previous = baseline_summary[(row['weather'], row['reference_set'], previous_name)]
            if (int(previous['count']) != row['count']
                    or any(not np.isclose(float(previous[key]), row[key], rtol=1e-10, atol=1e-10) for key in SCORES)):
                raise ValueError('Reused baseline scores changed; check image/reference integrity: ' + str(row))
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'per_image.csv', all_rows)
    write_csv(output / 'paired_deltas.csv', all_deltas)
    write_csv(output / 'delta_summary.csv', delta_summary)
    (output / 'summary.json').write_text(json.dumps(dict(protocol=provenance, results=summary), indent=2) + '\n', encoding='utf-8')
    print('\nClean-reference scores: weather model count FQI_E VIF Qabf MI MI_ir MI_vi')
    for row in summary:
        if row['reference_set'] == 'clean_reference':
            print(row['weather'], row['model'], row['count'], ' '.join('{:.6f}'.format(row[key]) for key in SCORES))
    print('Saved:', output)


if __name__ == '__main__':
    main()
