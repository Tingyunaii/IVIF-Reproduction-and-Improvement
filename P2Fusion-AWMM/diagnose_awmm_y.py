"""Separate decoder-Y quality from original P2Fusion color postprocessing, without training."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

import evaluate_p2fusion as metrics
from awmm.preprocessing import read_model_image
from compare_awmm import WEATHERS, MODELS, REFERENCES, audit_pairs, sha256, write_csv

SCORES = ('FQI_E', 'VIF', 'Qabf', 'MI', 'MI_ir', 'MI_vi')
PROTOCOLS = {
    'rgb_l': 'Prediction: original RGB restoration -> Pillow L; references: Pillow L (previous protocol).',
    'direct_y': 'Prediction: decoder uint8 Y before color restoration; references: SAME Pillow L as rgb_l.',
    'direct_y_opencv_reference': 'Prediction: same decoder uint8 Y; references: original OpenCV Y (training convention).',
}


def read_y(path):
    with Image.open(path) as image:
        if image.mode != 'L':
            raise ValueError('Direct Y must be an 8-bit grayscale image: ' + str(path))
        return np.asarray(image, dtype=np.float64)


def summarize(rows, groups, values=SCORES):
    buckets = defaultdict(list)
    for row in rows:
        for weather in (row['weather'], 'ALL'):
            key = (weather,) + tuple(row[name] for name in groups)
            buckets[key].append(row)
    return [dict(zip(('weather',) + groups, key), count=len(chosen),
                 **{name: float(np.mean([row[name] for row in chosen])) for name in values})
            for key, chosen in buckets.items()]


def evaluate_weather(weather, sources, predictions, vif_mode):
    names = sorted(sources['ir'])
    for model in MODELS:
        for representation in ('rgb', 'y'):
            if set(predictions[model][representation]) != set(names):
                raise ValueError('Prediction IDs differ: {}/{}/{}'.format(weather, model, representation))
    rows, model_deltas, effects, pixel_rows = [], [], [], []
    for index, name in enumerate(names, 1):
        pil_refs = {key: metrics.read_gray(items[name]) for key, items in sources.items()}
        cv_refs = {key: read_model_image(items[name], 'original_y')[:, :, 0].astype(np.float64)
                   for key, items in sources.items()}
        fused = {}
        for model in MODELS:
            rgb_l = metrics.read_gray(predictions[model]['rgb'][name])
            direct_y = read_y(predictions[model]['y'][name])
            if rgb_l.shape != direct_y.shape:
                raise ValueError('RGB/Y dimensions differ: {}/{}/{}'.format(weather, model, name))
            fused[model] = {'rgb_l': rgb_l, 'direct_y': direct_y,
                            'direct_y_opencv_reference': direct_y}
            difference = np.abs(rgb_l - direct_y)
            pixel_rows.append(dict(weather=weather, sample_id=name, model=model,
                                   mae_0_255=float(difference.mean()),
                                   max_abs_0_255=float(difference.max()),
                                   fraction_abs_gt_1=float(np.mean(difference > 1))))
        for reference, source_keys in REFERENCES.items():
            scores = {}
            for protocol in PROTOCOLS:
                refs = cv_refs if protocol == 'direct_y_opencv_reference' else pil_refs
                for model in MODELS:
                    score = metrics.evaluate(refs[source_keys[0]], refs[source_keys[1]], fused[model][protocol], vif_mode)
                    if not all(np.isfinite(value) for value in score.values()):
                        raise ValueError('Nonfinite score: {}/{}/{}/{}'.format(weather, name, model, protocol))
                    scores[(protocol, model)] = score
                    rows.append(dict(weather=weather, sample_id=name, reference_set=reference,
                                     protocol=protocol, model=model, **score))
                model_deltas.append(dict(weather=weather, sample_id=name, reference_set=reference, protocol=protocol,
                                         **{key: scores[(protocol, 'adapted')][key] - scores[(protocol, 'author')][key]
                                            for key in SCORES}))
            for model in MODELS:
                # Reference arrays are identical in these two terms: only prediction postprocessing changes.
                effects.append(dict(weather=weather, sample_id=name, reference_set=reference, model=model,
                                    **{key: scores[('direct_y', model)][key] - scores[('rgb_l', model)][key]
                                       for key in SCORES}))
        if index == 1 or index % 10 == 0 or index == len(names):
            print('Y diagnostic {}/{}: {}/{}'.format(weather, len(names), index, name), flush=True)
    return rows, model_deltas, effects, pixel_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='/home/zj/miniconda3/envs/p2fusion/project/Dataset/AWMM-100k')
    parser.add_argument('--author-checkpoint', default='/home/zj/miniconda3/envs/p2fusion/project/checkpoints/author_original/p2fusion_author_best_E.pth')
    parser.add_argument('--adapted-checkpoint', default='runs/awmm_original_y_bs32_300/best.pt')
    parser.add_argument('--out-dir', default='evaluation/awmm_y_diagnostic')
    parser.add_argument('--expected-count', type=int, default=50)
    parser.add_argument('--vif-mode', choices=('mean', 'sum', 'fusion'), default='mean')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.expected_count < 1:
        parser.error('--expected-count must be positive')
    project = Path(__file__).resolve().parent
    root = Path(args.data_root).expanduser().resolve()
    output = Path(args.out_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Use a NEW output directory to preserve previous results: ' + str(output))
    checkpoints = {'author': Path(args.author_checkpoint).expanduser().resolve(),
                   'adapted': Path(args.adapted_checkpoint).expanduser().resolve()}
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(str(path))
    pairs = audit_pairs(root, args.expected_count)
    import torch
    payload = torch.load(checkpoints['adapted'], map_location='cpu', weights_only=False)
    if payload.get('config', {}).get('dataset', {}).get('input_mode', 'rgb') != 'original_y':
        raise ValueError('Use the original_y-trained AWMM checkpoint, not the legacy RGB/R experiment')
    del payload
    output.mkdir(parents=True)
    scripts = ('diagnose_awmm_y.py', 'compare_awmm.py', 'evaluate_p2fusion.py',
               'infer_awmm.py', 'infer_original_p2fusion.py', 'awmm/preprocessing.py')
    provenance = dict(
        dataset='AWMM-Text/Test', data_root=str(root), expected_count_per_weather=args.expected_count,
        checkpoints={model: dict(path=str(path), sha256=sha256(path)) for model, path in checkpoints.items()},
        protocols=PROTOCOLS, reference_sets=REFERENCES, vif_mode=args.vif_mode,
        y_export='Decoder output clamped to [0,1], multiplied by 255, rounded to uint8; no per-image normalization.',
        color='Original OpenCV fused Y + observed degraded VIS Cr/Cb, unchanged.',
        clean_inputs_used_in_inference=False,
        model_delta_direction='adapted minus author, within the SAME protocol and reference set',
        postprocess_delta_direction='direct_y minus rgb_l, with identical Pillow L reference arrays',
        scripts_sha256={name: sha256(project / name) for name in scripts},
        note='No training or checkpoint writes. Clean modalities are references, not fused ground truth. '
             'Only direct_y versus rgb_l isolates prediction postprocessing; the OpenCV-reference protocol '
             'also changes reference conversion. These are reference metrics, not a verified official evaluator.')
    (output / 'run.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    all_rows, all_deltas, all_effects, all_pixels = [], [], [], []
    for weather in WEATHERS:
        folder = root / 'AWMM-Text' / 'Test' / weather
        predictions = {}
        for model in MODELS:
            rgb_dir = output / 'images' / 'rgb' / model / weather
            y_dir = output / 'images' / 'y' / model / weather
            script = 'infer_original_p2fusion.py' if model == 'author' else 'infer_awmm.py'
            command = [sys.executable, str(project / script), '--checkpoint', str(checkpoints[model]),
                       '--ir-dir', str(folder / 'ir'), '--vis-dir', str(folder / 'vi'),
                       '--output', str(rgb_dir), '--save-y-dir', str(y_dir), '--device', args.device]
            if model == 'adapted':
                command += ['--color-source', 'original']
            print('Inference: {} {} (RGB and pre-color Y from the SAME forward pass)'.format(model, weather), flush=True)
            subprocess.run(command, cwd=str(project), check=True)
            predictions[model] = {'rgb': metrics.image_files(rgb_dir, match_stem=True),
                                  'y': metrics.image_files(y_dir, match_stem=True)}
        rows, deltas, effects, pixels = evaluate_weather(weather, pairs[weather], predictions, args.vif_mode)
        all_rows.extend(rows)
        all_deltas.extend(deltas)
        all_effects.extend(effects)
        all_pixels.extend(pixels)
        write_csv(output / (weather + '_per_image.csv'), rows)
    summary = summarize(all_rows, ('reference_set', 'protocol', 'model'))
    effect_summary = summarize(all_effects, ('reference_set', 'model'))
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'per_image.csv', all_rows)
    write_csv(output / 'paired_deltas.csv', all_deltas)
    write_csv(output / 'postprocess_delta.csv', all_effects)
    write_csv(output / 'postprocess_delta_summary.csv', effect_summary)
    write_csv(output / 'pixel_changes.csv', all_pixels)
    (output / 'summary.json').write_text(json.dumps(dict(protocol=provenance, results=summary), indent=2) + '\n', encoding='utf-8')
    print('\nClean-reference scores (both models, all three protocols):')
    print('weather protocol model count FQI_E VIF Qabf MI MI_ir MI_vi')
    for row in summary:
        if row['reference_set'] == 'clean_reference':
            print(row['weather'], row['protocol'], row['model'], row['count'],
                  ' '.join('{:.6f}'.format(row[key]) for key in SCORES))
    print('\nPostprocess deltas: direct_y MINUS rgb_l, fixed Pillow L references. Positive = direct Y scores higher.')
    for row in effect_summary:
        if row['reference_set'] == 'clean_reference':
            print(row['weather'], row['model'], ' '.join('{:.6f}'.format(row[key]) for key in SCORES))
    print('Saved:', output)


if __name__ == '__main__':
    main()
