"""Derive a separate original-P2Fusion AWMM experiment from the completed Y experiment config."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch

FIVE = ('intensity', 'ssim', 'gradient', 'distill_ir', 'distill_vis')


def prepare(reference, output, run_dir):
    reference, output = Path(reference).expanduser().resolve(), Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Preserve existing config; use a new --out: ' + str(output))
    payload = torch.load(reference, map_location='cpu', weights_only=False)
    config = copy.deepcopy(payload.get('config'))
    if not config or config.get('dataset', {}).get('input_mode') != 'original_y':
        raise ValueError('Reference must be the corrected original_y experiment checkpoint')
    if config.get('model_variant', 'awmm') != 'awmm':
        raise ValueError('Reference must be the adapted Y model, not an existing original baseline')
    if config['train'].get('validation_batch_limit', 0):
        raise ValueError('Reference used limited validation; resolve the validation protocol before a matched baseline')
    if Path(run_dir).resolve() == Path(config['run_dir']).resolve():
        raise ValueError('Use a NEW baseline run directory')
    original_loss = copy.deepcopy(config['loss'])
    config['model_variant'] = 'original_p2fusion'
    config['loss']['weights'] = {name: original_loss['weights'][name] for name in FIVE}
    config['loss'].pop('residual_weight', None)
    config['run_dir'] = str(run_dir)
    config['init_checkpoint'] = None
    config.pop('initialization', None)
    config['baseline_reference'] = dict(
        checkpoint=str(reference), sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        note='Only config is reused. Fusion weights are NOT loaded. Original network, five common losses, no FDN/DWAM/auxiliary heads.',
        removed_loss_weights={name: value for name, value in original_loss['weights'].items() if name not in FIVE},
        selection='Baseline best.pt uses its five-term validation total; adapted best.pt used its eight-term total. Final metrics must use the same test evaluator.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-checkpoint', default='runs/awmm_original_y_bs32_300/best.pt')
    p.add_argument('--out', default='configs/awmm_original_baseline_matched.json')
    p.add_argument('--run-dir', default='runs/p2fusion_original_awmm_y_bs32_300')
    args = p.parse_args()
    config = prepare(args.reference_checkpoint, args.out, args.run_dir)
    print(json.dumps(dict(config=args.out, model_variant=config['model_variant'],
                         input_mode=config['dataset']['input_mode'], batch_size=config['train']['batch_size'],
                         epochs=config['train']['epochs'], milestones=config['train']['milestones'],
                         five_weights=config['loss']['weights'], run_dir=config['run_dir']), indent=2))


if __name__ == '__main__':
    main()
