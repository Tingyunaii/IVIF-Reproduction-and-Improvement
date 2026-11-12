"""Evaluate existing legacy color outputs with the user's CC/PSNR/SCD/SSIM functions."""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

METRICS = ('CC', 'PSNR', 'SCD', 'SSIM')
COUNTS = {'MSRS': 361, 'M3FD': 300, 'FMB': 280, 'RoadScene': 221}
WEATHERS = ('Haze', 'Rain', 'Snow')
AWMM_COUNT = 50
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_functions(metric_path, ssim_path):
    """Compile the original selected definitions verbatim; unrelated metric imports are not executed."""
    tree = ast.parse(Path(metric_path).read_text(encoding='utf-8-sig'), filename=str(metric_path))
    active = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    wanted = ['corr2'] + [name + '_function' for name in METRICS]
    if any(name not in active for name in wanted):
        raise ValueError('Metric_torch.py must define corr2 and the four requested *_function entries')
    ssim_path = Path(ssim_path).resolve()
    if not ssim_path.is_file():
        raise FileNotFoundError('SSIM still needs the server implementation: ' + str(ssim_path))
    module_name = '_ivif_user_ssim_' + sha256(ssim_path)[:12]
    spec = importlib.util.spec_from_file_location(module_name, str(ssim_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    previous_path = list(sys.path)
    try:
        sys.path.insert(0, str(ssim_path.parent))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous_path
    if not callable(getattr(module, 'ssim', None)):
        raise ValueError('Expected ssim() in ' + str(ssim_path))
    selected = ast.Module(body=[active[name] for name in wanted], type_ignores=[])
    namespace = {'torch': torch, 'ssim': module.ssim}
    exec(compile(selected, str(metric_path), 'exec'), namespace)
    digest = hashlib.sha256(ast.dump(selected, include_attributes=False).encode()).hexdigest()
    return {name: namespace[name + '_function'] for name in METRICS}, digest


def ssim_arguments(images, layout):
    if layout == 'nchw':
        return [image[None, None] for image in images]
    if layout == 'hw':
        return images
    if layout not in ('numpy_hw', 'pil'):
        raise ValueError('Unknown SSIM input format: ' + str(layout))
    converted = []
    for image in images:
        array = image.detach().cpu().numpy()
        if (not np.isfinite(array).all() or array.min() < 0 or array.max() > 255
                or not np.equal(array, np.rint(array)).all()):
            raise ValueError('Image-based SSIM requires the original integral 8-bit grayscale values')
        # uint8 matters: torchvision ToTensor divides byte images by255, but not float arrays.
        array = array.astype(np.uint8)
        converted.append(Image.fromarray(array) if layout == 'pil' else array)
    return converted


def detect_ssim_layout(function, device, requested):
    # Negotiate representation only; normalization inside the user's backend remains its responsibility.
    a = torch.arange(64 * 64, device=device, dtype=torch.float32).reshape(64, 64) % 256
    b = torch.flip(a, [0])
    f = (a + b) / 2
    layouts = ('nchw', 'hw', 'numpy_hw', 'pil') if requested == 'auto' else (requested,)
    failures = {}
    with torch.no_grad():
        for layout in layouts:
            try:
                images = ssim_arguments([a, b, f], layout)
                value = float(function(*images))
                if not np.isfinite(value):
                    raise ValueError('SSIM returned a nonfinite value on the format probe')
                return layout, failures
            except (ValueError, RuntimeError, TypeError, IndexError, AttributeError) as exc:
                failures[layout] = '{}: {}'.format(type(exc).__name__, exc)
    raise RuntimeError('Cannot call server SSIM with tensors, uint8 arrays or PIL grayscale images: ' + str(failures))


def image_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(str(folder))
    result = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.stem in result:
                raise ValueError('Duplicate sample stem: ' + str(path))
            result[path.stem] = path
    return result


def audit_job(ir_dir, vi_dir, fused_dir, count):
    files = [image_files(folder) for folder in (ir_dir, vi_dir, fused_dir)]
    if not files[0] or any(set(items) != set(files[0]) for items in files[1:]):
        raise ValueError('IR/VIS/fused IDs must match exactly: ' + str((ir_dir, vi_dir, fused_dir)))
    if len(files[0]) != count:
        raise ValueError('Expected {} pairs, found {} in {}'.format(count, len(files[0]), ir_dir))
    return files


def read_gray(path, device):
    with Image.open(path) as image:
        if image.mode not in ('L', 'RGB', 'RGBA', 'P'):
            raise ValueError('Expected an 8-bit image: ' + str(path))
        array = np.array(image.convert('L'), dtype=np.float32)
    return torch.from_numpy(array).to(device)


def score_images(images, functions, ssim_layout):
    if any(image.ndim != 2 or image.shape != images[0].shape for image in images):
        raise ValueError('Images must be aligned 2D grayscale, with identical dimensions; no resizing')
    scores = {}
    for name in METRICS:
        arguments = ssim_arguments(images, ssim_layout) if name == 'SSIM' else images
        value = functions[name](*arguments)
        scores[name] = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
    return scores


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row['dataset'], row['subset'], row['reference_set'])
        groups.setdefault(key, []).append(row)
        if row['dataset'] == 'AWMM':
            groups.setdefault(('AWMM', 'ALL', row['reference_set']), []).append(row)
    result = []
    for (dataset, subset, reference), chosen in groups.items():
        means, counts = {}, {}
        for name in METRICS:
            values = np.array([row[name] for row in chosen], dtype=np.float64)
            with np.errstate(invalid='ignore'):
                means[name] = float(np.mean(values))  # Preserve NaN/Inf; never silently discard samples.
            counts[name + '_finite'] = int(np.isfinite(values).sum())
        result.append(dict(dataset=dataset, subset=subset, reference_set=reference,
                           model='legacy_color', count=len(chosen), **means, **counts))
    return result


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metric-file', default='/home/zj/IVIF_ZOO/Metric/Metric_torch.py')
    p.add_argument('--ssim-file', default=None, help='Defaults to ssim.py beside Metric_torch.py')
    p.add_argument('--ssim-layout', choices=('auto', 'nchw', 'hw', 'numpy_hw', 'pil'), default='auto')
    p.add_argument('--awmm-root', default='/home/zj/miniconda3/envs/p2fusion/project/Dataset/AWMM-100k')
    p.add_argument('--test-root', default='/home/zj/miniconda3/envs/p2fusion/project/Dataset/valsets')
    p.add_argument('--awmm-images', default='evaluation/awmm_legacy_comparison/images/legacy_color')
    p.add_argument('--benchmark-images', default='evaluation/legacy_color_benchmarks/images')
    p.add_argument('--out-dir', default='evaluation/legacy_color_metric_torch_four')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.threads < 1:
        p.error('--threads must be positive')
    output = Path(args.out_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Use a NEW output directory: ' + str(output))
    metric_path = Path(args.metric_file).expanduser().resolve()
    ssim_path = Path(args.ssim_file).expanduser().resolve() if args.ssim_file else metric_path.with_name('ssim.py')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    functions, selected_digest = load_functions(metric_path, ssim_path)
    layout, probe_failures = detect_ssim_layout(functions['SSIM'], device, args.ssim_layout)
    print('SSIM input format: ' + layout, flush=True)
    jobs = []
    for weather in WEATHERS:
        folder = Path(args.awmm_root) / 'AWMM-Text/Test' / weather
        for reference, (ir, vi) in {'clean_reference': ('gt_ir', 'gt_vi'), 'degraded_reference': ('ir', 'vi')}.items():
            files = audit_job(folder / ir, folder / vi, Path(args.awmm_images) / weather, AWMM_COUNT)
            jobs.append(('AWMM', weather, reference, files))
    for dataset, count in COUNTS.items():
        folder = Path(args.test_root) / dataset
        files = audit_job(folder / 'ir', folder / 'vi', Path(args.benchmark_images) / dataset, count)
        jobs.append((dataset, 'ALL', 'input_reference', files))
    metadata = {}
    for label, path in [('awmm', Path(args.awmm_images).resolve().parent.parent / 'run.json'),
                        ('benchmarks', Path(args.benchmark_images).resolve().parent / 'run.json')]:
        if path.is_file():
            metadata[label] = dict(path=str(path), sha256=sha256(path), run=json.loads(path.read_text(encoding='utf-8')))
    if set(metadata) == {'awmm', 'benchmarks'}:
        awmm_checkpoint = metadata['awmm']['run'].get('legacy_checkpoint', {}).get('sha256')
        benchmark_checkpoint = metadata['benchmarks']['run'].get('checkpoint', {}).get('sha256')
        if awmm_checkpoint and benchmark_checkpoint and awmm_checkpoint != benchmark_checkpoint:
            raise ValueError('AWMM and benchmark outputs came from DIFFERENT legacy checkpoints')
    run = dict(
        metric_file=dict(path=str(metric_path), sha256=sha256(metric_path), selected_ast_sha256=selected_digest),
        ssim_file=dict(path=str(ssim_path), sha256=sha256(ssim_path)), metrics=METRICS,
        function_loading='Original selected *_function definitions and corr2 compiled verbatim; unused imports skipped.',
        gray='Pillow RGB-to-L uint8, then float32 tensors in 0..255; no min-max normalization.',
        ssim_layout=layout, ssim_format_probe_failures=probe_failures,
        ssim_range='Original grayscale values: float32 0..255 for tensor backends; uint8 0..255 for NumPy/PIL backends. '
                   'Any ToTensor or normalization INSIDE the server ssim() is retained, with no parameter overrides.',
        PSNR='Original wrapper normalizes by255 and uses average MSE against the two references.',
        pairing='Exact filename stems; all expected samples, no intersection or resizing.',
        expected_counts=dict(AWMM_per_weather=AWMM_COUNT, **COUNTS),
        inference=False, device=str(device), torch=torch.__version__, source_runs=metadata,
        inputs={key: str(Path(getattr(args, key)).expanduser().resolve()) for key in
                ('awmm_root', 'test_root', 'awmm_images', 'benchmark_images')},
        nonfinite='Original NaN/Inf preserved, *_finite counts included; no zero replacement or silent sample exclusion.',
        script_sha256=sha256(__file__))
    output.mkdir(parents=True)
    (output / 'run.json').write_text(json.dumps(run, indent=2) + '\n', encoding='utf-8')
    rows = []
    with torch.no_grad():
        for dataset, subset, reference, files in jobs:
            names = sorted(files[0])
            for index, name in enumerate(names, 1):
                images = [read_gray(items[name], device) for items in files]
                score = score_images(images, functions, layout)
                rows.append(dict(dataset=dataset, subset=subset, reference_set=reference,
                                 model='legacy_color', sample_id=name, **score))
                if not all(np.isfinite(value) for value in score.values()):
                    print('Nonfinite metric preserved: {}/{}/{}/{} {}'.format(dataset, subset, reference, name, score), flush=True)
                if index == 1 or index % 50 == 0 or index == len(names):
                    print('{}/{}/{}: {}/{}'.format(dataset, subset, reference, index, len(names)), flush=True)
            write_csv(output / 'per_image.csv', rows)
            summaries = summarize(rows)
            write_csv(output / 'summary.csv', [r for r in summaries if r['subset'] == 'ALL'])
            write_csv(output / 'summary_by_subset.csv', summaries)
    print('\nDataset reference count CC PSNR SCD SSIM')
    for row in summarize(rows):
        if row['subset'] == 'ALL':
            print(row['dataset'], row['reference_set'], row['count'],
                  ' '.join('{:.6f}'.format(row[name]) for name in METRICS))
    print('Saved:', output)


if __name__ == '__main__':
    main()
