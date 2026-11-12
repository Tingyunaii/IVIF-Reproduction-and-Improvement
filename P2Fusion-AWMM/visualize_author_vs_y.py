"""Offline visual comparison of author P2Fusion and the corrected original_y AWMM model."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

from PIL import Image, ImageDraw, ImageFont

COUNTS = {'MSRS': 361, 'M3FD': 300, 'FMB': 280, 'RoadScene': 221}
WEATHERS = ('Rain', 'Haze', 'Snow')
AWMM_COUNT = 50
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
STYLE = 'body{font:16px sans-serif;background:#f4f5f7;color:#17202b;margin:24px}img{max-width:100%;height:auto}figure{background:white;padding:12px;margin:18px 0;border:1px solid #ddd}a{color:#075eb4}figcaption{margin:8px 0}'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def natural_key(value):
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower()) for part in re.split(r'(\d+)', value))


def image_index(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(str(folder))
    result = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.stem in result:
                raise ValueError('Duplicate sample ID: ' + str(path))
            result[path.stem] = path.resolve()
    return result


def validate_sets(indices, count, label):
    reference = set(next(iter(indices.values())))
    if len(reference) != count or any(set(items) != reference for items in indices.values()):
        raise ValueError('All {} panels must have the SAME {} sample IDs: {}'.format(label, count,
                         {key: len(items) for key, items in indices.items()}))
    return sorted(reference, key=natural_key)


def spread(names, limit):
    if limit <= 0 or limit >= len(names):
        return list(names)
    if limit == 1:
        return [names[len(names) // 2]]
    return [names[round(i * (len(names) - 1) / (limit - 1))] for i in range(limit)]


def font(size):
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/usr/share/fonts/dejavu/DejaVuSans.ttf'):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_comparison(panels, destination, title, width=320, footer=''):
    loaded = []
    for label, path in panels:
        with Image.open(path) as image:
            if label in ('Author P2Fusion', 'Current Y model') and image.mode != 'RGB':
                raise ValueError('Expected restored RGB model output: ' + str(path))
            loaded.append((label, image.convert('RGB').copy()))
    sizes = {image.size for _, image in loaded}
    if len(sizes) != 1:
        raise ValueError('Native panel dimensions differ; refusing to hide misalignment: ' + str(sizes))
    native_width, native_height = next(iter(sizes))
    cell_width = min(width, native_width)
    cell_height = max(1, round(native_height * cell_width / native_width))
    margin, top, label_height, bottom = 8, 32, 28, 28
    canvas = Image.new('RGB', (len(loaded) * (cell_width + margin) + margin,
                               top + label_height + cell_height + bottom), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 6), title, fill='#182330', font=font(17))
    for index, (label, image) in enumerate(loaded):
        x = margin + index * (cell_width + margin)
        draw.text((x, top), label, fill='#182330', font=font(15))
        if image.size != (cell_width, cell_height):
            image = image.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
        canvas.paste(image, (x, top + label_height))
    draw.text((margin, top + label_height + cell_height + 6), footer, fill='#444444', font=font(12))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=95, subsampling=0)
    return (native_width, native_height)


def render_overview(comparisons, destination, width=1600):
    images = []
    for path in comparisons:
        with Image.open(path) as image:
            image = image.convert('RGB')
            target_width = min(width, image.width)
            image = image.resize((target_width, round(image.height * target_width / image.width)), Image.Resampling.LANCZOS)
            images.append(image)
    canvas = Image.new('RGB', (max(image.width for image in images), sum(image.height for image in images)), 'white')
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
    canvas.save(destination, quality=95, subsampling=0)


def html_page(path, title, body):
    Path(path).write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                         '<meta name="viewport" content="width=device-width,initial-scale=1">'
                         '<title>' + html.escape(title) + '</title><style>' + STYLE + '</style>'
                         '<h1>' + html.escape(title) + '</h1>' + body + '</html>', encoding='utf-8')


def infer_pair(project, sources, selected, checkpoints, destination, device, full_names):
    input_dirs = {key: next(iter(sources[key].values())).parent for key in ('ir', 'vi')}
    if set(selected) != set(full_names):
        # Selected-only inference uses new symlinks to read original images; the originals are not edited.
        input_dirs = {key: destination / '_inputs' / key for key in ('ir', 'vi')}
        for key, folder in input_dirs.items():
            folder.mkdir(parents=True)
            for name in selected:
                path = sources[key][name]
                (folder / path.name).symlink_to(path)
    predictions = {}
    for model, script in (('author', 'infer_original_p2fusion.py'), ('current_y', 'infer_awmm.py')):
        folder = destination / model
        command = [sys.executable, str(project / script), '--checkpoint', str(checkpoints[model]),
                   '--ir-dir', str(input_dirs['ir']), '--vis-dir', str(input_dirs['vi']),
                   '--output', str(folder), '--device', device]
        if model == 'current_y':
            command += ['--color-source', 'original']
        print('Inference {}: {}'.format(model, destination.name), flush=True)
        subprocess.run(command, cwd=str(project), check=True)
        predictions[model] = image_index(folder)
    validate_sets(predictions, len(selected), str(destination))
    return predictions


def cached_awmm(cache_dir, checkpoint_hashes, groups):
    run_path = cache_dir / 'run.json'
    if not run_path.is_file():
        return None, 'No AWMM cache run.json'
    run = json.loads(run_path.read_text(encoding='utf-8'))
    if (run.get('dataset') != 'AWMM-Text/Test' or run.get('clean_inputs_used_in_inference') is not False
            or any(run.get('checkpoints', {}).get(old_name, {}).get('sha256') != checkpoint_hashes[model]
                   for model, old_name in (('author', 'author'), ('current_y', 'adapted')))):
        return None, 'AWMM cache metadata does not match the requested checkpoints'
    predictions = {}
    for group in groups:
        weather = group['subset']
        indices = {model: image_index(cache_dir / 'images/rgb' / old_name / weather)
                   for model, old_name in (('author', 'author'), ('current_y', 'adapted'))}
        validate_sets(dict(group['sources'], **indices), AWMM_COUNT, weather + ' cached results')
        predictions[weather] = indices
    return predictions, dict(path=str(run_path), sha256=sha256(run_path), run=run)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--author-checkpoint', default='/home/zj/miniconda3/envs/p2fusion/project/checkpoints/author_original/p2fusion_author_best_E.pth')
    p.add_argument('--y-checkpoint', default='runs/awmm_original_y_bs32_300/best.pt')
    p.add_argument('--awmm-root', default='/home/zj/miniconda3/envs/p2fusion/project/Dataset/AWMM-100k')
    p.add_argument('--test-root', default='/home/zj/miniconda3/envs/p2fusion/project/Dataset/valsets')
    p.add_argument('--awmm-cache', default='evaluation/awmm_y_diagnostic')
    p.add_argument('--out-dir', default='visualizations/author_vs_current_y')
    p.add_argument('--datasets', nargs='+', choices=('AWMM',) + tuple(COUNTS), default=['AWMM'] + list(COUNTS))
    p.add_argument('--limit', type=int, default=0, help='0=all; positive N selects N evenly spaced sample IDs per group before inference')
    p.add_argument('--panel-width', type=int, default=320)
    p.add_argument('--overview-rows', type=int, default=4)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.limit < 0 or args.panel_width < 80 or args.overview_rows < 1 or len(args.datasets) != len(set(args.datasets)):
        p.error('Use nonnegative limit, panel width >=80, positive overview rows and unique datasets')
    project = Path(__file__).resolve().parent
    output = Path(args.out_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Use a NEW output directory to preserve previous visualizations: ' + str(output))
    checkpoints = {'author': Path(args.author_checkpoint).expanduser().resolve(),
                   'current_y': Path(args.y_checkpoint).expanduser().resolve()}
    checkpoint_hashes = {key: sha256(path) for key, path in checkpoints.items()}
    import torch
    payload = torch.load(checkpoints['current_y'], map_location='cpu', weights_only=False)
    if payload.get('config', {}).get('dataset', {}).get('input_mode', 'rgb') != 'original_y':
        raise ValueError('Current Y comparison requires the corrected original_y checkpoint; legacy RGB/R is excluded')
    del payload
    groups = []
    if 'AWMM' in args.datasets:
        for weather in WEATHERS:
            folder = Path(args.awmm_root).expanduser().resolve() / 'AWMM-Text/Test' / weather
            sources = {key: image_index(folder / key) for key in ('ir', 'vi', 'gt_ir', 'gt_vi')}
            names = validate_sets(sources, AWMM_COUNT, weather)
            groups.append(dict(dataset='AWMM', subset=weather, slug='AWMM_' + weather,
                               sources=sources, names=names, selected=spread(names, args.limit)))
    for dataset in args.datasets:
        if dataset == 'AWMM':
            continue
        folder = Path(args.test_root).expanduser().resolve() / dataset
        sources = {key: image_index(folder / key) for key in ('ir', 'vi')}
        names = validate_sets(sources, COUNTS[dataset], dataset)
        groups.append(dict(dataset=dataset, subset='ALL', slug=dataset,
                           sources=sources, names=names, selected=spread(names, args.limit)))
    awmm_groups = [group for group in groups if group['dataset'] == 'AWMM']
    awmm_predictions, cache_info = cached_awmm(Path(args.awmm_cache).expanduser().resolve(), checkpoint_hashes, awmm_groups) if awmm_groups else (None, None)
    output.mkdir(parents=True)
    preview = output / 'preview'
    preview.mkdir()
    run = dict(checkpoints={key: dict(path=str(path), sha256=checkpoint_hashes[key]) for key, path in checkpoints.items()},
               models=('author', 'current_y'), old_rgb_model_included=False, awmm_cache=cache_info,
               selection='All sample IDs' if args.limit == 0 else '{} evenly spaced IDs per group, independent of metrics'.format(args.limit),
               overview_selection='Evenly spaced IDs among exported samples; not ranked by metric scores',
               color='Both models use original Y + observed input VIS Cr/Cb restoration',
               clean_images_used_in_inference=False,
               display='Same scaling for all aligned panels; no contrast/brightness/color enhancement. JPEG previews only; source PNGs untouched.',
               script_sha256=sha256(__file__), scripts_sha256={name: sha256(project / name) for name in
                   ('infer_original_p2fusion.py', 'infer_awmm.py', 'awmm/preprocessing.py')}, groups=[])
    index_body = '<p>Only author P²Fusion and current Y model. Clean AWMM images are reference panels; both models receive degraded inputs.</p>'
    preview_body = '<p>作者模型与当前 Y 模型；仅展示固定间隔抽取的概览样本，不按指标筛选。完整逐图结果见服务器主图库。</p>'
    manifest = []
    for group in groups:
        is_awmm = group['dataset'] == 'AWMM'
        if is_awmm and awmm_predictions is not None:
            predictions = awmm_predictions[group['subset']]
            prediction_origin = 'Reused checkpoint-matched AWMM RGB cache'
        else:
            predictions = infer_pair(project, group['sources'], group['selected'], checkpoints,
                                     output / 'predictions' / group['slug'], args.device, group['names'])
            prediction_origin = 'New degraded/input-only inference with specified checkpoints'
        folder = output / group['slug']
        folder.mkdir()
        records, comparisons = [], {}
        for name in group['selected']:
            sources = group['sources']
            panels = ([('Clean IR', sources['gt_ir'][name]), ('Clean VIS', sources['gt_vi'][name]),
                       ('Degraded IR', sources['ir'][name]), ('Degraded VIS', sources['vi'][name])]
                      if is_awmm else [('IR input', sources['ir'][name]), ('VIS input', sources['vi'][name])])
            panels += [('Author P2Fusion', predictions['author'][name]), ('Current Y model', predictions['current_y'][name])]
            destination = folder / (name + '.jpg')
            title = '{} / {} / {}'.format(group['dataset'], group['subset'], name) if is_awmm else group['dataset'] + ' / ' + name
            footer = 'Both models: degraded IR + VIS only. Clean IR/VIS: references.' if is_awmm else 'Both models: the same IR + VIS input. Original Y/CrCb color restoration.'
            size = render_comparison(panels, destination, title, args.panel_width, footer)
            comparisons[name] = destination
            records.append(dict(sample_id=name, native_size=size, comparison=str(destination.relative_to(output)),
                                panels={label: dict(path=str(path), sha256=sha256(path)) for label, path in panels}))
            manifest.append(dict(dataset=group['dataset'], subset=group['subset'], sample_id=name,
                                 author=str(predictions['author'][name]), current_y=str(predictions['current_y'][name]),
                                 comparison=str(destination.relative_to(output))))
        overview_ids = spread(group['selected'], args.overview_rows)
        overview = folder / 'overview.jpg'
        render_overview([comparisons[name] for name in overview_ids], overview)
        shutil.copyfile(overview, preview / (group['slug'] + '.jpg'))
        body = '<p><a href="../index.html">返回所有数据集</a></p>'
        for name in group['selected']:
            url = html.escape(comparisons[name].name, quote=True)
            body += '<figure><figcaption>' + html.escape(name) + '</figcaption><a href="' + url + '"><img loading="lazy" src="' + url + '"></a></figure>'
        html_page(folder / 'index.html', group['slug'], body)
        index_body += '<figure><h2><a href="{0}/index.html">{0}</a> ({1} images)</h2><a href="{0}/index.html"><img src="{0}/overview.jpg"></a></figure>'.format(group['slug'], len(group['selected']))
        preview_body += '<figure><h2>{0}</h2><p>Sample IDs: {1}</p><a href="{0}.jpg"><img src="{0}.jpg"></a></figure>'.format(group['slug'], ', '.join(html.escape(name) for name in overview_ids))
        run['groups'].append(dict(dataset=group['dataset'], subset=group['subset'], available_count=len(group['names']),
                                  exported_count=len(records), overview_ids=overview_ids, prediction_origin=prediction_origin,
                                  samples=records))
        (output / 'run.json').write_text(json.dumps(run, indent=2) + '\n', encoding='utf-8')
        print('Visualization {}: {} comparisons'.format(group['slug'], len(records)), flush=True)
    html_page(output / 'index.html', 'Author P2Fusion vs current Y', index_body)
    html_page(preview / 'index.html', '作者模型与当前 Y 模型：七组概览', preview_body)
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    (preview / 'README.txt').write_text('概览图按固定间隔选样。两模型仅接收输入图；AWMM干净图是参考。完整逐图图库在服务器输出目录的index.html。\n', encoding='utf-8')
    with zipfile.ZipFile(output / 'preview.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(preview.iterdir()):
            archive.write(path, arcname=path.name)
    print('Full gallery:', output / 'index.html')
    print('Portable overview:', output / 'preview.zip')


if __name__ == '__main__':
    main()
