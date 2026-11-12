#!/usr/bin/env python3
"""Offline research evaluation; Python 3.8+, NumPy, SciPy and Pillow.

Piella QE: 8x8 uniform window, K=(1e-6,1e-6), Prewitt, alpha=1.
References and protocol limitations are in README.md.
VIFF adaptation: Copyright (c) 2012 Yu Han. Educational/research use only;
retain the accompanying original copyright notice and cite Han et al. (2013).
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import correlate2d, fftconvolve

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
PAPER = {'FQI_E': 0.850, 'VIF': 0.445, 'Qabf': 0.682, 'MI': 3.905}
PAPER_BY_DATASET = {
    'MSRS': PAPER,
    'M3FD': {'FQI_E': .743, 'VIF': .429, 'Qabf': .635, 'MI': 3.819},
    'FMB': {'FQI_E': .826, 'VIF': .448, 'Qabf': .677, 'MI': 4.030},
    'RoadScene': {'FQI_E': .617, 'VIF': .388, 'Qabf': .557, 'MI': 3.021},
}
COUNTS = {'MSRS': 361, 'M3FD': 300, 'FMB': 280, 'RoadScene': 221}


def filt(x, kernel):
    if min(x.shape) < max(kernel.shape):
        raise ValueError('Image too small for multiscale filtering')
    return fftconvolve(x, kernel[::-1, ::-1], mode='valid')


def local_stats(x, y, kernel):
    mx, my = filt(x, kernel), filt(y, kernel)
    vx = filt(x*x, kernel) - mx*mx
    vy = filt(y*y, kernel) - my*my
    cov = filt(x*y, kernel) - mx*my
    return mx, my, vx, vy, cov


def weighted_quality(a, b, f):
    window = np.ones((8, 8), dtype=np.float64) / 64.0
    c = (1e-6 * 255.0)**2
    ma, mf, va, vf, caf = local_stats(a, f, window)
    mb, _, vb, _, cbf = local_stats(b, f, window)
    # This matches the corrected 2019 metricPeilla.m regularization.
    zero = (va + vb == 0)
    wa, wb = va + zero*0.5, vb + zero*0.5
    denominator = wa + wb
    ratio = np.divide(wa, denominator, out=np.full_like(wa, 0.5),
                      where=np.abs(denominator) > 1e-20)
    qa = ((2*ma*mf+c)*(2*caf+c))/((ma*ma+mf*mf+c)*(va+vf+c))
    qb = ((2*mb*mf+c)*(2*cbf+c))/((mb*mb+mf*mf+c)*(vb+vf+c))
    salience = np.maximum(wa, wb)
    total = salience.sum()
    q = ratio*qa + (1-ratio)*qb
    return float((salience*q).sum()/total if total > 1e-20 else q.mean())


def prewitt(x):
    k = np.array([[1, 0, -1]]*3, dtype=np.float64)
    gx = correlate2d(x, k, mode='same', boundary='fill')
    gy = correlate2d(x, k.T, mode='same', boundary='fill')
    return np.hypot(gx, gy)


def qe(a, b, f):
    qw = weighted_quality(a, b, f)
    edge_qw = weighted_quality(prewitt(a), prewitt(b), prewitt(f))
    return qw*edge_qw, qw, edge_qw


def sobel(x):
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float64)
    gx = correlate2d(x, kx, mode='same', boundary='fill')
    gy = correlate2d(x, kx.T, mode='same', boundary='fill')
    angle = np.arctan(gy / np.where(gx == 0, 1e-5, gx))
    return np.hypot(gx, gy), angle


def qabf(a, b, f):
    gf, af = sobel(f)
    numerator, denominator = 0.0, 0.0
    for source in (a, b):
        gs, angle = sobel(source)
        strength = np.divide(np.minimum(gs, gf), np.maximum(gs, gf),
                             out=np.ones_like(gs), where=np.maximum(gs, gf)>0)
        orientation = 1 - np.abs(angle-af)*2/np.pi
        qg = 0.9994/(1+np.exp(-15*(strength-0.5)))
        qa = 0.9879/(1+np.exp(-22*(orientation-0.8)))
        numerator += float((qg*qa*gs).sum())
        denominator += float(gs.sum())
    return numerator/denominator if denominator > 0 else 0.0


def mi(x, y):
    x, y = x.astype(np.int64), y.astype(np.int64)
    counts = np.bincount((x*256+y).ravel(), minlength=65536).reshape(256,256)
    joint = counts/counts.sum()
    independent = joint.sum(1)[:,None]*joint.sum(0)[None,:]
    nz = joint > 0
    return float(np.sum(joint[nz]*np.log2(joint[nz]/independent[nz])))


def vif_maps(ref, dist, noise):
    maps = []
    for scale in range(1, 5):
        n = 2**(5-scale)+1
        axis = np.arange(n, dtype=np.float64)-(n-1)/2
        weights = np.exp(-axis*axis/(2*(n/5.0)**2))
        weights /= weights.sum()
        kernel = weights[:,None]*weights[None,:]
        if scale > 1:
            ref, dist = filt(ref, kernel)[::2,::2], filt(dist, kernel)[::2,::2]
        _, _, vr, vd, cov = local_stats(ref, dist, kernel)
        vr, vd = np.maximum(vr,0), np.maximum(vd,0)
        gain = cov/(vr+1e-10)
        residual = vd-gain*cov
        flat = vr < 1e-10
        gain[flat], residual[flat], vr[flat] = 0, vd[flat], 0
        flat_dist = vd < 1e-10
        gain[flat_dist], residual[flat_dist] = 0, 0
        negative = gain < 0
        residual[negative], gain[negative] = vd[negative], 0
        residual = np.maximum(residual,1e-10)
        num = np.log10(1+gain*gain*vr/(residual+noise))
        den = np.log10(1+vr/noise)
        maps.append((num, den, gain))
    return maps


def vif_pair(a, f):
    maps = vif_maps(a, f, 2.0)
    denominator = sum(float(d.sum()) for _,d,_ in maps)
    numerator = sum(float(n.sum()) for n,_,_ in maps)
    return numerator/denominator if denominator > 1e-12 else 0.0


def viff_han(a, b, f):
    ma, mb = vif_maps(a,f,0.005*255**2), vif_maps(b,f,0.005*255**2)
    values = []
    for (na,da,ga),(nb,db,gb) in zip(ma,mb):
        select_a = ga < gb
        n = np.where(select_a,na,nb)+1e-7
        d = np.where(select_a,da,db)+1e-7
        values.append(float(n.sum()/d.sum()))
    return float(np.dot(values,np.array([1,0,0.15,1])/2.15))


def read_gray(path):
    with Image.open(path) as image:
        if image.mode not in ('L','RGB','RGBA','P'):
            raise ValueError('Expected 8-bit image, got %s: %s' % (image.mode,path))
        return np.asarray(image.convert('L'),dtype=np.float64)


def evaluate(a,b,f,vif_mode='mean'):
    if a.shape != b.shape or a.shape != f.shape:
        raise ValueError('Image sizes differ; no automatic resizing is performed')
    if min(a.shape) < 64:
        raise ValueError('Images must be at least 64x64 for four-scale VIF')
    quality,qw,qwe = qe(a,b,f)
    va, vb = vif_pair(a,f), vif_pair(b,f)
    fusion = viff_han(a,b,f)
    vif = {'mean':(va+vb)/2, 'sum':va+vb, 'fusion':fusion}[vif_mode]
    mia, mib = mi(a,f), mi(b,f)
    return {'FQI_E':quality,'VIF':vif,'Qabf':qabf(a,b,f),'MI':mia+mib,
            'Qw':qw,'Qw_edges':qwe,'VIF_ir':va,'VIF_vi':vb,
            'VIF_mean':(va+vb)/2,'VIF_sum':va+vb,'VIFF_Han2013':fusion,
            'MI_ir':mia,'MI_vi':mib}


def image_files(folder, match_stem=False):
    p = Path(folder)
    if not p.is_dir():
        raise ValueError('Missing directory: '+str(p))
    result = {}
    for q in sorted(p.iterdir()):
        if not q.is_file() or q.suffix.lower() not in EXTENSIONS:
            continue
        key = q.stem if match_stem else q.name
        if key in result:
            raise ValueError('Duplicate pairing key %s: %s and %s' % (key,result[key],q))
        result[key] = q
    return result


def self_test():
    rng = np.random.RandomState(7)
    a = rng.randint(0,256,(96,96)).astype(np.float64)
    b = rng.randint(0,256,(96,96)).astype(np.float64)
    f = np.floor((a+b)/2)
    identity = evaluate(a,a,a)
    assert abs(identity['FQI_E']-1)<1e-6, identity
    assert abs(identity['VIF']-1)<1e-6, identity
    assert abs(identity['VIFF_Han2013']-1)<1e-6, identity
    assert abs(identity['MI']-2*mi(a,a))<1e-10
    ab, ba = evaluate(a,b,f), evaluate(b,a,f)
    for k in ('FQI_E','VIF','Qabf','MI','VIFF_Han2013'):
        assert abs(ab[k]-ba[k])<1e-8, (k,ab[k],ba[k])
    assert mi(np.zeros((64,64)),a[:64,:64])==0
    assert all(np.isfinite(x) for x in identity.values())
    print('Self-test passed: identity, source symmetry, MI zero-information cases.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=tuple(PAPER_BY_DATASET),default='MSRS')
    parser.add_argument('--match-stem',action='store_true',help='Match sample IDs across JPG/BMP input and PNG fusion; duplicate IDs are errors')
    parser.add_argument('--ir-dir',default='Dataset/valsets/MSRS/ir')
    parser.add_argument('--vi-dir',default='Dataset/valsets/MSRS/vi')
    parser.add_argument('--fused-dir',default='results/P2Fusion_MSRS_best_E')
    parser.add_argument('--out-dir',default='evaluation/MSRS_best_E')
    parser.add_argument('--expected-count',type=int,default=None)
    parser.add_argument('--vif-mode',choices=('mean','sum','fusion'),default='mean')
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    paper = PAPER_BY_DATASET[args.dataset]
    if args.expected_count is None:
        args.expected_count = COUNTS[args.dataset]
    if args.self_test:
        self_test()
        return
    files = [image_files(p,args.match_stem) for p in (args.ir_dir,args.vi_dir,args.fused_dir)]
    names = sorted(files[0])
    if not names or set(names)!=set(files[1]) or set(names)!=set(files[2]):
        union = set().union(*(set(items) for items in files))
        detail = {label: {'count':len(items),'missing_examples':sorted(union-set(items))[:8]}
                  for label,items in zip(('IR','VI','fused'),files)}
        raise ValueError('IR, VI and fused IDs must match exactly; no images discarded. '+str(detail))
    if args.expected_count and len(names)!=args.expected_count:
        raise ValueError('Expected %d pairs, found %d'%(args.expected_count,len(names)))
    out = Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'per_image.csv').exists() or (out/'summary.json').exists():
        raise ValueError('Output exists; use a new --out-dir to preserve prior results')
    rows=[]
    for i,name in enumerate(names,1):
        images=[read_gray(items[name]) for items in files]
        values=evaluate(*images,vif_mode=args.vif_mode)
        if not all(np.isfinite(x) for x in values.values()):
            raise ValueError('Nonfinite result: '+name)
        rows.append(dict(filename=name,**values))
        if i==1 or i%10==0 or i==len(names):
            print('[%d/%d] %s'%(i,len(names),name),flush=True)
    means={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='filename'}
    with (out/'per_image.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'dataset':args.dataset,'count':len(rows),'means':means,'paper_reference':paper,
             'difference_from_paper':{k:means[k]-v for k,v in paper.items()},
             'protocol':{'FQI_E':'Piella QE corrected 2019: UIQI-like K=1e-6, 8x8 uniform, Prewitt, alpha=1',
                         'VIF_selected':args.vif_mode,'VIF_pair_noise_variance':2.0,
                         'pairing':'stem' if args.match_stem else 'full filename',
                         'MI':'MI(IR,F)+MI(VI,F), 256 fixed bins, log2, no per-image normalization',
                         'gray':'Pillow RGB-to-L (BT.601 coefficients), uint8 0..255',
                         'Qabf':'Sobel zero padding, atan with epsilon 1e-5, Xydeas logistic parameters'},
             'official_p2fusion_evaluator':False,
             'note':'Reference implementation; P2Fusion exact grayscale/VIF/QE protocol remains unconfirmed.',
             'inputs':{k:str(Path(v).resolve()) for k,v in vars(args).items() if k.endswith('_dir')},
             'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['Metric | This run | Paper reference','--- | ---: | ---:']
    lines += ['%s | %.6f | %.3f'%(k,means[k],paper[k]) for k in paper]
    lines += ['', 'Reference protocol only; these differences do not establish strict paper reproduction.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines));print('Outputs:',out.resolve())


if __name__=='__main__':
    main()
