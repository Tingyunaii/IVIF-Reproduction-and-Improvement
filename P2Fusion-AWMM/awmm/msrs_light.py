"""MSRS paired fusion with on-the-fly mild illumination conditions, not synthetic weather."""
import json
import math
from pathlib import Path
import random

from awmm.data import AWMMFourImageDataset

DEFAULT_LIGHT = dict(probability=.75, gain=[.95, 1.05], contrast=[.95, 1.05],
                     gamma=[.95, 1.05], offset=[-.025, .025])


def audit_manifests(directory):
    documents = {}
    required = {'id', 'group', 'ir', 'vis', 'ir_sha256', 'vis_sha256'}
    for split in ('train', 'val', 'test'):
        document = json.loads((Path(directory) / (split + '.json')).read_text())
        if (document.get('schema_version') != 1 or document.get('task') != 'normal_paired_fusion'
                or document.get('dataset') != 'MSRS' or document.get('split') != split
                or not document.get('records')):
            raise ValueError('Expected nonempty prepared MSRS ' + split + ' manifest')
        if any(required - set(row) for row in document['records']):
            raise ValueError('MSRS manifest lacks pairing/leakage audit fields')
        ids = [row['id'] for row in document['records']]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate MSRS IDs within ' + split)
        documents[split] = document
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        for key in ('id', 'group', 'ir_sha256', 'vis_sha256'):
            if {row[key] for row in documents[left]['records']} & {row[key] for row in documents[right]['records']}:
                raise ValueError('MSRS leakage in {} between {} and {}'.format(key, left, right))
    return documents


class MSRSLightDataset(AWMMFourImageDataset):
    def __init__(self, root, manifest, crop_size=256, training=True, input_mode='original_y', light_pairing=None):
        if input_mode != 'original_y':
            raise ValueError('MSRS full Y experiment requires original_y input convention')
        document = json.loads(Path(manifest).read_text()) if isinstance(manifest, (str, Path)) else manifest
        if document.get('task') != 'normal_paired_fusion' or document.get('dataset') != 'MSRS':
            raise ValueError('Use prepare_msrs_pairs.py manifests, not AWMM four-image lists')
        self.light = dict(DEFAULT_LIGHT)
        self.light.update(light_pairing or {})
        if set(self.light) != set(DEFAULT_LIGHT):
            raise ValueError('Unknown illumination augmentation option')
        if not 0 <= self.light['probability'] <= 1:
            raise ValueError('Illumination probability must be in [0,1]')
        for key in ('gain', 'contrast', 'gamma', 'offset'):
            bounds = self.light[key]
            if len(bounds) != 2 or not all(math.isfinite(float(v)) for v in bounds) or bounds[0] > bounds[1]:
                raise ValueError('Invalid illumination range: ' + key)
            if key != 'offset' and bounds[0] <= 0:
                raise ValueError('Gain, contrast and gamma must be positive')
        # Internal reuse of the synchronized geometry loader only. Student tensors are changed below.
        virtual = dict(document, records=[dict(row, clean_ir=row['ir'], clean_vis=row['vis']) for row in document['records']])
        super().__init__(root, virtual, crop_size, training, input_mode)

    def perturb(self, image):
        gain, contrast, gamma, offset = (random.uniform(*self.light[key]) for key in ('gain', 'contrast', 'gamma', 'offset'))
        mean = image.mean()
        result = (image - mean) * contrast + mean
        result = result.clamp(0, 1).pow(gamma)
        return (result * gain + offset).clamp(0, 1)

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        # Geometry is already synchronized across both modalities and their references.
        # Independent photometric parameters change appearance without moving a single pixel.
        if self.training and random.random() < self.light['probability']:
            batch['ir'] = self.perturb(batch['clean_ir'])
            batch['vis'] = self.perturb(batch['clean_vis'])
        return batch
