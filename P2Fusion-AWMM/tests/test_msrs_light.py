import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from awmm.msrs_light import MSRSLightDataset, audit_manifests
from tools.prepare_msrs_pairs import prepare
import train_awmm
import infer_awmm

ROOT = Path(__file__).resolve().parents[1]


class Teachers(torch.nn.Module):
    def forward(self, ir, vis):
        if not torch.equal(ir[:, :1].expand_as(ir), ir):
            raise AssertionError('Teacher must receive repeated Y')
        return {'ir': ir[:, :1], 'vis': vis.new_full((vis.shape[0], 1), .7)}


class MSRSLightTests(unittest.TestCase):
    def fixture(self, root):
        rng = np.random.RandomState(18)
        for split, count in [('trainsets', 5), ('valsets', 1)]:
            for i in range(count):
                array = rng.randint(0, 256, (72, 80, 3), dtype=np.uint8)
                for modality in ('ir', 'vi'):
                    folder = root / split / 'MSRS' / modality
                    folder.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(array).save(folder / (str(i)+'.png'))
        prepare(root, root / 'splits', .25, 3407, 5, 1)

    def test_real_condition_pair_and_deterministic_unperturbed_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            audit_manifests(root / 'splits')
            options = dict(probability=1, gain=[.9,.9], contrast=[1,1], gamma=[1,1], offset=[0,0])
            train = MSRSLightDataset(root, root / 'splits/train.json', crop_size=64, light_pairing=options)
            batch = train[0]
            torch.testing.assert_close(batch['ir'], batch['clean_ir'] * .9)
            torch.testing.assert_close(batch['vis'], batch['clean_vis'] * .9)
            torch.testing.assert_close(batch['clean_ir'], batch['clean_vis'], rtol=0, atol=0)
            self.assertFalse(torch.equal(batch['ir'], batch['clean_ir']))
            val = MSRSLightDataset(root, root / 'splits/val.json', crop_size=64, training=False, light_pairing=options)
            a, b = val[0], val[0]
            for key in ('ir','vis','clean_ir','clean_vis'):
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            torch.testing.assert_close(a['ir'], a['clean_ir'], rtol=0, atol=0)
            original = json.loads((root / 'splits/test.json').read_text())
            val_doc = json.loads((root / 'splits/val.json').read_text())
            val_doc['records'][0]['group'] = original['records'][0]['group']
            (root / 'splits/val.json').write_text(json.dumps(val_doc))
            with self.assertRaisesRegex(ValueError, 'leakage'):
                audit_manifests(root / 'splits')

    def test_full_y_train_resume_infer_and_pairing_resume_guard(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            config = json.loads((ROOT / 'configs/msrs_y_light_bs32_300.json').read_text())
            config['dataset'].update(root=str(root), manifest_dir=str(root / 'splits'), crop_size=64)
            config['dataset']['light_pairing']['probability'] = 1
            config['network'].update(num_blocks=1, Ex_depths=[1], input_resolution=[16,16])
            config['train'].update(batch_size=2, num_workers=0, epochs=2, amp=False, val_every=1, save_every=1)
            config['run_dir'] = str(root / 'run')
            path = root / 'config.json'
            path.write_text(json.dumps(config))
            with patch('awmm.runtime.build_teachers', return_value=Teachers()), contextlib.redirect_stdout(io.StringIO()):
                train_awmm.main(['--config', str(path), '--from-scratch', '--device', 'cpu', '--max-steps', '1'])
                train_awmm.main(['--config', str(path), '--resume', str(root / 'run/last.pt'), '--device', 'cpu', '--max-steps', '2'])
            state = torch.load(root / 'run/last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(state['step'], 2)
            self.assertEqual(state['config']['dataset']['task'], 'msrs_light')
            self.assertIn('test.json', state['manifest_digest'])
            self.assertTrue(any(key.startswith('fdn_ir.') for key in state['model']))
            logs = [json.loads(line) for line in (root / 'run/metrics.jsonl').read_text().splitlines()]
            training = next(row for row in logs if row['event']=='training')
            self.assertGreater(training['consistency'], 0)
            validation = next(row for row in logs if row['event']=='validation')
            self.assertEqual(validation['consistency'], 0)
            self.assertIn('self_supervision', validation)
            config['dataset']['light_pairing']['probability'] = .5
            path.write_text(json.dumps(config))
            with patch('awmm.runtime.build_teachers', return_value=Teachers()), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'Resume configuration'):
                    train_awmm.main(['--config',str(path),'--resume',str(root/'run/last.pt'),'--device','cpu'])
            args = ['infer_awmm.py','--checkpoint',str(root/'run/best.pt'),
                    '--ir-dir',str(root/'valsets/MSRS/ir'),'--vis-dir',str(root/'valsets/MSRS/vi'),
                    '--output',str(root/'output'),'--device','cpu']
            with patch('sys.argv',args),contextlib.redirect_stdout(io.StringIO()):
                infer_awmm.main()
            with Image.open(root/'output/0.png') as image:
                self.assertEqual(image.mode,'RGB')
                self.assertEqual(image.size,(80,72))


if __name__ == '__main__':
    unittest.main()
