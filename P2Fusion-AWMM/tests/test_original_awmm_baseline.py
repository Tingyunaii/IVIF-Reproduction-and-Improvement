import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from awmm.data import write_awmm_splits
from awmm.runtime import make_model, make_criterion
from models.P2Fusion import p2fusion
from models.original_awmm_baseline import FIVE_TERMS
from tools.prepare_original_awmm_baseline import prepare
import train_awmm
import infer_awmm
import infer_original_p2fusion

ROOT = Path(__file__).resolve().parents[1]


class Teachers(torch.nn.Module):
    def forward(self, ir, vis):
        for image in (ir, vis):
            if not torch.equal(image[:, :1].expand_as(image), image):
                raise AssertionError('Teachers must get the same replicated-Y convention')
        return {'ir': ir[:, :1], 'vis': vis.new_full((vis.shape[0], 1), .7)}


class OriginalAWMMTests(unittest.TestCase):
    def config(self):
        config = json.loads((ROOT / 'configs/awmm_original_baseline_bs32_300.json').read_text())
        config['network'].update(num_blocks=1, Ex_depths=[1], input_resolution=[16, 16])
        return config

    def test_original_state_keys_and_forward_are_exact(self):
        torch.set_num_threads(1)
        config = self.config()
        original = p2fusion(**config['network']).eval()
        baseline = make_model(config['network'], 'original_p2fusion').eval()
        self.assertEqual(set(original.state_dict()), set(baseline.state_dict()))
        self.assertFalse(any('fdn_' in key or 'dwam_' in key or 'reconstructor' in key for key in baseline.state_dict()))
        baseline.load_state_dict(original.state_dict(), strict=True)
        images = [torch.rand(1, 1, 64, 64).repeat(1, 3, 1, 1) for _ in range(2)]
        score = []
        hook = original.patch_unembed_vis.register_forward_hook(lambda module, args, output: score.append(output[2]))
        with torch.no_grad():
            expected, actual = original(*images), baseline(*images)
        hook.remove()
        for name, index in [('fused', 0), ('prompt_ir', 1), ('prompt_vis', 2)]:
            torch.testing.assert_close(actual[name], expected[index], rtol=0, atol=0)
        torch.testing.assert_close(actual['vis_score'], score[0], rtol=0, atol=0)
        batch = {'ir': images[0], 'vis': images[1], 'clean_ir': images[0], 'clean_vis': images[1]}
        with torch.no_grad():
            loss = make_criterion(config, 'cpu')(actual, None, batch, Teachers()(*images))
        self.assertEqual(set(loss), set(FIVE_TERMS) | {'total'})

    def test_reference_configuration_is_reused_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = json.loads((ROOT / 'configs/awmm_original_y_bs32_300.json').read_text())
            path = root / 'adapted.pt'
            torch.save({'config': reference, 'model': {'dummy': torch.tensor(123)}}, path)
            baseline = prepare(path, root / 'baseline.json', root / 'new_run')
            for name in ('dataset', 'network', 'train', 'teachers'):
                self.assertEqual(baseline[name], reference[name])
            self.assertEqual(set(baseline['loss']['weights']), set(FIVE_TERMS))
            self.assertIsNone(baseline['init_checkpoint'])
            self.assertEqual(baseline['model_variant'], 'original_p2fusion')
            self.assertNotIn('model', baseline)
            self.assertNotEqual(train_awmm.resume_semantics(baseline), train_awmm.resume_semantics(reference))
            with self.assertRaises(FileExistsError):
                prepare(path, root / 'baseline.json', root / 'new_run')

    def test_train_resume_and_both_color_inference_entries(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, splits, run = root / 'data', root / 'splits', root / 'run'
            rng = np.random.RandomState(8)
            for folder, count, vis_key in [(data / 'Rain/Scene_A/Train', 5, 'vi_Medium'),
                                           (data / 'AWMM-Text/Test/Haze', 1, 'vi')]:
                for index in range(count):
                    ir, vis = [rng.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(2)]
                    for key, image in [('gt_ir', ir), ('gt_vi', vis), ('ir', (ir * .8).astype(np.uint8)),
                                       (vis_key, (vis * .7 + 20).astype(np.uint8))]:
                        path = folder / key / (str(index) + '.png')
                        path.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(image).save(path)
            write_awmm_splits(data, splits, val_ratio=.2)
            config = self.config()
            config['dataset'].update(root=str(data), manifest_dir=str(splits), crop_size=64)
            config['train'].update(batch_size=2, num_workers=0, epochs=2, amp=False,
                                    log_every=1, val_every=1, save_every=1)
            config['run_dir'] = str(run)
            path = root / 'config.json'
            path.write_text(json.dumps(config))
            with patch('awmm.runtime.build_teachers', return_value=Teachers()), contextlib.redirect_stdout(io.StringIO()):
                train_awmm.main(['--config', str(path), '--from-scratch', '--device', 'cpu', '--max-steps', '1'])
                train_awmm.main(['--config', str(path), '--resume', str(run / 'last.pt'), '--device', 'cpu', '--max-steps', '2'])
            state = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(state['step'], 2)
            self.assertEqual(state['config']['model_variant'], 'original_p2fusion')
            self.assertFalse(any(key.startswith(('fdn_', 'dwam_', 'clean_reconstructor', 'residual_reconstructor')) for key in state['model']))
            logs = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
            terms = next(row for row in logs if row['event'] == 'validation')
            self.assertTrue(set(FIVE_TERMS).issubset(terms))
            self.assertNotIn('consistency', terms)
            checkpoint = run / 'best.pt'
            base_args = ['--checkpoint', str(checkpoint), '--ir-dir', str(data / 'AWMM-Text/Test/Haze/ir'),
                         '--vis-dir', str(data / 'AWMM-Text/Test/Haze/vi'), '--device', 'cpu']
            with patch('sys.argv', ['infer_awmm.py'] + base_args + ['--output', str(root / 'adapted_entry')]), contextlib.redirect_stdout(io.StringIO()):
                infer_awmm.main()
            # Reference inference entry uses default architecture, so load the reduced test model directly.
            with patch('infer_original_p2fusion.p2fusion', side_effect=lambda: p2fusion(**config['network'])), patch('sys.argv', ['infer_original_p2fusion.py'] + base_args + ['--output', str(root / 'original_entry')]), contextlib.redirect_stdout(io.StringIO()):
                infer_original_p2fusion.main()
            with Image.open(root / 'adapted_entry/0.png') as a, Image.open(root / 'original_entry/0.png') as b:
                self.assertEqual(a.mode, 'RGB')
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


if __name__ == '__main__':
    unittest.main()
