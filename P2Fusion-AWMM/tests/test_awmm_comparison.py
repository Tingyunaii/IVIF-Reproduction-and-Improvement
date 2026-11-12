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
import compare_awmm as comparison


class ComparisonTests(unittest.TestCase):
    def fixture(self, root):
        rng = np.random.RandomState(8)
        for weather in comparison.WEATHERS:
            for key in ('ir', 'vi', 'gt_ir', 'gt_vi'):
                folder = root/'AWMM-Text'/'Test'/weather/key
                folder.mkdir(parents=True)
                for i in range(2):
                    Image.fromarray(rng.randint(0, 256, (96, 96, 3), dtype=np.uint8)).save(folder/(str(i)+'.png'))

    def test_audit_rejects_missing_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            comparison.audit_pairs(root, 2)
            (root/'AWMM-Text/Test/Rain/gt_vi/0.png').unlink()
            with self.assertRaisesRegex(ValueError, 'Four-way'):
                comparison.audit_pairs(root, 2)

    def test_full_orchestration_protocols_and_no_clean_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            author, adapted = root/'author.pth', root/'adapted.pt'
            author.write_bytes(b'test only')
            torch.save({'config': {'dataset': {'input_mode': 'original_y'}}}, adapted)
            calls = []
            def fake_inference(command, cwd, check):
                calls.append(command)
                self.assertTrue(check)
                for value in command:
                    self.assertNotIn('/gt_ir', value)
                    self.assertNotIn('/gt_vi', value)
                destination = Path(command[command.index('--output')+1])
                destination.mkdir(parents=True)
                source = Path(command[command.index('--vis-dir')+1])
                for image in source.glob('*.png'):
                    with Image.open(image) as value:
                        value.save(destination/image.name)
            args = ['compare_awmm.py', '--data-root', str(root), '--author-checkpoint', str(author),
                    '--adapted-checkpoint', str(adapted), '--expected-count', '2',
                    '--out-dir', str(root/'results'), '--device', 'cpu']
            with patch('sys.argv', args), patch('compare_awmm.subprocess.run', side_effect=fake_inference), contextlib.redirect_stdout(io.StringIO()):
                comparison.main()
            self.assertEqual(len(calls), 6)
            result = json.loads((root/'results/summary.json').read_text())
            self.assertFalse(result['protocol']['clean_inputs_used_in_inference'])
            self.assertEqual(len(result['results']), 16)
            overall = [r for r in result['results'] if r['weather'] == 'ALL']
            self.assertTrue(all(r['count'] == 6 for r in overall))
            self.assertTrue((root/'results/paired_deltas.csv').is_file())
            # Both test-only prediction streams are identical, so their scores must match.
            for reference in comparison.REFERENCES:
                rows = [r for r in overall if r['reference_set'] == reference]
                for key in comparison.METRICS:
                    self.assertEqual(rows[0][key], rows[1][key])
            with patch('sys.argv', args):
                with self.assertRaises(FileExistsError):
                    comparison.main()
