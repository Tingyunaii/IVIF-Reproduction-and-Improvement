import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from awmm.preprocessing import original_color_restore
import diagnose_awmm_y as diagnostic
import evaluate_p2fusion as metrics


class YDiagnosticTests(unittest.TestCase):
    def test_saturated_color_roundtrip_can_change_decoder_y(self):
        # A chroma-heavy reference and a new Y can clip RGB channels; recovering L is not exact.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'vis.png'
            Image.new('RGB', (96, 96), (255, 0, 0)).save(path)
            y = torch.full((1, 1, 96, 96), 0.8)
            restored = original_color_restore(y, path)
            rgb_l = np.asarray(Image.fromarray(restored).convert('L'))
            self.assertGreater(float(np.abs(rgb_l.astype(float) - 204).mean()), 10)

    def test_full_protocols_same_references_and_delta_directions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.RandomState(32)
            for weather in diagnostic.WEATHERS:
                for key in ('ir', 'vi', 'gt_ir', 'gt_vi'):
                    folder = root / 'AWMM-Text/Test' / weather / key
                    folder.mkdir(parents=True)
                    Image.fromarray(rng.randint(0, 256, (96, 96, 3), dtype=np.uint8)).save(folder / '0.png')
            author, adapted = root / 'author.pth', root / 'adapted.pt'
            author.write_bytes(b'fixture only')
            torch.save({'config': {'dataset': {'input_mode': 'original_y'}}}, adapted)
            calls = []

            def inference(command, cwd, check):
                calls.append(command)
                self.assertTrue(check)
                self.assertFalse(any('/gt_ir' in word or '/gt_vi' in word for word in command))
                rgb_dir = Path(command[command.index('--output') + 1])
                y_dir = Path(command[command.index('--save-y-dir') + 1])
                rgb_dir.mkdir(parents=True)
                y_dir.mkdir(parents=True)
                # The same outputs for both models; this test checks protocol, not model quality.
                y = ((np.indices((96, 96)).sum(axis=0) * 5) % 256).astype(np.uint8)
                vis = Path(command[command.index('--vis-dir') + 1]) / '0.png'
                tensor = torch.from_numpy(y.astype(np.float32) / 255).view(1, 1, 96, 96)
                Image.fromarray(original_color_restore(tensor, vis)).save(rgb_dir / '0.png')
                Image.fromarray(y).save(y_dir / '0.png')

            args = ['diagnose_awmm_y.py', '--data-root', str(root), '--author-checkpoint', str(author),
                    '--adapted-checkpoint', str(adapted), '--expected-count', '1',
                    '--out-dir', str(root / 'out'), '--device', 'cpu']
            with patch('sys.argv', args), patch('diagnose_awmm_y.subprocess.run', side_effect=inference), contextlib.redirect_stdout(io.StringIO()):
                diagnostic.main()
            self.assertEqual(len(calls), 6)
            protocol = json.loads((root / 'out/run.json').read_text())
            self.assertFalse(protocol['clean_inputs_used_in_inference'])
            def read_csv(name):
                with (root / 'out' / name).open() as handle:
                    return list(csv.DictReader(handle))
            summary = read_csv('summary.csv')
            self.assertEqual(len(summary), 48)
            self.assertTrue(all(int(row['count']) == 3 for row in summary if row['weather'] == 'ALL'))
            self.assertEqual(len(read_csv('per_image.csv')), 36)
            self.assertEqual(len(read_csv('paired_deltas.csv')), 18)
            for row in read_csv('paired_deltas.csv'):
                self.assertTrue(all(float(row[key]) == 0 for key in diagnostic.SCORES))
            refs = root / 'AWMM-Text/Test/Haze'
            a, b = (metrics.read_gray(refs / key / '0.png') for key in ('gt_ir', 'gt_vi'))
            y = diagnostic.read_y(root / 'out/images/y/author/Haze/0.png')
            rgb_l = metrics.read_gray(root / 'out/images/rgb/author/Haze/0.png')
            direct_score, color_score = metrics.evaluate(a, b, y), metrics.evaluate(a, b, rgb_l)
            effects = [r for r in read_csv('postprocess_delta.csv')
                       if r['weather'] == 'Haze' and r['model'] == 'author' and r['reference_set'] == 'clean_reference']
            self.assertEqual(len(effects), 1)
            for key in diagnostic.SCORES:
                self.assertAlmostEqual(float(effects[0][key]), direct_score[key] - color_score[key], places=12)
            self.assertEqual(len(read_csv('pixel_changes.csv')), 6)
            with patch('sys.argv', args):
                with self.assertRaises(FileExistsError):
                    diagnostic.main()

    def test_rejects_rgb_image_mislabelled_as_direct_y(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rgb.png'
            Image.new('RGB', (96, 96)).save(path)
            with self.assertRaisesRegex(ValueError, 'grayscale'):
                diagnostic.read_y(path)


if __name__ == '__main__':
    unittest.main()
