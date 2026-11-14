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

import compare_awmm_original_baseline as comparison
from compare_awmm import sha256, write_csv


class OriginalComparisonTests(unittest.TestCase):
    def test_three_models_both_references_and_checkpoint_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / 'cache'
            rng = np.random.RandomState(9)
            cached_rows = []
            for weather in comparison.WEATHERS:
                sources, predictions = {}, {}
                for key in ('ir', 'vi', 'gt_ir', 'gt_vi'):
                    path = root / 'AWMM-Text/Test' / weather / key / '0.png'
                    path.parent.mkdir(parents=True)
                    Image.fromarray(rng.randint(0, 256, (96, 96, 3), dtype=np.uint8)).save(path)
                    sources[key] = {'0': path}
                for model, old in [('author', 'author'), ('current_y', 'adapted')]:
                    path = cache / 'images/rgb' / old / weather / '0.png'
                    path.parent.mkdir(parents=True)
                    Image.fromarray(rng.randint(0, 256, (96, 96, 3), dtype=np.uint8)).save(path)
                    predictions[model] = {'0': path}
                predictions['original_awmm'] = predictions['author']
                with contextlib.redirect_stdout(io.StringIO()):
                    rows, _ = comparison.evaluate_weather(weather, sources, predictions, 'mean')
                for row in rows:
                    if row['model'] != 'original_awmm':
                        row['model'] = 'adapted' if row['model'] == 'current_y' else 'author'
                        cached_rows.append(row)
            summaries = comparison.summarize(cached_rows, ('reference_set', 'model'))
            write_csv(cache / 'summary.csv', [dict(protocol='rgb_l', **row) for row in summaries])
            (cache / 'run.json').write_text(json.dumps(dict(dataset='AWMM-Text/Test', data_root=str(root),
                clean_inputs_used_in_inference=False, expected_count_per_weather=1, vif_mode='mean',
                checkpoints={'adapted': {'sha256': 'matched-y'}, 'author': {'sha256': 'author'}},
                scripts_sha256={'evaluate_p2fusion.py': sha256(Path(comparison.__file__).parent / 'evaluate_p2fusion.py')})))
            config = dict(model_variant='original_p2fusion', dataset={'input_mode': 'original_y'},
                          baseline_reference={'sha256': 'matched-y'})
            checkpoint = root / 'baseline.pt'
            torch.save({'config': config}, checkpoint)
            calls = []
            def inference(command, cwd, check):
                calls.append(command)
                self.assertEqual(command[command.index('--checkpoint') + 1], str(checkpoint))
                self.assertEqual(command[command.index('--color-source') + 1], 'original')
                self.assertFalse(any('/gt_' in word for word in command))
                destination = Path(command[command.index('--output') + 1])
                destination.mkdir(parents=True)
                source = Path(command[command.index('--vis-dir') + 1]) / '0.png'
                with Image.open(source) as image:
                    image.save(destination / '0.png')
            args = ['compare_awmm_original_baseline.py', '--checkpoint', str(checkpoint),
                    '--data-root', str(root), '--comparison-dir', str(cache), '--expected-count', '1',
                    '--out-dir', str(root / 'out'), '--device', 'cpu']
            with patch('sys.argv', args), patch('compare_awmm_original_baseline.subprocess.run', side_effect=inference), contextlib.redirect_stdout(io.StringIO()):
                comparison.main()
            self.assertEqual(len(calls), 3)
            def read(name):
                with (root / 'out' / name).open() as handle:
                    return list(csv.DictReader(handle))
            summary = read('summary.csv')
            self.assertEqual(len(summary), 24)
            self.assertEqual(len(read('paired_deltas.csv')), 12)
            self.assertEqual(len(read('per_image.csv')), 18)
            idx = {(row['weather'], row['reference_set'], row['model']): row for row in summary}
            for row in read('delta_summary.csv'):
                a, b = comparison.COMPARISONS[row['comparison']]
                for key in comparison.SCORES:
                    self.assertAlmostEqual(float(row[key]), float(idx[row['weather'], row['reference_set'], a][key])-float(idx[row['weather'], row['reference_set'], b][key]), places=12)
            config['baseline_reference']['sha256'] = 'wrong-y'
            torch.save({'config': config}, checkpoint)
            args[args.index('--out-dir') + 1] = str(root / 'bad')
            with patch('sys.argv', args), patch('compare_awmm_original_baseline.subprocess.run') as inference:
                with self.assertRaisesRegex(ValueError, 'differ from'):
                    comparison.main()
                inference.assert_not_called()
            self.assertFalse((root / 'bad').exists())


if __name__ == '__main__':
    unittest.main()
