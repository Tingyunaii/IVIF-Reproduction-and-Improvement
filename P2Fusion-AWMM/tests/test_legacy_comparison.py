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

import compare_awmm_legacy as legacy
import diagnose_awmm_y as diagnostic
from compare_awmm import sha256, write_csv


class LegacyComparisonTests(unittest.TestCase):
    def fixture(self, root):
        baseline = root / 'baseline'
        baseline.mkdir()
        rng = np.random.RandomState(7)
        all_rows = []
        for weather in legacy.WEATHERS:
            sources = {}
            for key in ('ir', 'vi', 'gt_ir', 'gt_vi'):
                folder = root / 'AWMM-Text/Test' / weather / key
                folder.mkdir(parents=True)
                path = folder / '0.png'
                Image.fromarray(rng.randint(0, 256, (96, 96, 3), dtype=np.uint8)).save(path)
                sources[key] = {'0': path}
            predictions = {}
            for model, old_name in (('author', 'author'), ('original_y', 'adapted')):
                path = baseline / 'images/rgb' / old_name / weather / '0.png'
                path.parent.mkdir(parents=True)
                Image.fromarray(rng.randint(0, 256, (96, 96, 3), dtype=np.uint8)).save(path)
                predictions[model] = {'0': path}
            predictions.update(legacy_gray=predictions['author'], legacy_color=predictions['original_y'])
            with contextlib.redirect_stdout(io.StringIO()):
                rows, _ = legacy.evaluate_weather(weather, sources, predictions, 'mean')
            for row in rows:
                if row['model'] in ('author', 'original_y'):
                    row['model'] = 'adapted' if row['model'] == 'original_y' else 'author'
                    row['protocol'] = 'rgb_l'
                    all_rows.append(row)
        summary = diagnostic.summarize(all_rows, ('reference_set', 'protocol', 'model'))
        write_csv(baseline / 'summary.csv', summary)
        project = Path(legacy.__file__).parent
        provenance = dict(dataset='AWMM-Text/Test', data_root=str(root), vif_mode='mean',
                          expected_count_per_weather=1, clean_inputs_used_in_inference=False,
                          protocols={key: value for key, value in diagnostic.PROTOCOLS.items()},
                          scripts_sha256={'evaluate_p2fusion.py': sha256(project / 'evaluate_p2fusion.py')},
                          checkpoints={'author': {'sha256': 'fixture-author'}, 'adapted': {'sha256': 'fixture-y'}})
        (baseline / 'run.json').write_text(json.dumps(provenance))
        checkpoint = root / 'legacy.pt'
        torch.save({'config': {'dataset': {'input_mode': 'rgb'}, 'loss': {'fusion_reference': 'first_channel'}}}, checkpoint)
        return baseline, checkpoint

    def test_four_models_both_references_and_delta_directions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline, checkpoint = self.fixture(root)
            commands = []
            def inference(command, cwd, check):
                commands.append(command)
                self.assertTrue(check)
                self.assertFalse(any('/gt_ir' in word or '/gt_vi' in word for word in command))
                self.assertEqual(command[command.index('--checkpoint') + 1], str(checkpoint))
                out = Path(command[command.index('--output') + 1])
                out.mkdir(parents=True)
                path = Path(command[command.index('--vis-dir') + 1]) / '0.png'
                color = command[command.index('--color-source') + 1]
                self.assertIn(color, ('none', 'input'))
                with Image.open(path) as image:
                    (image if color == 'input' else image.convert('L')).save(out / '0.png')
            args = ['compare_awmm_legacy.py', '--data-root', str(root), '--legacy-checkpoint', str(checkpoint),
                    '--baseline-dir', str(baseline), '--out-dir', str(root / 'out'),
                    '--expected-count', '1', '--device', 'cpu']
            with patch('sys.argv', args), patch('compare_awmm_legacy.subprocess.run', side_effect=inference), contextlib.redirect_stdout(io.StringIO()):
                legacy.main()
            self.assertEqual(len(commands), 6)
            def read(name):
                with (root / 'out' / name).open() as handle:
                    return list(csv.DictReader(handle))
            summary = read('summary.csv')
            self.assertEqual(len(summary), 32)
            self.assertEqual(len(read('per_image.csv')), 24)
            self.assertEqual(len(read('paired_deltas.csv')), 18)
            overall = [row for row in summary if row['weather'] == 'ALL']
            self.assertTrue(all(int(row['count']) == 3 for row in overall))
            rows = {(row['weather'], row['reference_set'], row['model']): row for row in summary}
            for delta in read('delta_summary.csv'):
                left, right = legacy.COMPARISONS[delta['comparison']]
                for key in diagnostic.SCORES:
                    expected = float(rows[delta['weather'], delta['reference_set'], left][key]) - float(rows[delta['weather'], delta['reference_set'], right][key])
                    self.assertAlmostEqual(float(delta[key]), expected, places=12)
            run = json.loads((root / 'out/run.json').read_text())
            self.assertEqual(run['legacy_checkpoint']['input_mode'], 'rgb')
            self.assertFalse(run['clean_inputs_used_in_inference'])
            self.assertIn('NOT decoder Y', run['models']['legacy_gray'])
            with patch('sys.argv', args):
                with self.assertRaises(FileExistsError):
                    legacy.main()

    def test_reject_y_checkpoint_as_first_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline, checkpoint = self.fixture(root)
            torch.save({'config': {'dataset': {'input_mode': 'original_y'}}}, checkpoint)
            args = ['compare_awmm_legacy.py', '--data-root', str(root), '--legacy-checkpoint', str(checkpoint),
                    '--baseline-dir', str(baseline), '--out-dir', str(root / 'out'), '--expected-count', '1']
            with patch('sys.argv', args), patch('compare_awmm_legacy.subprocess.run') as inference:
                with self.assertRaisesRegex(ValueError, 'FIRST RGB/R'):
                    legacy.main()
                inference.assert_not_called()
            self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
