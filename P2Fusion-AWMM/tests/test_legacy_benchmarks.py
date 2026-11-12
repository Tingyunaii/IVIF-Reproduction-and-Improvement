import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import evaluate_legacy_benchmarks as runner
import evaluate_p2fusion as evaluator


class LegacyBenchmarksTests(unittest.TestCase):
    def fixture(self, root):
        checkpoint = root / 'legacy.pt'
        torch.save({'config': {'dataset': {'input_mode': 'rgb'},
                               'loss': {'fusion_reference': 'first_channel'}}}, checkpoint)
        rng = np.random.RandomState(16)
        for dataset in runner.COUNTS:
            for modal in ('ir', 'vi'):
                directory = root / 'data' / dataset / modal
                directory.mkdir(parents=True)
                extension = ('bmp' if modal == 'ir' else 'jpg') if dataset == 'RoadScene' else 'png'
                Image.fromarray(rng.randint(0, 256, (96, 96, 3), dtype=np.uint8)).save(directory / ('one.' + extension))
        return checkpoint

    def arguments(self, root, checkpoint):
        return ['evaluate_legacy_benchmarks.py', '--checkpoint', str(checkpoint),
                '--test-root', str(root / 'data'), '--out-dir', str(root / 'out'), '--device', 'cpu']

    def fake_child(self, commands, failure=None):
        def run(command, cwd, check):
            commands.append(command)
            self.assertTrue(check)
            if command[1].endswith('infer_awmm.py'):
                self.assertEqual(command[command.index('--color-source') + 1], 'input')
                source = Path(command[command.index('--vis-dir') + 1])
                if source.parent.name == failure:
                    raise subprocess.CalledProcessError(1, command)
                output = Path(command[command.index('--output') + 1])
                output.mkdir(parents=True)
                for name, path in evaluator.image_files(source, match_stem=True).items():
                    with Image.open(path) as image:
                        image.convert('RGB').save(output / (name + '.png'))
            else:
                self.assertTrue(command[1].endswith('evaluate_p2fusion.py'))
                self.assertIn('--match-stem', command)
                with patch('sys.argv', command[1:]):
                    evaluator.main()
        return run

    def test_four_datasets_color_output_and_real_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self.fixture(root)
            args = self.arguments(root, checkpoint)
            commands = []
            with patch.object(runner, 'COUNTS', {name: 1 for name in runner.COUNTS}), patch('sys.argv', args), patch('evaluate_legacy_benchmarks.subprocess.run', side_effect=self.fake_child(commands)), contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            self.assertEqual(len(commands), 8)
            with (root / 'out/summary.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row['dataset'] for row in rows}, set(runner.COUNTS))
            self.assertTrue(all(row['count'] == '1' for row in rows))
            for row in rows:
                self.assertTrue(all(np.isfinite(float(row[key])) for key in runner.SCORES))
                with Image.open(root / 'out/images' / row['dataset'] / 'one.png') as image:
                    self.assertEqual(image.mode, 'RGB')
            road = json.loads((root / 'out/metrics/RoadScene/summary.json').read_text())
            self.assertEqual(road['paper_reference']['MI'], 3.021)
            self.assertEqual(road['protocol']['pairing'], 'stem')
            self.assertEqual(road['protocol']['VIF_selected'], 'mean')
            run = json.loads((root / 'out/run.json').read_text())
            self.assertEqual(run['input_mode'], 'rgb')
            self.assertEqual(run['color_source'], 'input')
            self.assertFalse(run['training'])
            with patch('sys.argv', args):
                with self.assertRaises(FileExistsError):
                    runner.main()

    def test_failed_inference_stops_and_preserves_completed_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self.fixture(root)
            commands = []
            with patch.object(runner, 'COUNTS', {name: 1 for name in runner.COUNTS}), patch('sys.argv', self.arguments(root, checkpoint)), patch('evaluate_legacy_benchmarks.subprocess.run', side_effect=self.fake_child(commands, failure='FMB')), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.main()
            self.assertEqual(len(commands), 5)
            with (root / 'out/summary.csv').open() as handle:
                self.assertEqual([row['dataset'] for row in csv.DictReader(handle)], ['MSRS', 'M3FD'])
            self.assertFalse((root / 'out/images/RoadScene').exists())

    def test_audit_rejects_count_and_missing_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            runner.audit_dataset(root / 'data', 'RoadScene', 1)
            with self.assertRaisesRegex(ValueError, 'expected 2'):
                runner.audit_dataset(root / 'data', 'RoadScene', 2)
            (root / 'data/RoadScene/vi/one.jpg').rename(root / 'data/RoadScene/vi/other.jpg')
            with self.assertRaisesRegex(ValueError, 'IDs differ'):
                runner.audit_dataset(root / 'data', 'RoadScene', 1)


if __name__ == '__main__':
    unittest.main()
