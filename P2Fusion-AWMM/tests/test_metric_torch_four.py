import ast
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

import evaluate_metric_torch_four as runner

UPLOAD = Path('/workspace/attachments/df3a662d-c820-4a5c-9546-6c2ec4b0b14f/Metric_torch.py')


@unittest.skipUnless(UPLOAD.is_file(), 'Uploaded Metric_torch.py fidelity audit is development-only')
class MetricFourTests(unittest.TestCase):
    def backend(self, root):
        # Real test SSIM backend; the user's server SSIM file is intentionally not substituted in production.
        path = root / 'ssim.py'
        path.write_text('from pytorch_msssim import ssim\n')
        return path

    def test_original_definitions_and_rank_adapter(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            functions, _ = runner.load_functions(UPLOAD, self.backend(root))
            layout, _ = runner.detect_ssim_layout(functions['SSIM'], torch.device('cpu'), 'auto')
            self.assertEqual(layout, 'nchw')
            # Independent full-function compilation from the actual upload, without production selection.
            namespace = {'torch': torch, 'ssim': __import__('pytorch_msssim').ssim}
            nodes = [n for n in ast.parse(UPLOAD.read_text()).body if isinstance(n, ast.FunctionDef)]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), 'uploaded_reference', 'exec'), namespace)
            generator = torch.Generator().manual_seed(16)
            images = [torch.randint(0, 256, (96, 96), generator=generator).float() for _ in range(3)]
            actual = runner.score_images(images, functions, layout)
            for name in runner.METRICS:
                args = [image[None, None] for image in images] if name == 'SSIM' else images
                self.assertEqual(actual[name], float(namespace[name + '_function'](*args)))

    def test_five_datasets_both_awmm_references_and_read_only_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ssim = self.backend(root)
            rng = np.random.RandomState(16)
            input_paths = []
            for weather in runner.WEATHERS:
                for key in ('ir', 'vi', 'gt_ir', 'gt_vi'):
                    path = root / 'awmm/AWMM-Text/Test' / weather / key / 'one.png'
                    path.parent.mkdir(parents=True)
                    Image.fromarray(rng.randint(0, 256, (96, 96, 3), dtype=np.uint8)).save(path)
                    input_paths.append(path)
                path = root / 'awmm_images' / weather / 'one.png'
                path.parent.mkdir(parents=True)
                Image.fromarray(rng.randint(0, 256, (96, 96, 3), dtype=np.uint8)).save(path)
                input_paths.append(path)
            for dataset in runner.COUNTS:
                for key in ('ir', 'vi'):
                    extension = ('bmp' if key == 'ir' else 'jpg') if dataset == 'RoadScene' else 'png'
                    path = root / 'data' / dataset / key / ('one.' + extension)
                    path.parent.mkdir(parents=True)
                    Image.fromarray(rng.randint(0, 256, (96, 96, 3), dtype=np.uint8)).save(path)
                    input_paths.append(path)
                path = root / 'benchmark_images' / dataset / 'one.png'
                path.parent.mkdir(parents=True)
                Image.fromarray(rng.randint(0, 256, (96, 96, 3), dtype=np.uint8)).save(path)
                input_paths.append(path)
            before = {str(path): runner.sha256(path) for path in input_paths}
            args = ['evaluate_metric_torch_four.py', '--metric-file', str(UPLOAD), '--ssim-file', str(ssim),
                    '--awmm-root', str(root / 'awmm'), '--test-root', str(root / 'data'),
                    '--awmm-images', str(root / 'awmm_images'), '--benchmark-images', str(root / 'benchmark_images'),
                    '--out-dir', str(root / 'out'), '--device', 'cpu', '--threads', '1']
            with patch.object(runner, 'COUNTS', {name: 1 for name in runner.COUNTS}), patch.object(runner, 'AWMM_COUNT', 1), patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            with (root / 'out/summary.csv').open() as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(len(summary), 6)
            awmm = [row for row in summary if row['dataset'] == 'AWMM']
            self.assertEqual({row['reference_set'] for row in awmm}, {'clean_reference', 'degraded_reference'})
            self.assertTrue(all(row['count'] == '3' for row in awmm))
            with (root / 'out/per_image.csv').open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 10)
            for row in summary:
                self.assertTrue(all(np.isfinite(float(row[name])) for name in runner.METRICS))
                self.assertTrue(all(row[name + '_finite'] == row['count'] for name in runner.METRICS))
            run = json.loads((root / 'out/run.json').read_text())
            self.assertEqual(run['ssim_layout'], 'nchw')
            self.assertFalse(run['inference'])
            self.assertEqual(run['metric_file']['sha256'], runner.sha256(UPLOAD))
            self.assertEqual(before, {str(path): runner.sha256(path) for path in input_paths})
            with patch('sys.argv', args):
                with self.assertRaises(FileExistsError):
                    runner.main()

    def test_nonfinite_original_scd_is_not_silently_replaced_or_excluded(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            functions, _ = runner.load_functions(UPLOAD, self.backend(root))
            a, b = torch.rand(96, 96) * 255, torch.rand(96, 96) * 255
            score = runner.score_images([a, b, b], functions, 'nchw')
            self.assertTrue(np.isnan(score['SCD']))
            rows = [dict(dataset='MSRS', subset='ALL', reference_set='input_reference', **score)]
            summary = runner.summarize(rows)[0]
            self.assertTrue(np.isnan(summary['SCD']))
            self.assertEqual(summary['SCD_finite'], 0)
            self.assertEqual(summary['count'], 1)


if __name__ == '__main__':
    unittest.main()
