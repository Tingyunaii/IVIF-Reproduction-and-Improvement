import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image
import torch

import visualize_author_vs_y as visual


class VisualComparisonTests(unittest.TestCase):
    def fixture(self, root):
        author, current = root / 'author.pth', root / 'current.pt'
        author.write_bytes(b'original model fixture')
        torch.save({'config': {'dataset': {'input_mode': 'original_y'}}}, current)
        original_paths = []
        for weather in visual.WEATHERS:
            for key in ('ir', 'vi', 'gt_ir', 'gt_vi'):
                folder = root / 'awmm/AWMM-Text/Test' / weather / key
                folder.mkdir(parents=True)
                for i in range(2):
                    path = folder / (str(i) + '.png')
                    Image.new('RGB', (80, 64), (30 + i, 50 + i, 70 + i)).save(path)
                    original_paths.append(path)
            for model in ('author', 'adapted'):
                folder = root / 'cache/images/rgb' / model / weather
                folder.mkdir(parents=True)
                for i in range(2):
                    path = folder / (str(i) + '.png')
                    Image.new('RGB', (80, 64), (60, 70, 80)).save(path)
                    original_paths.append(path)
        cache = dict(dataset='AWMM-Text/Test', clean_inputs_used_in_inference=False,
                     checkpoints={'author': {'sha256': visual.sha256(author)}, 'adapted': {'sha256': visual.sha256(current)}})
        (root / 'cache/run.json').write_text(json.dumps(cache))
        for dataset in visual.COUNTS:
            for key in ('ir', 'vi'):
                folder = root / 'data' / dataset / key
                folder.mkdir(parents=True)
                extension = ('bmp' if key == 'ir' else 'jpg') if dataset == 'RoadScene' else 'png'
                for i in range(2):
                    path = folder / (str(i) + '.' + extension)
                    Image.new('RGB', (80, 64), (20 + i, 40 + i, 60 + i)).save(path)
                    original_paths.append(path)
        return author, current, original_paths

    def test_all_seven_groups_correct_models_selected_only_inference_and_portable_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            author, current, paths = self.fixture(root)
            before = {str(path): visual.sha256(path) for path in paths}
            commands = []

            def inference(command, cwd, check):
                commands.append(command)
                self.assertTrue(check)
                source = Path(command[command.index('--vis-dir') + 1])
                self.assertEqual(len(visual.image_index(source)), 1)
                if command[1].endswith('infer_awmm.py'):
                    self.assertEqual(command[command.index('--checkpoint') + 1], str(current))
                    self.assertEqual(command[command.index('--color-source') + 1], 'original')
                else:
                    self.assertTrue(command[1].endswith('infer_original_p2fusion.py'))
                    self.assertEqual(command[command.index('--checkpoint') + 1], str(author))
                self.assertFalse(any('/gt_ir' in word or '/gt_vi' in word for word in command))
                destination = Path(command[command.index('--output') + 1])
                destination.mkdir(parents=True)
                for name, path in visual.image_index(source).items():
                    with Image.open(path) as image:
                        image.convert('RGB').save(destination / (name + '.png'))

            args = ['visualize_author_vs_y.py', '--author-checkpoint', str(author), '--y-checkpoint', str(current),
                    '--awmm-root', str(root / 'awmm'), '--test-root', str(root / 'data'),
                    '--awmm-cache', str(root / 'cache'), '--out-dir', str(root / 'out'),
                    '--limit', '1', '--device', 'cpu', '--panel-width', '80']
            with patch.object(visual, 'COUNTS', {key: 2 for key in visual.COUNTS}), patch.object(visual, 'AWMM_COUNT', 2), patch('sys.argv', args), patch('visualize_author_vs_y.subprocess.run', side_effect=inference), contextlib.redirect_stdout(io.StringIO()):
                visual.main()
            # Cached AWMM is reused; four conventional datasets run two specified models each.
            self.assertEqual(len(commands), 8)
            run = json.loads((root / 'out/run.json').read_text())
            self.assertEqual(len(run['groups']), 7)
            self.assertFalse(run['old_rgb_model_included'])
            self.assertFalse(run['clean_images_used_in_inference'])
            for group in run['groups']:
                self.assertEqual(group['exported_count'], 1)
                labels = list(group['samples'][0]['panels'])
                if group['dataset'] == 'AWMM':
                    self.assertEqual(labels, ['Clean IR', 'Clean VIS', 'Degraded IR', 'Degraded VIS', 'Author P2Fusion', 'Current Y model'])
                else:
                    self.assertEqual(labels, ['IR input', 'VIS input', 'Author P2Fusion', 'Current Y model'])
            with zipfile.ZipFile(root / 'out/preview.zip') as archive:
                self.assertIn('index.html', archive.namelist())
                self.assertEqual(len([name for name in archive.namelist() if name.endswith('.jpg')]), 7)
            self.assertEqual(before, {str(path): visual.sha256(path) for path in paths})
            with patch('sys.argv', args):
                with self.assertRaises(FileExistsError):
                    visual.main()

    def test_rejects_legacy_checkpoint_and_mismatched_native_panels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            author = root / 'author.pth'
            author.write_bytes(b'original')
            checkpoint = root / 'legacy.pt'
            torch.save({'config': {'dataset': {'input_mode': 'rgb'}}}, checkpoint)
            args = ['visualize_author_vs_y.py', '--author-checkpoint', str(author),
                    '--y-checkpoint', str(checkpoint), '--out-dir', str(root / 'out')]
            with patch('sys.argv', args), patch('visualize_author_vs_y.subprocess.run') as inference:
                with self.assertRaisesRegex(ValueError, 'original_y'):
                    visual.main()
                inference.assert_not_called()
            self.assertFalse((root / 'out').exists())
            a, b = root / 'a.png', root / 'b.png'
            Image.new('RGB', (80, 64)).save(a)
            Image.new('RGB', (81, 64)).save(b)
            with self.assertRaisesRegex(ValueError, 'dimensions differ'):
                visual.render_comparison([('input', a), ('Current Y model', b)], root / 'bad.jpg', 'test')


if __name__ == '__main__':
    unittest.main()
