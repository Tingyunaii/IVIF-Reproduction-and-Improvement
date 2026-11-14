from pathlib import Path
import tempfile
import unittest
from PIL import Image

from tools.prepare_msrs_pairs import prepare
import json


class MSRSPairTests(unittest.TestCase):
    def test_grouped_split_preserves_test_and_excludes_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, values in [('trainsets', [10, 20, 30, 40, 50]), ('valsets', [10])]:
                for modality in ('ir', 'vi'):
                    folder = root / split / 'MSRS' / modality
                    folder.mkdir(parents=True)
                    for index, value in enumerate(values):
                        Image.new('RGB', (16, 16), (value, value, value)).save(folder / (str(index)+'.png'))
            report = prepare(root, root / 'out', .25, 3407, 5, 1)
            self.assertEqual(report['counts']['test']['samples'], 1)
            self.assertEqual(report['counts']['excluded']['samples'], 1)
            self.assertEqual(report['counts']['train']['samples'], 3)
            self.assertEqual(report['counts']['val']['samples'], 1)
            rows = {split: json.loads((root/'out'/(split+'.json')).read_text())['records'] for split in ('train','val','test')}
            for key in ('group','ir_sha256','vis_sha256'):
                self.assertFalse({r[key] for r in rows['train']} & {r[key] for r in rows['val']})
                self.assertFalse({r[key] for r in rows['train']} & {r[key] for r in rows['test']})
            self.assertFalse(any('clean_ir' in r for rs in rows.values() for r in rs))
            with self.assertRaises(FileExistsError):
                prepare(root, root / 'out', .25, 3407, 5, 1)


if __name__ == '__main__':
    unittest.main()
