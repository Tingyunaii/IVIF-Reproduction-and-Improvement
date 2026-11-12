import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import evaluate_p2fusion as evaluator


class EvaluationTests(unittest.TestCase):
    def test_stem_pairing_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (96,96)).save(root/'a.jpg')
            self.assertEqual(set(evaluator.image_files(root,True)), {'a'})
            Image.new('RGB', (96,96)).save(root/'a.png')
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                evaluator.image_files(root,True)

    def test_cli_roadscene_reference_and_mixed_extensions(self):
        rng = np.random.RandomState(7)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder, extension in (('ir','bmp'),('vi','jpg'),('fused','png')):
                (root/folder).mkdir()
                Image.fromarray(rng.randint(0,256,(96,96),dtype=np.uint8)).save(root/folder/('one.'+extension))
            args=['evaluate_p2fusion.py','--dataset','RoadScene','--match-stem','--expected-count','1',
                  '--ir-dir',str(root/'ir'),'--vi-dir',str(root/'vi'),
                  '--fused-dir',str(root/'fused'),'--out-dir',str(root/'out')]
            with patch('sys.argv',args),contextlib.redirect_stdout(io.StringIO()):
                evaluator.main()
            summary=json.loads((root/'out/summary.json').read_text())
            self.assertEqual(summary['dataset'],'RoadScene')
            self.assertEqual(summary['count'],1)
            self.assertEqual(summary['paper_reference']['MI'],3.021)
            self.assertEqual(summary['protocol']['pairing'],'stem')
            self.assertFalse(summary['official_p2fusion_evaluator'])
            for key in evaluator.PAPER:
                self.assertTrue(np.isfinite(summary['means'][key]))

    def test_metric_algorithms_match_uploaded_script(self):
        path=Path('/workspace/attachments/5ce2856e-c134-4db7-820e-e87498a5df37/evaluate_p2fusion.py')
        if not path.exists():
            self.skipTest('Original upload available only in development audit environment')
        spec=importlib.util.spec_from_file_location('uploaded_evaluator',path)
        original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
        rng=np.random.RandomState(9)
        images=[rng.randint(0,256,(96,96)).astype(np.float64) for _ in range(3)]
        self.assertEqual(evaluator.evaluate(*images),original.evaluate(*images))


if __name__=='__main__':
    unittest.main()
