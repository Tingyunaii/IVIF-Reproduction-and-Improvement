import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from models.P2Fusion import p2fusion
import infer_original_p2fusion
from awmm.preprocessing import original_color_restore, fused_y_uint8
from infer_awmm import image_tensor


class BaselineTests(unittest.TestCase):
    def test_original_checkpoint_rgb_output(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            checkpoint=root/'original.pth'
            model=p2fusion()
            torch.save({'params':model.state_dict()},checkpoint)
            for folder in ('ir','vi'):
                (root/folder).mkdir()
                Image.fromarray(np.random.RandomState(len(folder)).randint(0,256,(64,64,3),dtype=np.uint8)).save(root/folder/'one.png')
            args=['infer_original_p2fusion.py','--checkpoint',str(checkpoint),
                  '--ir-dir',str(root/'ir'),'--vis-dir',str(root/'vi'),
                  '--output',str(root/'out'),'--save-y-dir',str(root/'y'),'--device','cpu']
            with patch('sys.argv',args),contextlib.redirect_stdout(io.StringIO()):
                infer_original_p2fusion.main()
            with Image.open(root/'out/one.png') as image:
                self.assertEqual(image.mode,'RGB')
                self.assertEqual(image.size,(64,64))
            model.eval()
            with torch.no_grad():
                expected = model(image_tensor(root/'ir/one.png', 'original_y'),
                                 image_tensor(root/'vi/one.png', 'original_y'))[0]
            with Image.open(root/'y/one.png') as image:
                self.assertEqual(image.mode, 'L')
                np.testing.assert_array_equal(np.asarray(image), fused_y_uint8(expected))
            with Image.open(root/'out/one.png') as image:
                np.testing.assert_array_equal(np.asarray(image), original_color_restore(expected, root/'vi/one.png'))
