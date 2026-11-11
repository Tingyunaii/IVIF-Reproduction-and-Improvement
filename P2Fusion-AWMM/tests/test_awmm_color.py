import unittest
import torch
from infer_awmm import restore_color


class ColorTests(unittest.TestCase):
    def test_first_channel_identity(self):
        rgb = torch.rand(2, 3, 9, 7)
        torch.testing.assert_close(restore_color(rgb[:, :1], rgb, "first_channel"), rgb)

    def test_luminance_identity(self):
        rgb = torch.rand(1, 3, 8, 8)
        y = (rgb * rgb.new_tensor([.299, .587, .114]).view(1, 3, 1, 1)).sum(1, keepdim=True)
        torch.testing.assert_close(restore_color(y, rgb, "luminance"), rgb)

    def test_preserves_chroma_and_red_anchor(self):
        rgb = torch.tensor([.3, .4, .5]).view(1, 3, 1, 1)
        fused = torch.full((1, 1, 1, 1), .5)
        output = restore_color(fused, rgb, "first_channel")
        torch.testing.assert_close(output[:, :1], fused)
        torch.testing.assert_close(output[:, 1:] - output[:, :1], rgb[:, 1:] - rgb[:, :1])
        self.assertFalse(torch.equal(output[:, :1], output[:, 1:2]))

    def test_achromatic_and_clipping(self):
        source = torch.full((1, 3, 2, 2), .3)
        fused = torch.full((1, 1, 2, 2), .7)
        torch.testing.assert_close(restore_color(fused, source, "first_channel"), fused.expand_as(source))
        source[:, 2] = 1
        self.assertLessEqual(float(restore_color(fused, source, "first_channel").max()), 1)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            restore_color(torch.ones(1, 1, 2, 2), torch.ones(1, 3, 3, 3), "first_channel")
        with self.assertRaises(ValueError):
            restore_color(torch.ones(1, 1, 2, 2), torch.ones(1, 3, 2, 2), "unknown")
