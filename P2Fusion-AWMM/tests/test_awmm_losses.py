"""Synthetic checks: no dataset, pretrained weights or GPU required."""

import unittest

import torch
import torch.nn.functional as F

from awmm.losses import AWMMLoss, DEFAULT_WEIGHTS


def synthetic_batch():
    torch.manual_seed(12)
    batch = {key: torch.rand(2, 3, 32, 32)
             for key in ("ir", "vis", "clean_ir", "clean_vis")}

    def aux():
        invariant = torch.randn(2, 4, 8, 8, requires_grad=True)
        private = torch.randn(2, 4, 8, 8, requires_grad=True)
        return {
            "inv": invariant,
            "private": private,
            "clean_recon": F.interpolate(invariant[:, :3].sigmoid(), size=(32, 32),
                                         mode="bilinear", align_corners=False),
            "residual_recon": F.interpolate(private[:, :3], size=(32, 32),
                                            mode="bilinear", align_corners=False),
        }

    degraded = {
        "fused": torch.rand(2, 1, 32, 32, requires_grad=True),
        "prompt_ir": torch.rand(2, 8, 8, 8, requires_grad=True),
        "prompt_vis": torch.rand(2, 1, 8, 8, requires_grad=True),
        "vis_score": torch.rand(2, 1, requires_grad=True),
        "aux": {key: aux() for key in ("ir", "vis")},
    }
    clean = {key: aux() for key in ("ir", "vis")}
    teacher = {"ir": torch.rand(2, 1, 32, 32, requires_grad=True),
               "vis": torch.rand(2, 1, requires_grad=True)}
    return degraded, clean, batch, teacher


class AWMMLossTests(unittest.TestCase):
    def test_eight_terms_sum_and_backward(self):
        degraded, clean, batch, teacher = synthetic_batch()
        criterion = AWMMLoss(device="cpu")
        losses = criterion(degraded, clean, batch, teacher)
        self.assertTrue(set(DEFAULT_WEIGHTS).issubset(losses))
        expected = sum(DEFAULT_WEIGHTS[name] * losses[name] for name in DEFAULT_WEIGHTS)
        torch.testing.assert_close(losses["total"], expected)
        self.assertTrue(all(torch.isfinite(value).all() for value in losses.values()))
        losses["total"].backward()
        for modality in ("ir", "vis"):
            for branch in (degraded["aux"], clean):
                for name in ("inv", "private"):
                    grad = branch[modality][name].grad
                    self.assertIsNotNone(grad)
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(), 0)
        self.assertGreater(degraded["fused"].grad.abs().sum().item(), 0)
        self.assertGreater(degraded["prompt_ir"].grad.abs().sum().item(), 0)
        self.assertGreater(degraded["vis_score"].grad.abs().sum().item(), 0)
        self.assertIsNone(teacher["ir"].grad)
        self.assertIsNone(teacher["vis"].grad)

    def test_consistency_clean_reference_is_detached(self):
        degraded, clean, batch, teacher = synthetic_batch()
        loss = AWMMLoss()(degraded, clean, batch, teacher)["consistency"]
        loss.backward()
        for modality in ("ir", "vis"):
            self.assertIsNone(clean[modality]["inv"].grad)
            self.assertIsNone(degraded["aux"][modality]["private"].grad)
            self.assertGreater(degraded["aux"][modality]["inv"].grad.abs().sum().item(), 0)

    def test_orthogonality_cannot_reward_anticorrelation(self):
        x = torch.randn(2, 4, 3, 3)
        parallel = AWMMLoss.orthogonality(x, x)
        antiparallel = AWMMLoss.orthogonality(x, -x)
        torch.testing.assert_close(parallel, torch.tensor(1.0))
        torch.testing.assert_close(antiparallel, parallel)
        self.assertEqual(AWMMLoss.orthogonality(x, torch.zeros_like(x)).item(), 0)

    def test_reference_channel_is_explicit(self):
        image = torch.zeros(1, 3, 4, 4)
        image[:, 1] = 1
        self.assertEqual(AWMMLoss().reference_channel(image).sum().item(), 0)
        expected = torch.full((1, 1, 4, 4), 0.587)
        torch.testing.assert_close(AWMMLoss(fusion_reference="luminance").reference_channel(image), expected)

    def test_original_five_losses_unchanged(self):
        degraded, clean, batch, teacher = synthetic_batch()
        criterion = AWMMLoss()
        result = criterion(degraded, clean, batch, teacher)
        target = F.interpolate(teacher["ir"].detach(), size=(8, 8),
                               mode="bilinear", align_corners=False)[:, 0]
        original = criterion.original(
            degraded["prompt_ir"].clamp(1e-6, 1 - 1e-6), target,
            degraded["vis_score"], teacher["vis"].detach(),
            degraded["fused"], batch["clean_vis"][:, :1], batch["clean_ir"][:, :1])
        for name, original_name in (("intensity", "adaptive_intensity"),
                                    ("ssim", "ssim"), ("gradient", "gradient"),
                                    ("distill_ir", "distill_ir"),
                                    ("distill_vis", "distill_vis")):
            torch.testing.assert_close(result[name], original[original_name])

    def test_ssim_range_is_explicit_and_nontrivial(self):
        degraded, clean, batch, teacher = synthetic_batch()
        corrected = AWMMLoss(ssim_data_range=1)(degraded, clean, batch, teacher)["ssim"]
        legacy_range = AWMMLoss(ssim_data_range=255)(degraded, clean, batch, teacher)["ssim"]
        self.assertGreater(corrected.item(), 0.5)
        self.assertLess(legacy_range.item(), 0.1)
        for key in ("clean_ir", "clean_vis"):
            batch[key] = torch.zeros_like(batch[key])
        degraded["fused"] = torch.zeros_like(degraded["fused"])
        flat = AWMMLoss()(degraded, clean, batch, teacher)["ssim"]
        torch.testing.assert_close(flat, torch.tensor(0.0))

    def test_nonfinite_teacher_and_unknown_options_fail(self):
        degraded, clean, batch, teacher = synthetic_batch()
        teacher["vis"] = torch.full((2, 1), float("nan"))
        with self.assertRaisesRegex(ValueError, "Teacher targets"):
            AWMMLoss()(degraded, clean, batch, teacher)
        with self.assertRaises(ValueError):
            AWMMLoss(weights={"typo": 1})
        with self.assertRaises(ValueError):
            AWMMLoss(weights={"ssim": -1})
        with self.assertRaises(ValueError):
            AWMMLoss(fusion_reference="unknown")


if __name__ == "__main__":
    unittest.main()
