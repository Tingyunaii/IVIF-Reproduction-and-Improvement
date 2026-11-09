"""Behavioral checks for the FDN insertion, gating gradients and checkpoint API."""

import tempfile
import unittest
from pathlib import Path

import torch

from models.P2Fusion import p2fusion
from models.awmm_network import AWMMP2Fusion, DynamicWeatherAwareness, FeatureDisentanglement


SMALL = {
    "num_blocks": 1,
    "Ex_depths": [1],
    "input_resolution": (16, 16),
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.0,
}


class AWMMNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_weather_path_detaches_private_but_trains_gate(self):
        torch.manual_seed(7)
        module = DynamicWeatherAwareness()
        invariant = torch.randn(2, 60, 8, 8, requires_grad=True)
        private = torch.randn_like(invariant, requires_grad=True)
        adjusted, weather, region = module(invariant, private)
        self.assertTrue(torch.allclose(adjusted, invariant * (1 + weather * region)))
        adjusted.square().mean().backward()
        self.assertIsNone(private.grad)
        self.assertGreater(invariant.grad.abs().sum().item(), 0)
        self.assertGreater(module.weather[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(module.region[0].weight.grad.abs().sum().item(), 0)

    def test_private_encoder_still_learns_from_its_own_objective(self):
        fdn = FeatureDisentanglement()
        invariant, private = fdn(torch.randn(1, 60, 8, 8))
        private.square().mean().backward()
        self.assertGreater(fdn.private_encoder[0].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(fdn.invariant_residual[0].weight.grad)

    def test_both_original_heads_receive_same_weather_adjusted_map(self):
        torch.manual_seed(1)
        model = AWMMP2Fusion(**SMALL).eval()
        observed = {}
        handles = []
        for modality, unembed in (("ir", model.patch_unembed), ("vis", model.patch_unembed_vis)):
            def record_weather(module, args, output, key=modality):
                observed[key + "_weather"] = output[0].detach().clone()

            def record_proj(module, args, key=modality):
                observed[key + "_proj"] = args[0].detach().clone()

            def record_prompt(module, args, key=modality):
                observed[key + "_prompt"] = args[0].detach().clone()

            handles.append(getattr(model, "dwam_" + modality).register_forward_hook(record_weather))
            handles.append(unembed.proj.register_forward_pre_hook(record_proj))
            handles.append(unembed.prompt_head.register_forward_pre_hook(record_prompt))
        with torch.no_grad():
            output = model(torch.rand(1, 3, 33, 65), torch.rand(1, 3, 33, 65), return_aux=True)
        for handle in handles:
            handle.remove()
        for modality in ("ir", "vis"):
            self.assertTrue(torch.equal(observed[modality + "_weather"], observed[modality + "_proj"]))
            self.assertTrue(torch.equal(observed[modality + "_weather"], observed[modality + "_prompt"]))
            self.assertEqual(output["aux"][modality]["clean_recon"].shape, (1, 3, 33, 65))
            self.assertEqual(output["aux"][modality]["residual_recon"].shape, (1, 3, 33, 65))
        self.assertEqual(output["fused"].shape, (1, 1, 33, 65))
        self.assertEqual(output["vis_score"].shape, (1, 1))

    def test_inference_does_not_execute_auxiliary_heads_and_handles_tiny_image(self):
        model = AWMMP2Fusion(**SMALL).eval()

        def forbidden(*args):
            raise AssertionError("Training-only reconstruction head executed during inference")

        handles = []
        for modality in ("ir", "vis"):
            handles.append(getattr(model, "clean_reconstructor_" + modality).register_forward_hook(forbidden))
            handles.append(getattr(model, "residual_reconstructor_" + modality).register_forward_hook(forbidden))
        with torch.no_grad():
            output = model(torch.rand(1, 3, 3, 5), torch.rand(1, 3, 3, 5))
        for handle in handles:
            handle.remove()
        self.assertNotIn("aux", output)
        self.assertEqual(output["fused"].shape, (1, 1, 3, 5))

    def test_clean_encoding_skips_fusion_and_prompt_heads(self):
        model = AWMMP2Fusion(**SMALL).eval()

        def forbidden(*args):
            raise AssertionError("Unused clean-branch fusion/prompt module executed")

        handles = [block.register_forward_hook(forbidden) for block in model.fusion_blocks]
        handles.append(model.patch_unembed.register_forward_hook(forbidden))
        handles.append(model.patch_unembed_vis.register_forward_hook(forbidden))
        with torch.no_grad():
            output = model.encode_aux(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32))
        for handle in handles:
            handle.remove()
        self.assertEqual(set(output), {"ir", "vis"})
        self.assertEqual(output["ir"]["clean_recon"].shape, (1, 3, 32, 32))

    def test_training_backward_reaches_both_private_and_weather_modules(self):
        model = AWMMP2Fusion(**SMALL).train()
        output = model(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64), return_aux=True)
        loss = output["fused"].mean() + output["prompt_ir"].mean() + output["vis_score"].mean()
        for aux in output["aux"].values():
            loss = loss + aux["clean_recon"].mean() + aux["residual_recon"].square().mean()
        loss.backward()
        for modality in ("ir", "vis"):
            private = getattr(model, "fdn_" + modality).private_encoder[0].weight.grad
            weather = getattr(model, "dwam_" + modality).weather[0].weight.grad
            self.assertTrue(torch.isfinite(private).all())
            self.assertGreater(private.abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(weather).all())
            self.assertGreater(weather.abs().sum().item(), 0)

    def test_original_checkpoint_load_preserves_original_names_and_rejects_missing(self):
        base = p2fusion(**SMALL)
        model = AWMMP2Fusion(**SMALL)
        self.assertEqual(set(base.state_dict()), model._base_state_keys)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original.pth"
            state = {"module." + key: value for key, value in base.state_dict().items()}
            torch.save({"params": state}, path)
            report = model.load_initial_checkpoint(path)
            self.assertEqual(report["loaded"], len(base.state_dict()))
            self.assertTrue(report["initialized_adaptation"])
            self.assertTrue(all(key.startswith(model.adaptation_parameter_prefixes)
                                for key in report["initialized_adaptation"]))
            for key, value in base.state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
            state.pop(next(iter(state)))
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "missing_original"):
                model.load_initial_checkpoint(path)

    def test_incompatible_architecture_rejected_early(self):
        with self.assertRaisesRegex(ValueError, "embed_dim"):
            AWMMP2Fusion(embed_dim=96)
        with self.assertRaisesRegex(ValueError, "num_blocks"):
            AWMMP2Fusion(num_blocks=0)


if __name__ == "__main__":
    unittest.main()
