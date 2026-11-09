"""The five original P2Fusion losses plus three paired-weather constraints.

The reconstruction constraint uses paired clean images, so this is reference-
guided training rather than a claim of fully unsupervised image restoration.
All returned components are *unweighted*. ``total`` is the weighted sum of the
eight entries in ``DEFAULT_WEIGHTS``; diagnostic entries are not added again.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import ssim

from models.loss_total import FusionLoss, Sobelxy


DEFAULT_WEIGHTS = {
    "intensity": 8.0,
    "ssim": 25.0,
    "gradient": 20.0,
    "distill_ir": 0.5,
    "distill_vis": 5.0,
    "consistency": 1.0,
    "orthogonality": 0.01,
    "self_supervision": 1.0,
}


class RangeAwareSSIM(nn.Module):
    """Original gradient-weighted SSIM with an explicit intensity range.

    Uploaded source implicitly used the library default 255 on [0, 1] images.
    We explicitly correct it to 1 by default. ``data_range=255`` reproduces that
    range convention; all-zero references use equal weights instead of a
    constant loss of one. Batch-mean gradient weighting is otherwise unchanged.
    """

    def __init__(self, data_range):
        super().__init__()
        self.sobelconv = Sobelxy()
        self.data_range = float(data_range)

    def forward(self, image_a, image_b, fused):
        mean_a = self.sobelconv(image_a).mean()
        mean_b = self.sobelconv(image_b).mean()
        total = mean_a + mean_b
        equal = total.new_tensor(0.5)
        weight_a = torch.where(total > 1e-8, mean_a / (total + 1e-8), equal)
        weight_b = torch.where(total > 1e-8, mean_b / (total + 1e-8), equal)
        return (weight_a * ssim(image_a, fused, data_range=self.data_range)
                + weight_b * ssim(image_b, fused, data_range=self.data_range))


class AWMMLoss(nn.Module):
    """Loss for aligned ``ir/vis/clean_ir/clean_vis`` AWMM batches.

    ``degraded_outputs`` is returned by ``AWMMP2Fusion(..., return_aux=True)``.
    ``clean_aux`` is returned by ``AWMMP2Fusion.encode_aux(clean_ir, clean_vis)``.
    Teacher targets are ``{'ir': Bx1xHxW probability, 'vis': Bx1 quality}``,
    both generated from the clean references. IR saliency is resized to prompt
    resolution; training crops should be multiples of the network's 32-pixel
    stride/window product, as configured in the provided training script.

    ``first_channel`` deliberately preserves the uploaded training code's
    ``image[:, :1]`` reference convention. ``luminance`` is an explicit optional
    experiment, not an unannounced change to the baseline. The original SSIM
    formulas and weights are retained. ``ssim_data_range=1`` explicitly fixes
    the uploaded source's implicit library default of 255 on normalized images.
    """

    def __init__(self, weights=None, device="cpu", fusion_reference="first_channel",
                 residual_weight=1.0, ssim_data_range=1.0):
        super().__init__()
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights is not None:
            unknown = set(weights) - set(self.weights)
            if unknown:
                raise ValueError("Unknown loss weights: {}".format(sorted(unknown)))
            self.weights.update({key: float(value) for key, value in weights.items()})
        if any(not math.isfinite(value) or value < 0 for value in self.weights.values()):
            raise ValueError("Loss weights must be finite and nonnegative")
        if fusion_reference not in ("first_channel", "luminance"):
            raise ValueError("fusion_reference must be first_channel or luminance")
        if not math.isfinite(float(residual_weight)) or residual_weight < 0:
            raise ValueError("residual_weight must be finite and nonnegative")
        if not math.isfinite(float(ssim_data_range)) or ssim_data_range <= 0:
            raise ValueError("ssim_data_range must be finite and positive")
        self.fusion_reference = fusion_reference
        self.residual_weight = float(residual_weight)
        # Set all five internal weights to one: apply the public weights exactly
        # once below, alongside the three added constraints.
        self.original = FusionLoss(
            lambda_distill_ir=1.0,
            lambda_distill_vis=1.0,
            lambda_adaptive_intensity=1.0,
            lambda_ssim_global=1.0,
            lambda_gradient=1.0,
            type_distill_ir="BCE",
            type_distill_vis="MSE",
            device=device,
        )
        self.original.L_SSIM = RangeAwareSSIM(ssim_data_range).to(device)

    def reference_channel(self, image):
        """Convert RGB/one-channel input to a one-channel fusion reference."""
        if image.ndim != 4 or image.shape[1] not in (1, 3):
            raise ValueError("Fusion references must be BCHW RGB or grayscale tensors")
        if image.shape[1] == 1 or self.fusion_reference == "first_channel":
            return image[:, :1]
        coeff = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (image * coeff).sum(dim=1, keepdim=True)

    @staticmethod
    def orthogonality(invariant, private):
        """Mean squared channel-wise cosine; a negative cosine is not rewarded."""
        if invariant.shape != private.shape:
            raise ValueError("Invariant/private feature shapes must match")
        invariant = F.normalize(invariant.float(), p=2, dim=1, eps=1e-6)
        private = F.normalize(private.float(), p=2, dim=1, eps=1e-6)
        return (invariant * private).sum(dim=1).square().mean()

    @staticmethod
    def _matching_shape(prediction, target, name):
        if prediction.shape != target.shape:
            raise ValueError("{} shape mismatch: {} versus {}".format(
                name, tuple(prediction.shape), tuple(target.shape)))

    def forward(self, degraded_outputs, clean_aux, batch, teacher_targets):
        # BCELoss is unsafe/disallowed in CUDA autocast. Using float32 for all
        # components also protects normalization, SSIM and low-amplitude losses.
        device_type = degraded_outputs["fused"].device.type
        with torch.autocast(device_type=device_type, enabled=False):
            return self._forward_float32(degraded_outputs, clean_aux, batch, teacher_targets)

    def _forward_float32(self, degraded_outputs, clean_aux, batch, teacher_targets):
        fused = degraded_outputs["fused"].float()
        clean_ir = batch["clean_ir"].float()
        clean_vis = batch["clean_vis"].float()
        ref_ir = self.reference_channel(clean_ir)
        ref_vis = self.reference_channel(clean_vis)
        self._matching_shape(fused, ref_ir, "IR fusion reference")
        self._matching_shape(fused, ref_vis, "VIS fusion reference")

        prompt_ir = degraded_outputs["prompt_ir"].float()
        if prompt_ir.ndim != 4 or prompt_ir.shape[0] != fused.shape[0]:
            raise ValueError("prompt_ir must be a BCHW tensor with the batch size")
        target_ir = teacher_targets["ir"].detach().float()
        if target_ir.ndim == 3:
            target_ir = target_ir.unsqueeze(1)
        if target_ir.ndim != 4 or target_ir.shape[:2] != (fused.shape[0], 1):
            raise ValueError("IR teacher target must have shape Bx1xHxW or BxHxW")
        target_ir = F.interpolate(target_ir, size=prompt_ir.shape[-2:],
                                  mode="bilinear", align_corners=False)
        score_vis = degraded_outputs["vis_score"].float()
        target_vis = teacher_targets["vis"].detach().float().reshape(fused.shape[0], -1)
        if score_vis.shape != (fused.shape[0], 1) or target_vis.shape != score_vis.shape:
            raise ValueError("Visible prediction/teacher must contain one score per image")
        if not torch.isfinite(target_ir).all() or not torch.isfinite(target_vis).all():
            raise ValueError("Teacher targets contain NaN or Inf")

        original = self.original(
            P_ir=prompt_ir.clamp(1e-6, 1.0 - 1e-6),
            GT_P_ir=target_ir[:, 0].clamp(0.0, 1.0),
            P_vis=score_vis,
            GT_P_vis=target_vis.clamp(0.0, 1.0),
            I_f=fused,
            I_vis=ref_vis,
            I_ir=ref_ir,
        )
        result = {
            "intensity": original["adaptive_intensity"],
            "ssim": original["ssim"],
            "gradient": original["gradient"],
            "distill_ir": original["distill_ir"],
            "distill_vis": original["distill_vis"],
        }

        consistency, orthogonal, content, residual = [], [], [], []
        for modality in ("ir", "vis"):
            degraded = degraded_outputs["aux"][modality]
            clean = clean_aux[modality]
            inv_d = degraded["inv"].float()
            inv_c = clean["inv"].float()
            self._matching_shape(inv_d, inv_c, modality + " consistency")
            consistency.append(F.mse_loss(inv_d, inv_c.detach()))
            target_clean = batch["clean_" + modality].float()
            target_residual = batch[modality].float() - target_clean
            for aux, residual_target in ((degraded, target_residual),
                                         (clean, torch.zeros_like(target_clean))):
                orthogonal.append(self.orthogonality(aux["inv"], aux["private"]))
                reconstructed = aux["clean_recon"].float()
                reconstructed_residual = aux["residual_recon"].float()
                self._matching_shape(reconstructed, target_clean, modality + " content reconstruction")
                self._matching_shape(reconstructed_residual, residual_target,
                                     modality + " residual reconstruction")
                content.append(F.l1_loss(reconstructed, target_clean))
                residual.append(F.l1_loss(reconstructed_residual, residual_target))

        result["consistency"] = torch.stack(consistency).mean()
        result["orthogonality"] = torch.stack(orthogonal).mean()
        result["content_reconstruction"] = torch.stack(content).mean()
        result["residual_reconstruction"] = torch.stack(residual).mean()
        result["self_supervision"] = (
            result["content_reconstruction"]
            + self.residual_weight * result["residual_reconstruction"]
        )
        result["total"] = sum(self.weights[name] * result[name] for name in self.weights)
        return result
