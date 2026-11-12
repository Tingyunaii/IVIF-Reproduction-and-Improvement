"""AWMM adaptation of P²Fusion.

The original model is inherited directly so its checkpoint parameter names stay
unchanged.  FDN and DWAM process spatial encoder features immediately before
the original ``proj``/``prompt_head`` split.  Auxiliary RGB reconstruction heads
are used for training only; inference needs only the degraded IR/VIS pair.
"""

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .P2Fusion import p2fusion


class FeatureDisentanglement(nn.Module):
    """Separate near-identity content features and degradation-related features."""

    def __init__(self, channels=60):
        super().__init__()
        groups = 6 if channels % 6 == 0 else 1
        self.invariant_residual = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        # A small, nonzero residual preserves pretrained feature scale without
        # preventing the earlier residual layers from learning on the first step.
        nn.init.normal_(self.invariant_residual[-1].weight, std=1e-3)
        nn.init.zeros_(self.invariant_residual[-1].bias)
        self.private_encoder = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, features):
        return features + self.invariant_residual(features), self.private_encoder(features)


class DynamicWeatherAwareness(nn.Module):
    """Weather-conditioned *relative enhancement*, following the agreed formula.

    The private input is detached only on the weather-gating path.  Its encoder
    still receives gradients from orthogonality and residual reconstruction.
    Since both gates are sigmoid outputs, the residual scales U by [1, 2]; this
    module alone is not an absolute attenuation or a guarantee of deweathering.
    """

    def __init__(self, channels=60, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.weather = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.region = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())

    def forward(self, invariant, private):
        weather = self.weather(private.detach())
        summary = torch.cat(
            [invariant.mean(dim=1, keepdim=True), invariant.amax(dim=1, keepdim=True)],
            dim=1,
        )
        region = self.region(summary)
        adjusted = invariant + invariant * weather * region
        return adjusted, weather, region


class AuxiliaryReconstruction(nn.Module):
    """A lightweight decoder with no raw-image/raw-feature skip connection."""

    def __init__(self, channels=60, signed=False):
        super().__init__()
        self.signed = signed
        self.decode = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def forward(self, features, output_size):
        rgb = self.decode(features)
        rgb = F.interpolate(rgb, size=output_size, mode="bilinear", align_corners=False)
        return torch.tanh(rgb) if self.signed else torch.sigmoid(rgb)


class AWMMP2Fusion(p2fusion):
    """P²Fusion + modality-specific FDN/DWAM and training-only auxiliary heads.

    ``forward(..., return_aux=True)`` returns ``aux['ir'|'vis']`` containing
    ``inv``, ``private``, gate maps, and RGB ``clean_recon``/``residual_recon``.
    ``encode_aux(clean_ir, clean_vis)`` returns that same auxiliary dictionary
    without running the fusion decoder.  Both weather conditions share weights.

    The original implementation hard-codes 60-dimensional fusion features,
    eight IR prompt channels, RGB inputs, 4x patchification and 8x8 cross windows;
    incompatible settings are rejected rather than failing later in a reshape.
    """

    def __init__(self, **kwargs):
        constraints = {
            "embed_dim": 60, "prompt_channels": 8, "in_chans": 3,
            "patch_size": 4, "window_size": 8,
        }
        for key, required in constraints.items():
            if kwargs.get(key, required) != required:
                raise ValueError("Original P²Fusion requires {}={}".format(key, required))
        if kwargs.get("num_blocks", 4) < 1:
            raise ValueError("num_blocks must be at least 1")
        super().__init__(**kwargs)
        self._base_state_keys = frozenset(self.state_dict())
        self.fdn_ir = FeatureDisentanglement(self.embed_dim)
        self.fdn_vis = FeatureDisentanglement(self.embed_dim)
        self.dwam_ir = DynamicWeatherAwareness(self.embed_dim)
        self.dwam_vis = DynamicWeatherAwareness(self.embed_dim)
        self.clean_reconstructor_ir = AuxiliaryReconstruction(self.embed_dim)
        self.clean_reconstructor_vis = AuxiliaryReconstruction(self.embed_dim)
        self.residual_reconstructor_ir = AuxiliaryReconstruction(self.embed_dim, signed=True)
        self.residual_reconstructor_vis = AuxiliaryReconstruction(self.embed_dim, signed=True)

    @property
    def adaptation_parameter_prefixes(self):
        return (
            "fdn_ir.", "fdn_vis.", "dwam_ir.", "dwam_vis.",
            "clean_reconstructor_ir.", "clean_reconstructor_vis.",
            "residual_reconstructor_ir.", "residual_reconstructor_vis.",
        )

    def check_image_size(self, image, downsample_ratio=4):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected RGB tensor [B,3,H,W]")
        height, width = image.shape[-2:]
        if height < 1 or width < 1:
            raise ValueError("Image dimensions must be positive")
        multiple = self.window_size * downsample_ratio
        pad_h, pad_w = (-height) % multiple, (-width) % multiple
        mode = "reflect" if pad_h < height and pad_w < width else "replicate"
        return F.pad(image, (0, pad_w, 0, pad_h), mode=mode)

    def _encode_spatial(self, image):
        features = self.conv_first(image)
        size = (features.shape[-2] // self.patch_size, features.shape[-1] // self.patch_size)
        tokens = self.norm_Ex_ir(self.rstb(self.patch_embed(features), size))
        spatial = tokens.transpose(1, 2).reshape(tokens.shape[0], self.embed_dim, *size)
        return spatial, size

    def _encode_modality(self, image, modality, original_size, return_aux, run_heads=True):
        spatial, size = self._encode_spatial(image)
        fdn = getattr(self, "fdn_" + modality)
        dwam = getattr(self, "dwam_" + modality)
        invariant, private = fdn(spatial)
        adjusted, weather, region = dwam(invariant, private)
        # This conversion happens before either original unembedding head sees
        # the features.  Both proj and prompt_head consume the same adjusted map.
        result = None
        if run_heads:
            tokens = adjusted.flatten(2).transpose(1, 2).contiguous()
            unembed = self.patch_unembed if modality == "ir" else self.patch_unembed_vis
            result = unembed(tokens, size)
        aux = None
        if return_aux:
            padded_size = image.shape[-2:]
            h, w = original_size
            clean = getattr(self, "clean_reconstructor_" + modality)(invariant, padded_size)
            residual = getattr(self, "residual_reconstructor_" + modality)(private, padded_size)
            aux = {
                "inv": invariant,
                "private": private,
                "weather_gate": weather,
                "region_gate": region,
                "clean_recon": clean[..., :h, :w],
                "residual_recon": residual[..., :h, :w],
            }
        return result, aux

    @staticmethod
    def _validate_pair(ir, vis):
        if ir.shape != vis.shape:
            raise ValueError("IR/VIS must be registered RGB tensors of equal shape")

    def forward_features_ir(self, image):
        padded = self.check_image_size(image)
        result, _ = self._encode_modality(padded, "ir", image.shape[-2:], False)
        return result

    def forward_features_vis(self, image):
        padded = self.check_image_size(image)
        result, _ = self._encode_modality(padded, "vis", image.shape[-2:], False)
        return result

    def encode_aux(self, ir, vis):
        """Return clean-branch auxiliary outputs without fusion reconstruction."""
        self._validate_pair(ir, vis)
        size = ir.shape[-2:]
        _, ir_aux = self._encode_modality(self.check_image_size(ir), "ir", size, True, run_heads=False)
        _, vis_aux = self._encode_modality(self.check_image_size(vis), "vis", size, True, run_heads=False)
        return {"ir": ir_aux, "vis": vis_aux}

    def forward(self, ir, vis, return_aux=False):
        self._validate_pair(ir, vis)
        size = ir.shape[-2:]
        (ir_features, ir_prompt), ir_aux = self._encode_modality(
            self.check_image_size(ir), "ir", size, return_aux
        )
        (vis_features, vis_prompt, vis_score), vis_aux = self._encode_modality(
            self.check_image_size(vis), "vis", size, return_aux
        )
        fused_features = self.forward_features_Fusion(ir_features, vis_features, ir_prompt, vis_prompt)
        fused = self.forward_features_Re(fused_features[0], fused_features[1])
        output = {
            "fused": fused[..., :size[0], :size[1]],
            "prompt_ir": ir_prompt,
            "prompt_vis": vis_prompt,
            "vis_score": vis_score,
        }
        if return_aux:
            output["aux"] = {"ir": ir_aux, "vis": vis_aux}
        return output

    def load_initial_checkpoint(self, path):
        """Load an original P²Fusion checkpoint, rejecting incompatible weights.

        A missing adaptation parameter is expected for an original checkpoint.
        Any missing original parameter, unknown key, or incompatible shape is an
        error.  For full AWMM resume/inference use ``load_state_dict(strict=True)``.
        """
        payload = torch.load(path, map_location="cpu")
        state = payload
        if isinstance(payload, Mapping):
            for name in ("params", "params_ema", "state_dict", "model"):
                if name in payload and isinstance(payload[name], Mapping):
                    state = payload[name]
                    break
        if not isinstance(state, Mapping) or not state:
            raise ValueError("Checkpoint does not contain a nonempty state dictionary")
        stripped = {}
        for key, value in state.items():
            if not isinstance(key, str) or not isinstance(value, torch.Tensor):
                raise ValueError("Checkpoint state dictionary must map names to tensors")
            while key.startswith("module."):
                key = key[len("module."):]
            if key in stripped:
                raise ValueError("Duplicate key after removing module prefix: " + key)
            stripped[key] = value
        own_state = self.state_dict()
        unknown = sorted(set(stripped) - set(own_state))
        missing_original = sorted(self._base_state_keys - set(stripped))
        mismatched = sorted(
            key for key in set(stripped) & set(own_state)
            if stripped[key].shape != own_state[key].shape
        )
        if unknown or missing_original or mismatched:
            raise ValueError(
                "Incompatible P²Fusion checkpoint: unknown={}, missing_original={}, shape_mismatch={}".format(
                    unknown, missing_original, mismatched
                )
            )
        result = self.load_state_dict(stripped, strict=False)
        return {"loaded": len(stripped), "initialized_adaptation": sorted(result.missing_keys)}
