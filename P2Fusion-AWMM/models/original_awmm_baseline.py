"""Original P2Fusion architecture with a training return adapter; no adaptation modules."""
import torch

from .P2Fusion import p2fusion
from awmm.losses import AWMMLoss

FIVE_TERMS = ("intensity", "ssim", "gradient", "distill_ir", "distill_vis")


class OriginalP2FusionAWMM(p2fusion):
    is_original_baseline = True

    def forward(self, ir, vis, return_aux=False):
        if ir.shape != vis.shape:
            raise ValueError("Original P2Fusion requires aligned IR/VIS pairs")
        h, w = ir.shape[-2:]
        ir_features, ir_prompt = self.forward_features_ir(self.check_image_size(ir))
        vis_features, vis_prompt, vis_score = self.forward_features_vis(self.check_image_size(vis))
        fused_features = self.forward_features_Fusion(ir_features, vis_features, ir_prompt, vis_prompt)
        fused = self.forward_features_Re(fused_features[0], fused_features[1])
        return {"fused": fused[..., :h, :w], "prompt_ir": ir_prompt,
                "prompt_vis": vis_prompt, "vis_score": vis_score}

    def load_initial_checkpoint(self, path):
        from awmm.runtime import extract_state
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(extract_state(payload), strict=True)
        return {"loaded": len(self.state_dict()), "initialized_adaptation": []}


class OriginalFiveLoss(AWMMLoss):
    def __init__(self, weights=None, **kwargs):
        if weights is not None and set(weights) - set(FIVE_TERMS):
            raise ValueError("Original baseline allows only the five fusion/distillation loss weights")
        super().__init__(weights=weights, **kwargs)
        self.weights = {name: self.weights[name] for name in FIVE_TERMS}

    def _forward_float32(self, outputs, clean_aux, batch, targets):
        result = self.fusion_components(outputs, batch, targets)
        result["total"] = sum(self.weights[name] * result[name] for name in FIVE_TERMS)
        return result
