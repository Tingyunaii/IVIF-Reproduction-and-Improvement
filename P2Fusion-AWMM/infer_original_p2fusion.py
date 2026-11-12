"""Original P2Fusion checkpoint baseline, with uploaded-original Y/color protocol."""
import argparse
from pathlib import Path
import torch
from PIL import Image
from models.P2Fusion import p2fusion
from awmm.runtime import extract_state
from awmm.preprocessing import original_color_restore, fused_y_uint8
from infer_awmm import image_index, image_tensor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--ir-dir', required=True)
    p.add_argument('--vis-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--save-y-dir', default=None, help='Also save quantized decoder Y before color restoration')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    device = torch.device(args.device)
    model = p2fusion().to(device)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(extract_state(payload), strict=True)
    model.eval()
    ir, vis = image_index(args.ir_dir), image_index(args.vis_dir)
    if not ir or set(ir) != set(vis):
        raise ValueError('Nonempty IR/VIS sample IDs must match exactly')
    out = Path(args.output)
    y_out = Path(args.save_y_dir) if args.save_y_dir else None
    if y_out is not None:
        if y_out.resolve() == out.resolve():
            raise ValueError('--save-y-dir must differ from --output')
        y_out.mkdir(parents=True, exist_ok=False)
    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for stem in sorted(ir):
            a = image_tensor(ir[stem], 'original_y').to(device)
            b = image_tensor(vis[stem], 'original_y').to(device)
            if a.shape != b.shape:
                raise ValueError('Image dimensions differ: '+stem)
            fused_y = model(a, b)[0]
            if y_out is not None:
                Image.fromarray(fused_y_uint8(fused_y)).save(y_out/(stem+'.png'))
            Image.fromarray(original_color_restore(fused_y, vis[stem])).save(out/(stem+'.png'))
    print('Original P2Fusion baseline saved:', len(ir), out)


if __name__ == '__main__':
    main()
