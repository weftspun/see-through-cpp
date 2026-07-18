#!/usr/bin/env python3
"""M8 reference: the Marigold depth stage end-to-end at 256px / 4 DDIM steps —
upstream MarigoldDepthPipeline with the frame-condition UNet on CPU f32.

The cond latents (page+layer VAE encodes incl. pyramid RGB bleed) are dumped
and injected into the C++ test — the bleed itself is gated separately in M10.
The init target noise is drawn from a seeded generator and dumped."""
import os
import sys

_ROOT = os.environ.get("SEETHROUGH_DIR", r"C:\Users\ernes\Desktop\see-through")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "common"))

import numpy as np
import torch

from modules.marigold import MarigoldDepthPipeline
from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel

REPO = "24yearsold/seethroughv0.0.1_marigold"
RES = 256
STEPS = 4


def make_layers():
    """3 synthetic RGBA layers + fullpage, deterministic."""
    yy, xx = np.mgrid[0:RES, 0:RES].astype(np.float32) / RES
    layers = []
    specs = [((0.3, 0.35), 0.05, (255, 200, 60)),   # face-ish disc
             ((0.5, 0.6), 0.12, (60, 120, 255)),    # body blob
             ((0.7, 0.3), 0.03, (250, 250, 250))]   # small highlight
    page = np.zeros((RES, RES, 4), dtype=np.uint8)
    for (cx, cy), r2, col in specs:
        m = (xx - cx) ** 2 + (yy - cy) ** 2 < r2
        img = np.zeros((RES, RES, 4), dtype=np.uint8)
        img[m] = list(col) + [255]
        layers.append(img)
        page[m] = list(col) + [255]
    layers.append(page)
    return layers


def main():
    unet = UNetFrameConditionModel.from_pretrained(REPO, subfolder="unet",
                                                   torch_dtype=torch.float32)
    pipe = MarigoldDepthPipeline.from_pretrained(REPO, unet=unet,
                                                 torch_dtype=torch.float32)
    pipe.cache_tag_embeds(unload_textencoders=False)

    img_list = make_layers()
    F = len(img_list)

    # record per-step v-prediction and latents for bisection
    from diffusers import DDIMScheduler
    step_eps, step_lat = [], []
    orig_step = DDIMScheduler.step

    def rec_step(self, model_output, *a, **k):
        step_eps.append(model_output.detach().clone())
        r = orig_step(self, model_output, *a, **k)
        step_lat.append(r.prev_sample.detach().clone())
        return r

    DDIMScheduler.step = rec_step

    # record the exact tensors entering the UNet
    unet_inputs = []
    orig_fwd = unet.forward

    def rec_fwd(sample, *a, **k):
        unet_inputs.append(sample.detach().clone())
        return orig_fwd(sample, *a, **k)

    unet.forward = rec_fwd

    # replicate single_infer's only generator draw so the C++ test can inject it
    g = torch.Generator().manual_seed(31)
    target_init = torch.randn((F, 4, RES // 8, RES // 8), generator=g)

    out = pipe(color_map=None, img_list=img_list, denoising_steps=STEPS,
               generator=torch.Generator().manual_seed(31), match_input_res=False)
    depth = out.depth_tensor                      # (F, H, W) in [0,1]
    print("depth:", tuple(depth.shape), "mean", float(depth.mean()))

    # rebuild the cond latents exactly as __call__ did (deterministic)
    from modules.marigold.marigold_depth_pipeline import encode_argb_list

    def _np_transform(img):
        arr = np.concatenate([img[..., 3:], img[..., :3]], axis=2).astype(np.float32) / 255.
        return torch.from_numpy(arr).movedim(-1, 0)

    tens = torch.stack([_np_transform(i) for i in img_list])
    lat = torch.cat([encode_argb_list(pipe.vae, i[None, None], pad_argb=True) for i in tens], dim=1)
    page_lat = encode_argb_list(pipe.vae, tens[-1][None][None], pad_argb=True)
    cond = torch.cat([page_lat.expand(-1, F, -1, -1, -1), lat], dim=2)[0]  # (F,8,h,w)

    ehs = pipe.empty_text_embed                   # (1, 2, 1024)
    arrays = [cond, ehs[0], target_init, depth.float()] + step_eps + step_lat \
        + [unet_inputs[0].squeeze(0)]
    with open("reference_marigold.bin", "wb") as f:
        for arr in arrays:
            a = arr.detach().numpy().astype("<f4")
            f.write(np.int32(a.ndim).tobytes())
            f.write(np.array(a.shape, dtype="<i8").tobytes())
            f.write(a.tobytes())
    print("wrote reference_marigold.bin,", len(arrays), "records")


if __name__ == "__main__":
    main()
