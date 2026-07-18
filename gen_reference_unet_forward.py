#!/usr/bin/env python3
"""M5 reference: the full LayerDiff UNetFrameConditionModel forward at 512px
latent (64x64, fully convolutional — validates small), 13 frames, group 0,
fixed timestep 999. Mirrors the real call in diffusers_kdiffusion_sdxl.py:
group_embeds2 on encoder_hidden_states and group_embeds on text_embeds are
applied INSIDE the model forward via group_index."""
import os
import sys

_ROOT = os.environ.get("SEETHROUGH_DIR", r"C:\Users\ernes\Desktop\see-through")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "common"))

import numpy as np
import torch

from common.modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel

REPO = "layerdifforg/seethroughv0.0.2_layerdiff3d"
F = 13


def main():
    unet = UNetFrameConditionModel.from_pretrained(REPO, subfolder="unet",
                                                   torch_dtype=torch.float32)
    unet.eval()

    g = torch.Generator().manual_seed(11)
    sample = torch.randn(1, F, 8, 64, 64, generator=g)
    ehs = torch.randn(F, 77, 2048, generator=g)
    text_embeds = torch.randn(F, 1280, generator=g)
    time_ids = torch.tensor([512.0, 512.0, 0.0, 0.0, 512.0, 512.0]).repeat(F, 1)

    taps = []

    def tap(mod):
        def hook(_m, _i, out):
            taps.append((out[0] if isinstance(out, tuple) else out).detach().clone())
        mod.register_forward_hook(hook)

    tap(unet.conv_in)
    for db in unet.down_blocks:
        tap(db)
    tap(unet.mid_block)

    with torch.no_grad():
        out = unet(sample, torch.tensor(999.0), encoder_hidden_states=ehs,
                   added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                   group_index=0, return_dict=False)[0]
    print("out:", tuple(out.shape), "mean", float(out.mean()), "std", float(out.std()))
    for t in taps:
        print("tap:", tuple(t.shape), float(t.mean()))

    with open("reference_unet_forward.bin", "wb") as f:
        for arr in [sample, ehs, text_embeds, out] + taps:
            a = arr.numpy().astype("<f4")
            f.write(np.int32(a.ndim).tobytes())
            f.write(np.array(a.shape, dtype="<i8").tobytes())
            f.write(a.tobytes())
    print("wrote reference_unet_forward.bin")


if __name__ == "__main__":
    main()
