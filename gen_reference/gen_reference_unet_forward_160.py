#!/usr/bin/env python3
"""1280px collapse root-cause check (docs/ggml-upstream-issues.md #4): the
same M5 reference as gen_reference_unet_forward.py, but at the ACTUAL
production latent size for res=1280 (ZR=160), not the 64x64 the existing
gate uses ("validates small" per that file's own docstring). The whole
1280px investigation so far has only ever compared our ggml implementation
against itself under different knobs (direct_conv/flash_attn/row-chunk/
cross_frame_block all excluded) -- never against the real upstream
PyTorch reference AT the actual problem shape. This closes that gap."""
import os
import sys

_ROOT = os.environ.get("SEETHROUGH_DIR", r"C:\Users\ernes\Desktop\see-through")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "common"))

import os
os.makedirs('gen_reference', exist_ok=True)
import numpy as np
import torch

from common.modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel

REPO = "layerdifforg/seethroughv0.0.2_layerdiff3d"
F = 13
ZR = 160  # res=1280 / 8


def main():
    unet = UNetFrameConditionModel.from_pretrained(REPO, subfolder="unet",
                                                   torch_dtype=torch.float32)
    unet.eval()

    g = torch.Generator().manual_seed(11)
    sample = torch.randn(1, F, 8, ZR, ZR, generator=g)
    ehs = torch.randn(F, 77, 2048, generator=g)
    text_embeds = torch.randn(F, 1280, generator=g)
    time_ids = torch.tensor([1280.0, 1280.0, 0.0, 0.0, 1280.0, 1280.0]).repeat(F, 1)

    taps = []

    def tap(mod):
        def hook(_m, _i, out):
            taps.append((out[0] if isinstance(out, tuple) else out).detach().clone())
        mod.register_forward_hook(hook)

    tap(unet.conv_in)
    tap(unet.down_blocks[0].resnets[0])  # down_blocks[0] is plain DownBlock2D
                                         # (no attentions) -- only checkpoint
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

    with open("gen_reference/reference_unet_forward_160.bin", "wb") as f:
        for arr in [sample, ehs, text_embeds, out] + taps:
            a = arr.numpy().astype("<f4")
            f.write(np.int32(a.ndim).tobytes())
            f.write(np.array(a.shape, dtype="<i8").tobytes())
            f.write(a.tobytes())
    print("wrote reference_unet_forward_160.bin")


if __name__ == "__main__":
    main()
