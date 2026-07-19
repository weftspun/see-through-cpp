#!/usr/bin/env python3
"""1280px collapse root-cause check (docs/ggml-upstream-issues.md #4): the
SDXL VAE *encoder* (encode the "page" conditioning image) at the actual
production shape, res=1280, with per-stage hooks -- this path was never
exercised at this resolution by any prior gate (gen_reference_vae_encode.py
only covers up to res=256 by default). Found via the layerdiff e2e 1280px/
8-step reference (gen_reference_layerdiff_160.py): c_concat (the encoded
page latent) diverges wildly from our ggml output at this resolution, even
with direct_conv/conv_row_chunk disabled for the VAE (matching production)
and the attn_block tiling fix enabled (which turned out to be a no-op here:
the encoder's mid_block attention runs single-head, head_dim=0, well under
the Vulkan buffer-overflow threshold already fixed elsewhere)."""
import os
import sys

import numpy as np
import torch
from diffusers import AutoencoderKL

REPO = "layerdifforg/seethroughv0.0.2_layerdiff3d"
RES = 1280


def main():
    os.makedirs('gen_reference', exist_ok=True)
    vae = AutoencoderKL.from_pretrained(REPO, subfolder="vae")
    vae.eval().float()

    g = torch.Generator().manual_seed(43)
    x = torch.rand(1, 3, RES, RES, generator=g) * 2 - 1

    taps = []

    def tap(mod):
        def hook(_m, _i, out):
            taps.append((out[0] if isinstance(out, tuple) else out).detach().clone())
        mod.register_forward_hook(hook)

    enc = vae.encoder
    tap(enc.conv_in)
    for db in enc.down_blocks:
        tap(db)
    tap(enc.mid_block.resnets[0])
    tap(enc.mid_block.attentions[0])
    tap(enc.mid_block.resnets[1])

    with torch.no_grad():
        z = vae.encode(x).latent_dist.mean
    print("latent:", tuple(z.shape), "mean", float(z.mean()), "std", float(z.std()))
    for t in taps:
        print("tap:", tuple(t.shape), float(t.mean()))

    out = "gen_reference/reference_vae_encode_160_taps.bin"
    with open(out, "wb") as f:
        for arr in [x, z] + taps:
            a = arr.numpy().astype("<f4")
            f.write(np.int32(a.ndim).tobytes())
            f.write(np.array(a.shape, dtype="<i8").tobytes())
            f.write(a.tobytes())
    print("wrote", out)


if __name__ == "__main__":
    main()
