#!/usr/bin/env python3
"""1280px collapse: does TILING fix it? (docs/ggml-upstream-issues.md #4)

This VAE's own trained config is sample_size=512 / tile_sample_min_size=512
-- diffusers ships AutoencoderKL.enable_tiling() specifically for running
beyond that size, but neither upstream's real pipeline nor this project's
own gen_reference_vae_encode_160_taps.py ever call it, so every prior
comparison in this investigation ran the encoder 2.5x past its native
resolution, fully untiled, on both sides. That is very likely *why*
mid_block.resnets.1 hits such an extreme near-cancelling activation
regime in the first place (confirmed present in upstream's own untiled
reference too, not just our ggml port) -- tiling keeps every tile within
the model's actual trained scale. Same seed/input as
gen_reference_vae_encode_160_taps.py for a direct before/after comparison."""
import os

import numpy as np
import torch
from diffusers import AutoencoderKL

REPO = "layerdifforg/seethroughv0.0.2_layerdiff3d"
RES = 1280


def main():
    os.makedirs('gen_reference', exist_ok=True)
    vae = AutoencoderKL.from_pretrained(REPO, subfolder="vae")
    vae.eval().float()
    vae.enable_tiling()

    g = torch.Generator().manual_seed(43)
    x = torch.rand(1, 3, RES, RES, generator=g) * 2 - 1

    with torch.no_grad():
        z = vae.encode(x).latent_dist.mean
    print("latent:", tuple(z.shape), "mean", float(z.mean()), "std", float(z.std()))

    out = "gen_reference/reference_vae_encode_160_tiled.bin"
    with open(out, "wb") as f:
        for arr in [x, z]:
            a = arr.numpy().astype("<f4")
            f.write(np.int32(a.ndim).tobytes())
            f.write(np.array(a.shape, dtype="<i8").tobytes())
            f.write(a.tobytes())
    print("wrote", out)


if __name__ == "__main__":
    main()
