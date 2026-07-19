#!/usr/bin/env python3
"""1280px collapse: isolate vae_decode_tiled from unet1024 (docs/ggml-
upstream-issues.md #4). After fixing vae_encode via tiling (confirmed: the
generated latent now has real structure), the full pipeline's decoded
output is still all-zero. This generates a real upstream reference for
JUST the AutoencoderKL tiled decode (vae.enable_tiling(); vae.decode(z)),
at the actual production shape (z: 1,4,160,160 -> pixel: 1,3,1280,1280),
to check whether vae_decode_tiled itself is now correct (pointing at
unet1024 as the remaining culprit) or still broken."""
import os

import numpy as np
import torch
from diffusers import AutoencoderKL

REPO = "layerdifforg/seethroughv0.0.2_layerdiff3d"
ZR = 160


def main():
    os.makedirs('gen_reference', exist_ok=True)
    vae = AutoencoderKL.from_pretrained(REPO, subfolder="vae")
    vae.eval().float()
    vae.enable_tiling()

    g = torch.Generator().manual_seed(7)
    z = torch.randn(1, 4, ZR, ZR, generator=g)

    with torch.no_grad():
        pixel = vae.decode(z).sample
    print("pixel:", tuple(pixel.shape), "mean", float(pixel.mean()), "std", float(pixel.std()))

    out = "gen_reference/reference_vae_decode_160_tiled.bin"
    with open(out, "wb") as f:
        for arr in [z, pixel]:
            a = arr.numpy().astype("<f4")
            f.write(np.int32(a.ndim).tobytes())
            f.write(np.array(a.shape, dtype="<i8").tobytes())
            f.write(a.tobytes())
    print("wrote", out)


if __name__ == "__main__":
    main()
