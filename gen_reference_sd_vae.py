#!/usr/bin/env python3
"""Generate an SD/SDXL VAE decoder reference activation pair for validating
the ggml implementation: fixed seeded latent -> decoded image, saved as
reference_sd_vae.bin. Component picks the HF repo (layerdiff-vae = SDXL VAE,
marigold-vae = SD VAE); both share the AutoencoderKL architecture."""
import sys

import numpy as np
import torch
from diffusers import AutoencoderKL

REPOS = {
    "layerdiff-vae": "layerdifforg/seethroughv0.0.2_layerdiff3d",
    "marigold-vae": "24yearsold/seethroughv0.0.1_marigold",
}


def main():
    comp = sys.argv[1] if len(sys.argv) > 1 else "layerdiff-vae"
    res = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    vae = AutoencoderKL.from_pretrained(REPOS[comp], subfolder="vae")
    vae.eval().float()

    g = torch.Generator().manual_seed(42)
    z = torch.randn(1, 4, res // 8, res // 8, generator=g)
    with torch.no_grad():
        y = vae.decode(z).sample
    print("output:", tuple(y.shape), "mean", float(y.mean()), "std", float(y.std()))

    # simple binary: for each of z, y: i32 ndim, i64 dims..., f32 data
    out = f"reference_sd_vae.bin" if comp == "layerdiff-vae" else f"reference_{comp}.bin"
    with open(out, "wb") as f:
        for arr in (z, y):
            a = arr.numpy().astype("<f4")
            f.write(np.int32(a.ndim).tobytes())
            f.write(np.array(a.shape, dtype="<i8").tobytes())
            f.write(a.tobytes())
    print("wrote", out)


if __name__ == "__main__":
    main()
