#!/usr/bin/env python3
"""Generate an AutoencoderKL *encoder* reference: fixed seeded image ->
latent_dist.mean (deterministic; the port never uses .sample()). Covers both
layerdiff-vae (SDXL) and marigold-vae (SD)."""
import sys

import os
os.makedirs('gen_reference', exist_ok=True)
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

    g = torch.Generator().manual_seed(43)
    x = torch.rand(1, 3, res, res, generator=g) * 2 - 1
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mean
    print("latent:", tuple(z.shape), "mean", float(z.mean()), "std", float(z.std()))

    out = f"gen_reference/reference_vae_encode_{comp}.bin"
    with open(out, "wb") as f:
        for arr in (x, z):
            a = arr.numpy().astype("<f4")
            f.write(np.int32(a.ndim).tobytes())
            f.write(np.array(a.shape, dtype="<i8").tobytes())
            f.write(a.tobytes())
    print("wrote", out)


if __name__ == "__main__":
    main()
