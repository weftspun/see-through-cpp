#!/usr/bin/env python3
"""Generate a full TransparentVAE decode reference for validating the chained
ggml graph (SDXL VAE decode -> *0.5+0.5 -> UNet1024 -> clip): seeded latent ->
RGBA, via the upstream TransparentVAEDecoder.forward itself, saved as
reference_trans_vae_full.bin. Run with the see-through repo on sys.path."""
import os
import sys

import numpy as np
import torch

SEETHROUGH = os.environ.get("SEETHROUGH_DIR", r"C:\Users\ernes\Desktop\see-through")
sys.path.insert(0, os.path.join(SEETHROUGH, "common"))
sys.path.insert(0, os.path.join(SEETHROUGH, "common", "modules"))

from layerdiffuse.vae import TransparentVAEDecoder  # noqa: E402
from diffusers import AutoencoderKL  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

REPO = "layerdifforg/seethroughv0.0.2_layerdiff3d"


def main():
    res = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    vae = AutoencoderKL.from_pretrained(REPO, subfolder="vae").eval().float()

    ckpt = hf_hub_download(REPO, "trans_vae/diffusion_pytorch_model.safetensors")
    sd = load_file(ckpt)
    dec = {k[len("decoder.model."):]: v for k, v in sd.items()
           if k.startswith("decoder.model.")}
    trans = TransparentVAEDecoder()
    trans.model.load_state_dict(dec, strict=True)
    trans.eval().float()

    g = torch.Generator().manual_seed(42)
    z = torch.randn(1, 4, res // 8, res // 8, generator=g)
    result_list, _ = trans(vae, z, return_type='tensor')
    y = result_list[0]
    print("output:", tuple(y.shape), "mean", float(y.mean()), "std", float(y.std()))

    # simple binary: for each of z, y: i32 ndim, i64 dims..., f32 data
    with open("reference_trans_vae_full.bin", "wb") as f:
        for arr in (z, y):
            a = arr.numpy().astype("<f4")
            f.write(np.int32(a.ndim).tobytes())
            f.write(np.array(a.shape, dtype="<i8").tobytes())
            f.write(a.tobytes())
    print("wrote reference_trans_vae_full.bin")


if __name__ == "__main__":
    main()
