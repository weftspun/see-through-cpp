#!/usr/bin/env python3
"""Generate a UNet1024 (TransparentVAE decoder head) reference activation pair
for validating the ggml implementation: fixed seeded inputs -> output, saved
as reference_trans_vae.npz. Run with the see-through repo on sys.path."""
import os
import sys

import os
os.makedirs('gen_reference', exist_ok=True)
import numpy as np
import torch

SEETHROUGH = os.environ.get("SEETHROUGH_DIR", r"C:\Users\ernes\Desktop\see-through")
sys.path.insert(0, os.path.join(SEETHROUGH, "common"))
sys.path.insert(0, os.path.join(SEETHROUGH, "common", "modules"))

from layerdiffuse.vae import UNet1024  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from safetensors.torch import load_file  # noqa: E402


def main():
    res = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    ckpt = hf_hub_download("layerdifforg/seethroughv0.0.2_layerdiff3d",
                           "trans_vae/diffusion_pytorch_model.safetensors")
    sd = load_file(ckpt)
    dec = {k[len("decoder.model."):]: v for k, v in sd.items()
           if k.startswith("decoder.model.")}
    in_ch = dec["conv_in.weight"].shape[1]
    out_ch = dec["conv_out.weight"].shape[0]
    print(f"UNet1024 in_channels={in_ch} out_channels={out_ch}, "
          f"{len(dec)} tensors, input {res}x{res}")

    model = UNet1024(in_channels=in_ch, out_channels=out_ch)
    missing, unexpected = model.load_state_dict(dec, strict=True), None
    model.eval().float()

    g = torch.Generator().manual_seed(42)
    x = torch.randn(1, in_ch, res, res, generator=g)
    latent = torch.randn(1, 4, res // 8, res // 8, generator=g)
    with torch.no_grad():
        y = model(x, latent)
    print("output:", tuple(y.shape), "mean", float(y.mean()), "std", float(y.std()))

    # simple binary: for each of x, latent, y: i32 ndim, i64 dims..., f32 data
    with open("gen_reference/reference_trans_vae.bin", "wb") as f:
        for arr in (x, latent, y):
            a = arr.numpy().astype("<f4")
            f.write(np.int32(a.ndim).tobytes())
            f.write(np.array(a.shape, dtype="<i8").tobytes())
            f.write(a.tobytes())
    print("wrote reference_trans_vae.bin")


if __name__ == "__main__":
    main()
