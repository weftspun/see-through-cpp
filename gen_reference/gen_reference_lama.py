#!/usr/bin/env python3
"""M9 reference: LaMa generator (large arch, no MPE) on a synthetic 256px
image + blob mask, CPU f32, composited output as the upstream eval path
produces it."""
import os
import sys

_ROOT = os.environ.get("SEETHROUGH_DIR", r"C:\Users\ernes\Desktop\see-through")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "common"))

import os
os.makedirs('gen_reference', exist_ok=True)
import numpy as np
import torch

from annotators.lama_inpainter import load_lama_mpe

RES = 256


def main():
    model = load_lama_mpe(device="cpu", use_mpe=False, large_arch=True)

    yy, xx = np.mgrid[0:RES, 0:RES].astype(np.float32) / RES
    img = np.stack([xx, yy, (xx + yy) / 2], axis=0)[None]          # (1,3,H,W) [0,1]
    img += 0.1 * np.sin(xx * 40)[None, None]
    img = np.clip(img, 0, 1)
    mask = (((xx - 0.5) ** 2 + (yy - 0.5) ** 2) < 0.04).astype(np.float32)[None, None]

    img_t = torch.from_numpy(img)
    mask_t = torch.from_numpy(mask)
    with torch.no_grad():
        out = model(img_t * (1 - mask_t), mask_t)
    print("out:", tuple(out.shape), "mean", float(out.mean()))

    with open("gen_reference/reference_lama.bin", "wb") as f:
        for arr in (img_t, mask_t, out):
            a = arr.numpy().astype("<f4")
            f.write(np.int32(a.ndim).tobytes())
            f.write(np.array(a.shape, dtype="<i8").tobytes())
            f.write(a.tobytes())
    print("wrote reference_lama.bin")


if __name__ == "__main__":
    main()
