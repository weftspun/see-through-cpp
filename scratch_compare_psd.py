#!/usr/bin/env python3
"""Per-tag alpha comparison between two psd_layers.exe output dirs.

Usage: python scratch_compare_psd.py <dirA> <dirB>
Matches layers by tag (filename after NNN_), reports nonzero-alpha pixel
counts and IoU (alpha>0.125 threshold) on the shared canvas.
"""
import glob, os, re, sys
import numpy as np
from PIL import Image

def load(d):
    out = {}
    for p in glob.glob(os.path.join(d, "*.png")):
        tag = re.sub(r"^\d+_", "", os.path.basename(p)[:-4])
        out[tag] = np.array(Image.open(p).convert("RGBA"))[:, :, 3] > 32
    return out

a, b = load(sys.argv[1]), load(sys.argv[2])
tags = sorted(set(a) | set(b))
print(f"{'tag':22s} {'A_px':>8s} {'B_px':>8s} {'IoU':>6s}")
for t in tags:
    ma = a.get(t); mb = b.get(t)
    pa = int(ma.sum()) if ma is not None else -1
    pb = int(mb.sum()) if mb is not None else -1
    iou = ""
    if ma is not None and mb is not None:
        u = (ma | mb).sum()
        iou = f"{(ma & mb).sum() / u:.3f}" if u else "empty"
    print(f"{t:22s} {pa:8d} {pb:8d} {iou:>6s}")
