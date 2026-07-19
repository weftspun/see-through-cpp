#!/usr/bin/env python3
"""Upstream-parity check: per-tag alpha-mask IoU between our own --png-dir
SVG output and a real upstream PSD (already expanded to per-layer PNGs by
tests/psd_layers.cpp). See docs/decisions/0008-open-work.md's
"Upstream parity" checklist item.

Usage:
    python tools/psd_iou.py <our.svg> <upstream_layers_dir> [--thr 0.5]

Matches layers by tag name (our SVG's data-tag attribute vs. the upstream
PNGs' <NNN>_<tag>.png filenames, both already normalized to the same
alnum/-/_ charset). For each match, thresholds both alpha channels and
reports intersection-over-union on the full canvas.
"""
import argparse
import base64
import io
import re
import sys

from PIL import Image
import numpy as np


def load_our_layers(svg_path):
    svg = open(svg_path, encoding="utf-8").read()
    canvas = re.search(r'width="(\d+)" height="(\d+)"', svg)
    w, h = int(canvas.group(1)), int(canvas.group(2))
    layers = {}
    pattern = (
        r'<image id="([^"]+)" x="(-?\d+)" y="(-?\d+)" width="(\d+)" height="(\d+)"'
        r'[^>]*xlink:href="data:image/png;base64,([^"]+)"'
    )
    for m in re.finditer(pattern, svg):
        tag, x, y, iw, ih, data = m.groups()
        if tag.startswith("depth-"):
            continue
        img = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGBA")
        alpha = np.zeros((h, w), dtype=np.uint8)
        crop_alpha = np.array(img)[:, :, 3]
        x, y = int(x), int(y)
        alpha[y:y + int(ih), x:x + int(iw)] = crop_alpha
        layers[tag] = alpha
    return w, h, layers


def load_upstream_layers(layers_dir):
    import glob
    import os

    layers = {}
    for path in sorted(glob.glob(os.path.join(layers_dir, "*.png"))):
        base = os.path.splitext(os.path.basename(path))[0]
        tag = re.sub(r"^\d+_", "", base)
        img = Image.open(path).convert("RGBA")
        layers[tag] = np.array(img)[:, :, 3]
    return layers


def iou(a, b, thr):
    am = a >= int(thr * 255)
    bm = b >= int(thr * 255)
    inter = np.logical_and(am, bm).sum()
    union = np.logical_or(am, bm).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("our_svg")
    ap.add_argument("upstream_layers_dir")
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    ow, oh, ours = load_our_layers(args.our_svg)
    theirs = load_upstream_layers(args.upstream_layers_dir)

    matched = sorted(set(ours) & set(theirs))
    only_ours = sorted(set(ours) - set(theirs))
    only_theirs = sorted(set(theirs) - set(ours))

    scores = []
    print(f"canvas: ours {ow}x{oh}")
    print(f"{'tag':<16} {'IoU':>8}")
    for tag in matched:
        a, b = ours[tag], theirs[tag]
        if a.shape != b.shape:
            print(f"{tag:<16} {'shape mismatch ' + str(a.shape) + ' vs ' + str(b.shape)}")
            continue
        score = iou(a, b, args.thr)
        scores.append(score)
        print(f"{tag:<16} {score:>8.4f}")

    if only_ours:
        print(f"\nonly in ours ({len(only_ours)}): {', '.join(only_ours)}")
    if only_theirs:
        print(f"only in upstream ({len(only_theirs)}): {', '.join(only_theirs)}")

    if scores:
        print(f"\nmatched tags: {len(scores)}/{len(matched)}")
        print(f"mean IoU: {sum(scores) / len(scores):.4f}")
        print(f"min IoU: {min(scores):.4f}")
    else:
        print("\nno matching tags -- nothing to compare")
        sys.exit(1)


if __name__ == "__main__":
    main()
