#!/usr/bin/env python3
"""Upstream-parity check: per-tag alpha-mask IoU between our own --png-dir
output (placed via its out.psd.json sidecar) and a real upstream PSD
(already expanded to per-layer PNGs by tests/psd_layers.cpp). See
docs/decisions/0008-open-work.md's "Upstream parity" checklist item.

Usage:
    python tools/psd_iou.py <our_png_dir> <our.psd.json> <upstream_layers_dir> [--thr 0.5]

Matches layers by tag name (our --png-dir's <NNN>_<tag>.png filenames vs.
the upstream PNGs' <NNN>_<tag>.png filenames, both already normalized to
the same alnum/-/_ charset). For each match, thresholds both alpha channels
and reports intersection-over-union on the full canvas.
"""
import argparse
import glob
import json
import os
import re
import sys

from PIL import Image
import numpy as np


def load_our_layers(png_dir, json_path):
    meta = json.load(open(json_path, encoding="utf-8"))
    h, w = meta["frame_size"]
    layers = {}
    for path in sorted(glob.glob(os.path.join(png_dir, "*.png"))):
        base = os.path.splitext(os.path.basename(path))[0]
        tag = re.sub(r"^\d+_", "", base)
        info = meta["parts"].get(tag)
        if info is None:
            continue
        x0, y0, x1, y1 = info["xyxy"]
        img = Image.open(path).convert("RGBA")
        alpha = np.zeros((h, w), dtype=np.uint8)
        alpha[y0:y1, x0:x1] = np.array(img)[:, :, 3]
        layers[tag] = alpha
    return w, h, layers


def load_upstream_layers(layers_dir):
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
    ap.add_argument("our_png_dir")
    ap.add_argument("our_psd_json")
    ap.add_argument("upstream_layers_dir")
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    ow, oh, ours = load_our_layers(args.our_png_dir, args.our_psd_json)
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
