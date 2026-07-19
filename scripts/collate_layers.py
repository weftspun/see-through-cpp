#!/usr/bin/env python3
"""Upstream parity check: match our SVG's per-tag layers against a PSD
produced by upstream's `inference_psd.py --save_to_psd` on the same input/
seed, and report per-tag alpha-mask IoU + IoU-implied depth ordering drift.

Upstream only emits PSD (no SVG writer there, and we deliberately don't
reintroduce a PSD writer here just for this comparison -- see MADR 0006).
This script is the one place that needs psd-tools; it's a scripts/ only dep,
not a pipeline dependency.

Usage:
    uv run --with psd-tools --with numpy --with pillow \
        python scripts/collate_layers.py ours.svg upstream.psd
"""
import argparse
import base64
import io
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

SVG_NS = {"svg": "http://www.w3.org/2000/svg"}


def normalize_tag(name: str) -> str:
    """Canonical form for matching tag names across the two pipelines'
    naming conventions (spaces/underscores/hyphens, left/right suffixes)."""
    n = name.strip().lower()
    n = n.replace("_", " ").replace("-", " ")
    n = re.sub(r"\s+", " ", n).strip()
    # left/right suffix normalization: "eyewhite l" / "eyewhite left" -> "eyewhite-l"
    n = re.sub(r"\b(left)\b", "l", n)
    n = re.sub(r"\b(right)\b", "r", n)
    parts = n.split(" ")
    if len(parts) >= 2 and parts[-1] in ("l", "r"):
        n = "-".join(["".join(parts[:-1]), parts[-1]])
    else:
        n = "".join(parts)
    return n


def load_svg_layers(svg_path):
    """Returns {normalized_tag: (rgba uint8 HxWx4, x, y)} for every visible
    (non-depth-group) <image> in document order, skipping the hidden depth
    group entirely."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    layers = {}
    for img in root.findall("svg:image", SVG_NS):
        tag = img.get("data-tag")
        href = img.get("{http://www.w3.org/1999/xlink}href")
        if tag is None or href is None:
            continue
        b64 = href.split(",", 1)[1]
        png_bytes = base64.b64decode(b64)
        im = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGBA"))
        x = int(float(img.get("x", "0")))
        y = int(float(img.get("y", "0")))
        layers[normalize_tag(tag)] = (im, x, y)
    return layers


def load_psd_layers(psd_path):
    from psd_tools import PSDImage

    psd = PSDImage.open(psd_path)
    layers = {}
    for layer in psd:
        if layer.numpy() is None:
            continue
        arr = layer.numpy()
        rgba = np.round(arr * 255).astype(np.uint8)
        if rgba.shape[2] == 3:
            alpha = np.full(rgba.shape[:2] + (1,), 255, dtype=np.uint8)
            rgba = np.concatenate([rgba, alpha], axis=2)
        layers[normalize_tag(layer.name)] = (rgba, layer.left, layer.top)
    return layers


def alpha_mask(rgba, x, y, canvas_size, thr=15):
    """Place this layer's alpha channel (thresholded to a binary mask) onto
    a canvas_size x canvas_size canvas at (x,y), for IoU comparison across
    layers whose crops may differ slightly in position/size."""
    h, w = rgba.shape[:2]
    canvas = np.zeros((canvas_size, canvas_size), dtype=bool)
    y1, x1 = min(y + h, canvas_size), min(x + w, canvas_size)
    if y1 <= y or x1 <= x:
        return canvas
    canvas[y:y1, x:x1] = rgba[: y1 - y, : x1 - x, 3] > thr
    return canvas


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else inter / union


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("svg_path")
    ap.add_argument("psd_path")
    ap.add_argument("--canvas-size", type=int, default=1280)
    ap.add_argument("--iou-threshold", type=float, default=0.98)
    args = ap.parse_args()

    ours = load_svg_layers(args.svg_path)
    theirs = load_psd_layers(args.psd_path)

    ours_tags, their_tags = set(ours), set(theirs)
    matched = sorted(ours_tags & their_tags)
    only_ours = sorted(ours_tags - their_tags)
    only_theirs = sorted(their_tags - ours_tags)

    print(f"matched tags: {len(matched)}  ours-only: {len(only_ours)}  "
          f"upstream-only: {len(only_theirs)}")
    if only_ours:
        print(f"  ours-only: {only_ours}")
    if only_theirs:
        print(f"  upstream-only: {only_theirs}")

    fails = 0
    ious = []
    for tag in matched:
        o_rgba, ox, oy = ours[tag]
        t_rgba, tx, ty = theirs[tag]
        o_mask = alpha_mask(o_rgba, ox, oy, args.canvas_size)
        t_mask = alpha_mask(t_rgba, tx, ty, args.canvas_size)
        score = iou(o_mask, t_mask)
        ious.append(score)
        status = "ok" if score >= args.iou_threshold else "FAIL"
        if status == "FAIL":
            fails += 1
        print(f"  {tag}: IoU={score:.4f} {status}")

    if ious:
        print(f"mean IoU over matched tags: {np.mean(ious):.4f}")
    print("VALIDATION FAIL" if fails else "VALIDATION PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
