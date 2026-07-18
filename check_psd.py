#!/usr/bin/env python3
"""M11 gate: open a PSD written by write_psd with psd-tools and require exact
round-trip of layer names, offsets and pixels against the raw dump emitted by
test_psd (mirrors upstream's psd2partdicts access pattern)."""
import struct
import sys

import numpy as np
from psd_tools import PSDImage


def main():
    psd_path, raw_path = sys.argv[1], sys.argv[2]
    expected = []
    with open(raw_path, "rb") as f:
        while True:
            hdr = f.read(20)
            if len(hdr) < 20:
                break
            nlen, top, left, w, h = struct.unpack("<5i", hdr)
            name = f.read(nlen).decode()
            rgba = np.frombuffer(f.read(w * h * 4), dtype=np.uint8).reshape(h, w, 4)
            expected.append((name, top, left, rgba))

    psd = PSDImage.open(psd_path)
    layers = list(psd)
    assert len(layers) == len(expected), (len(layers), len(expected))
    for layer, (name, top, left, rgba) in zip(layers, expected):
        assert layer.name == name, (layer.name, name)
        assert (layer.top, layer.left) == (top, left), (layer.top, layer.left, top, left)
        img = np.round(layer.numpy() * 255).astype(np.uint8)
        assert img.shape == rgba.shape, (img.shape, rgba.shape)
        assert np.array_equal(img, rgba), f"pixel mismatch in {name}"
        print(f"layer '{name}' ok: {img.shape} at ({top},{left})")
    print("VALIDATION PASS")


if __name__ == "__main__":
    main()
