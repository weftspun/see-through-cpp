#!/usr/bin/env python3
"""Convert the AnimeMangaInpainting LaMa checkpoint (lama_large_512px.ckpt,
FFCResNetGenerator 4->3, 18 FFC blocks, use_mpe=False) to lama.gguf.
Generator state-dict names kept verbatim; everything f32 (the model is
~200MB and BatchNorm folding happens at graph build)."""
import json
import os
import struct

import numpy as np
import torch

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGN = 32
VT_U32, VT_STR = 4, 8

URL = "https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/lama_large_512px.ckpt"


def gguf_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def main():
    sd = torch.hub.load_state_dict_from_url(URL, map_location="cpu")["gen_state_dict"]
    tensors = []
    for name in sorted(sd.keys()):
        if name.endswith("num_batches_tracked"):
            continue
        arr = sd[name].float().numpy()
        dims = list(reversed(arr.shape)) if arr.ndim else [1]
        tensors.append((name, dims, np.ascontiguousarray(arr, dtype="<f4").tobytes()))
    print(f"tensors: {len(tensors)}")

    metadata = [
        gguf_str("general.architecture") + struct.pack("<I", VT_STR) + gguf_str("seethrough-lama"),
        gguf_str("general.alignment") + struct.pack("<I", VT_U32) + struct.pack("<I", ALIGN),
    ]

    header = bytearray(GGUF_MAGIC)
    header += struct.pack("<I", GGUF_VERSION)
    header += struct.pack("<Q", len(tensors))
    header += struct.pack("<Q", len(metadata))
    for m in metadata:
        header += m

    infos = bytearray()
    offset = 0
    offsets = []
    for _n, _d, raw in tensors:
        offsets.append(offset)
        offset = align(offset + len(raw))
    for (name, dims, raw), off in zip(tensors, offsets):
        infos += gguf_str(name)
        infos += struct.pack("<I", len(dims))
        for d in dims:
            infos += struct.pack("<Q", int(d))
        infos += struct.pack("<I", 0)   # f32
        infos += struct.pack("<Q", off)

    pre = len(header) + len(infos)
    with open("lama.gguf", "wb") as f:
        f.write(header)
        f.write(infos)
        f.write(b"\x00" * (align(pre) - pre))
        for (_n, _d, raw), _o in zip(tensors, offsets):
            f.write(raw)
            pad = align(len(raw)) - len(raw)
            if pad:
                f.write(b"\x00" * pad)
    print("wrote lama.gguf", os.path.getsize("lama.gguf"), "bytes")


if __name__ == "__main__":
    main()
