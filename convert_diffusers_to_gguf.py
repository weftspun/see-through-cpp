#!/usr/bin/env python3
"""
Convert a See-Through diffusers component (HF safetensors) to GGUF v3 for
see-through.cpp. Same self-contained writer as trellis2cpp's converters —
no `gguf` package, no torch; only numpy + huggingface_hub.

Components (from the two See-Through HF repos):

  layerdiff-unet   layerdifforg/seethroughv0.0.2_layerdiff3d  unet/
                   (SDXL UNet with CrossFrame attention blocks)
  layerdiff-vae    ...                                        vae/       (stock SDXL VAE)
  trans-vae        ...                                        trans_vae/ (LayerDiffuse transparent VAE, UNet1024 head)
  marigold-unet    24yearsold/seethroughv0.0.1_marigold       unet/
                   (SD2 UNet, frame-condition variant, depth)
  marigold-vae     ...                                        vae/       (stock SD VAE)

Tensors are keyed by their original diffusers state-dict names; the full
component config.json travels as a single string KV
(`seethrough.<component>.config_json`) so the C++ loader owns interpretation.

Usage:
    python convert_diffusers_to_gguf.py --component layerdiff-unet [--ftype 1]
    python convert_diffusers_to_gguf.py --repo <id> --subfolder unet --arch my-arch
ftype: 0 = f32, 1 = f16 (default; 2D+ weights only, norms/biases stay f32)
"""

import argparse
import json
import os
import struct

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGUF_VT_UINT32 = 4
GGUF_VT_STRING = 8

COMPONENTS = {
    "layerdiff-unet": ("layerdifforg/seethroughv0.0.2_layerdiff3d", "unet"),
    "layerdiff-vae": ("layerdifforg/seethroughv0.0.2_layerdiff3d", "vae"),
    "trans-vae": ("layerdifforg/seethroughv0.0.2_layerdiff3d", "trans_vae"),
    "marigold-unet": ("24yearsold/seethroughv0.0.1_marigold", "unet"),
    "marigold-vae": ("24yearsold/seethroughv0.0.1_marigold", "vae"),
}


def gguf_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_u32(key, v):
    return gguf_str(key) + struct.pack("<I", GGUF_VT_UINT32) + struct.pack("<I", int(v))


def kv_str(key, v):
    return gguf_str(key) + struct.pack("<I", GGUF_VT_STRING) + gguf_str(str(v))


def align(n, a=GGUF_ALIGNMENT):
    return (n + a - 1) // a * a


def load_safetensors(path):
    """Minimal safetensors reader -> {name: np.ndarray(f32)}. Handles F32/F16/BF16."""
    out = {}
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype, shape = info["dtype"], info["shape"]
            beg, end = info["data_offsets"]
            f.seek(data_start + beg)
            raw = f.read(end - beg)
            if dtype == "F32":
                arr = np.frombuffer(raw, dtype="<f4")
            elif dtype == "F16":
                arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
            elif dtype == "BF16":
                u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
                arr = u16.view(np.float32)
            else:
                raise ValueError(f"{name}: unsupported dtype {dtype}")
            out[name] = arr.reshape(shape) if shape else arr
    return out


def choose_type(name, shape, ftype):
    if ftype == 0:
        return GGML_TYPE_F32
    if len(shape) >= 2 and "norm" not in name:
        return GGML_TYPE_F16
    return GGML_TYPE_F32


def to_bytes(arr, gtype):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if gtype == GGML_TYPE_F32:
        return arr.astype("<f4", copy=False).tobytes()
    return arr.astype("<f2").tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--component", choices=sorted(COMPONENTS), default=None)
    ap.add_argument("--repo", default=None, help="HF repo id (overrides --component)")
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--arch", default=None, help="general.architecture value")
    ap.add_argument("--output", default=None)
    ap.add_argument("--ftype", type=int, default=1, choices=[0, 1])
    args = ap.parse_args()

    if args.repo:
        repo, sub = args.repo, args.subfolder or "unet"
        comp = args.arch or f"{os.path.basename(repo)}-{sub}"
    else:
        comp = args.component or "layerdiff-unet"
        repo, sub = COMPONENTS[comp]

    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(repo, f"{sub}/diffusion_pytorch_model.safetensors")
    cfg_path = hf_hub_download(repo, f"{sub}/config.json")
    out_path = args.output or f"{comp}.gguf"
    print(f"model : {model_path}\noutput: {out_path}  (ftype={args.ftype})")

    with open(cfg_path) as f:
        cfg_json = f.read()
    cfg = json.loads(cfg_json)
    print(f"class : {cfg.get('_class_name')}")

    kv_prefix = f"seethrough.{comp}."
    metadata = [
        kv_str("general.architecture", f"seethrough-{comp}"),
        kv_str("general.name", f"{repo}/{sub}"),
        kv_u32("general.file_type", args.ftype),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_str(kv_prefix + "config_json", cfg_json),
    ]

    print("loading state_dict...")
    sd = load_safetensors(model_path)
    tensors = []
    n_f16 = 0
    for name in sorted(sd.keys()):
        arr = sd[name]
        gtype = choose_type(name, arr.shape, args.ftype)
        n_f16 += gtype == GGML_TYPE_F16
        dims = list(reversed(arr.shape)) if arr.ndim else [1]  # ggml ne[] order
        tensors.append((name, gtype, dims, to_bytes(arr, gtype)))
    print(f"tensors: {len(tensors)}  (f16={n_f16}, f32={len(tensors) - n_f16})")

    header = bytearray()
    header += GGUF_MAGIC
    header += struct.pack("<I", GGUF_VERSION)
    header += struct.pack("<Q", len(tensors))
    header += struct.pack("<Q", len(metadata))
    for m in metadata:
        header += m

    infos = bytearray()
    offset = 0
    offsets = []
    for _name, _gtype, _dims, raw in tensors:
        offsets.append(offset)
        offset = align(offset + len(raw))
    for (name, gtype, dims, raw), off in zip(tensors, offsets):
        infos += gguf_str(name)
        infos += struct.pack("<I", len(dims))
        for d in dims:
            infos += struct.pack("<Q", int(d))
        infos += struct.pack("<I", gtype)
        infos += struct.pack("<Q", off)

    pre = len(header) + len(infos)
    with open(out_path, "wb") as fout:
        fout.write(header)
        fout.write(infos)
        fout.write(b"\x00" * (align(pre) - pre))
        for (_name, _gtype, _dims, raw), _off in zip(tensors, offsets):
            fout.write(raw)
            pad = align(len(raw)) - len(raw)
            if pad:
                fout.write(b"\x00" * pad)
    print(f"wrote {out_path}  ({os.path.getsize(out_path):,} bytes)")


if __name__ == "__main__":
    main()
