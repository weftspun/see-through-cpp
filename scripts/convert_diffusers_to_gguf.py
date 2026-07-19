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
       2 = q4_0 (nn.Linear weights only, matching upstream's own NF4 scope;
           conv kernels/norms/biases stay f16/f32 -- see quantize_q4_0())
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
GGML_TYPE_Q4_0 = 2
GGUF_VT_UINT32 = 4
GGUF_VT_STRING = 8

QK4_0 = 32  # ggml block_q4_0: one f16 scale + 32 packed 4-bit values (16 bytes)


def quantize_q4_0(arr):
    """arr: 2D float32 (out_features, in_features), in_features % QK4_0 == 0.
    Matches ggml's quantize_row_q4_0_ref bit-for-bit (row-major blocks never
    cross a row boundary since in_features % QK4_0 == 0)."""
    x = arr.reshape(-1, QK4_0).astype(np.float64)  # widen: avoid fp32 tie/round drift
    amax_idx = np.argmax(np.abs(x), axis=1)
    max_val = x[np.arange(x.shape[0]), amax_idx]
    d = max_val / -8.0
    with np.errstate(divide="ignore", invalid="ignore"):
        id_ = np.where(d != 0, 1.0 / d, 0.0)
    xq = (x * id_[:, None]).astype(np.float32)
    x0, x1 = xq[:, 0:16], xq[:, 16:32]
    xi0 = np.minimum(15, (x0 + 8.5).astype(np.int8)).astype(np.uint8)
    xi1 = np.minimum(15, (x1 + 8.5).astype(np.int8)).astype(np.uint8)
    qs = (xi0 | (xi1 << 4)).astype(np.uint8)
    d16 = d.astype(np.float32).astype("<f2").tobytes()
    out = bytearray()
    d16_view = np.frombuffer(d16, dtype="<f2")
    for i in range(x.shape[0]):
        out += d16_view[i].tobytes()
        out += qs[i].tobytes()
    return bytes(out)

COMPONENTS = {
    "layerdiff-unet": ("layerdifforg/seethroughv0.0.2_layerdiff3d", "unet"),
    "layerdiff-vae": ("layerdifforg/seethroughv0.0.2_layerdiff3d", "vae"),
    "trans-vae": ("layerdifforg/seethroughv0.0.2_layerdiff3d", "trans_vae"),
    "marigold-unet": ("24yearsold/seethroughv0.0.1_marigold", "unet"),
    "marigold-vae": ("24yearsold/seethroughv0.0.1_marigold", "vae"),
    # text encoders (transformers CLIPTextModel*, model.safetensors); the value
    # third element names the paired tokenizer subfolder whose vocab/merges are
    # embedded as KV strings so the C++ tokenizer is self-contained
    "layerdiff-te1": ("layerdifforg/seethroughv0.0.2_layerdiff3d", "text_encoder", "tokenizer"),
    "layerdiff-te2": ("layerdifforg/seethroughv0.0.2_layerdiff3d", "text_encoder_2", "tokenizer_2"),
    "marigold-te": ("24yearsold/seethroughv0.0.1_marigold", "text_encoder", "tokenizer"),
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
    if ftype == 2:
        # nn.Linear weights only (2D, in_features % QK4_0 == 0) — matches
        # upstream's own NF4 scope (bitsandbytes Linear4bit; conv kernels are
        # untouched there too, and their innermost dim (kw=3) can't form
        # 32-element blocks anyway)
        if len(shape) == 2 and "norm" not in name and shape[-1] % QK4_0 == 0:
            return GGML_TYPE_Q4_0
        if len(shape) >= 2 and "norm" not in name:
            return GGML_TYPE_F16
        return GGML_TYPE_F32
    if len(shape) >= 2 and "norm" not in name:
        return GGML_TYPE_F16
    return GGML_TYPE_F32


def to_bytes(arr, gtype):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if gtype == GGML_TYPE_F32:
        return arr.astype("<f4", copy=False).tobytes()
    if gtype == GGML_TYPE_Q4_0:
        return quantize_q4_0(arr)
    return arr.astype("<f2").tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--component", choices=sorted(COMPONENTS), default=None)
    ap.add_argument("--repo", default=None, help="HF repo id (overrides --component)")
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--arch", default=None, help="general.architecture value")
    ap.add_argument("--output", default=None)
    ap.add_argument("--ftype", type=int, default=1, choices=[0, 1, 2],
                     help="0=f32, 1=f16 (default), 2=q4_0 (Linear weights only, "
                          "rest stays f16/f32)")
    args = ap.parse_args()

    tok_sub = None
    if args.repo:
        repo, sub = args.repo, args.subfolder or "unet"
        comp = args.arch or f"{os.path.basename(repo)}-{sub}"
    else:
        comp = args.component or "layerdiff-unet"
        entry = COMPONENTS[comp]
        repo, sub = entry[0], entry[1]
        tok_sub = entry[2] if len(entry) > 2 else None

    from huggingface_hub import hf_hub_download
    fname = "model.safetensors" if tok_sub else "diffusion_pytorch_model.safetensors"
    model_path = hf_hub_download(repo, f"{sub}/{fname}")
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
    if tok_sub:
        with open(hf_hub_download(repo, f"{tok_sub}/vocab.json"), encoding="utf-8") as f:
            metadata.append(kv_str(kv_prefix + "vocab_json", f.read()))
        with open(hf_hub_download(repo, f"{tok_sub}/merges.txt"), encoding="utf-8") as f:
            metadata.append(kv_str(kv_prefix + "merges_txt", f.read()))

    print("loading state_dict...")
    sd = load_safetensors(model_path)
    tensors = []
    n_f16 = n_q4 = 0
    for name in sorted(sd.keys()):
        arr = sd[name]
        gtype = choose_type(name, arr.shape, args.ftype)
        n_f16 += gtype == GGML_TYPE_F16
        n_q4 += gtype == GGML_TYPE_Q4_0
        dims = list(reversed(arr.shape)) if arr.ndim else [1]  # ggml ne[] order
        tensors.append((name, gtype, dims, to_bytes(arr, gtype)))
    n_f32 = len(tensors) - n_f16 - n_q4
    print(f"tensors: {len(tensors)}  (q4_0={n_q4}, f16={n_f16}, f32={n_f32})")

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
