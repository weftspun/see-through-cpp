#!/usr/bin/env python3
"""End-to-end upstream reference: call the hosted see-through-demo Space's
/inference API on a given image with our CLI-default settings, save the
returned PSD, and record the wall time. Independent ground truth for
comparing against a local ggml run of the same image/settings -- doesn't
require a local PyTorch install.

Usage: python tools/hf_space_infer.py <image> [out.psd]
Requires HF_TOKEN in the environment and gradio_client installed."""
import os, shutil, sys, time

from gradio_client import Client, handle_file

if len(sys.argv) < 2:
    print("usage: hf_space_infer.py <image> [out.psd]", file=sys.stderr)
    sys.exit(1)
img = sys.argv[1]
out_psd = sys.argv[2] if len(sys.argv) > 2 else "upstream_reference.psd"

t0 = time.time()
client = Client("24yearsold/see-through-demo", token=os.environ["HF_TOKEN"])
print(f"[hf] connected in {time.time()-t0:.1f}s", flush=True)

t1 = time.time()
result = client.predict(
    image=handle_file(img),
    resolution=1280,
    seed=42,
    tblr_split=True,
    api_name="/inference",
)
dt = time.time() - t1
print(f"[hf] inference wall time: {dt:.1f}s", flush=True)
print(f"[hf] result: {result}", flush=True)

# result may be a single path or a tuple/list of paths; find the .psd
paths = result if isinstance(result, (list, tuple)) else [result]
flat = []
for p in paths:
    if isinstance(p, dict):
        flat.append(p.get("value") or p.get("path") or "")
    else:
        flat.append(str(p))
for p in flat:
    if p.endswith(".psd"):
        shutil.copy(p, out_psd)
        print(f"[hf] saved {out_psd} ({os.path.getsize(out_psd)} bytes)", flush=True)
        break
else:
    print(f"[hf] no .psd in result; components: {flat}", flush=True)
