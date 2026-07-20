# see-through.cpp

Turns a single anime character illustration into up to 23 separate,
fully-inpainted layers (hair, face, clothing, accessories, etc.) plus a
depth map, saved as a layered PSD. C++/[ggml](https://github.com/ggml-org/ggml)
port of [See-Through](https://github.com/weftspun/see-through)
(Shitagaki Lab, SIGGRAPH 2026) — no PyTorch needed at runtime.

Requires a GPU with Vulkan support (no CPU fallback).

## Run it

**1. Get the weights** — download and unpack into a `models/` folder:

```sh
mkdir models && cd models
gh release download v0.0.2-dev --repo weftspun/see-through-cpp \
    -p "lama.gguf" -p "layerdiff-te1.gguf" -p "layerdiff-te2.gguf" \
    -p "layerdiff-vae.gguf" -p "marigold-te.gguf" -p "marigold-vae.gguf" \
    -p "trans-vae.gguf"
gh release download v0.0.2-dev --repo weftspun/see-through-cpp \
    -p "layerdiff-unet.gguf.zst.part*" -p "marigold-unet.gguf.zst.part*"
cat layerdiff-unet.gguf.zst.part* | zstd -d -o layerdiff-unet.gguf
cat marigold-unet.gguf.zst.part* | zstd -d -o marigold-unet.gguf
rm *.zst.part*
```

(Or grab a prebuilt release binary instead of building — see
[Releases](../../releases).)

**2. Build:**

```sh
cmake -B build -G Ninja && cmake --build build
```

**3. Run:**

```sh
./build/see-through -m models -i in.png -o out.psd
```

Produces `out.psd` (the layers), `out_depth.psd`, and `out.psd.json`
(bounding box + depth per layer). Useful flags:

- `--steps 30` / `--res 1280` / `--depth-res 768` — quality/speed knobs
- `--png-dir <dir>` — also export each layer as a separate PNG
- `--seed 42` — for reproducible output
