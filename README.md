# see-through.cpp

Turns a single anime character illustration into up to 23 separate,
fully-inpainted layers (hair, face, clothing, accessories, etc.) plus a
depth map, saved as a layered SVG. C++
port of [See-Through](https://github.com/shitagaki-lab/see-through)
(Shitagaki Lab, SIGGRAPH 2026).

Requires a GPU with Vulkan support (no CPU fallback).

## Get the weights — download and unpack into a `models/` folder:

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

## Build (Vulkan backend must be enabled explicitly — it's off by default in ggml):

```sh
cmake -B build -G Ninja -DGGML_VULKAN=ON && cmake --build build
```

## Run:

```sh
./build/see-through -m models -i in.png -o out.svg
```

Input is loaded via [stb_image](https://github.com/nothings/stb), so PNG, JPEG,
BMP, TGA, GIF, and PSD are supported — **not WebP**; convert first (e.g.
`ffmpeg -i in.webp in.png`).

Produces `out.svg` — one `<image>` element per layer, each with a
`data-tag`/`data-z`/`data-depth-median` attribute. Useful flags:

- `--steps 30` / `--res 768` / `--depth-res 768` — quality/speed knobs
  (`--res 768` is noticeably faster with little quality loss — a good
  default while iterating)
- `--png-dir <dir>` — also export each layer as a separate PNG
- `--seed 42` — for reproducible output
