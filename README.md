# see-through.cpp

Turns a single anime character illustration into up to 23 separate,
fully-inpainted layers (hair, face, clothing, accessories, etc.) plus a
depth map, saved as a layered PSD. C++
port of [See-Through](https://github.com/shitagaki-lab/see-through)
(Shitagaki Lab, SIGGRAPH 2026).

Requires an NVIDIA GPU with CUDA support (no CPU fallback).

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

(Or grab a prebuilt release binary instead of building)

## Build (CUDA backend must be enabled explicitly — it's off by default in ggml):

```sh
cmake -B build -G Ninja -DGGML_CUDA=ON && cmake --build build
```

Requires the NVIDIA CUDA Toolkit installed, with `nvcc` able to find a
compatible host compiler (on Windows: MSVC's `cl.exe`; nvcc pins a maximum
supported MSVC version per release, which can lag behind a freshly-installed
Visual Studio — if configure fails with `cl.exe` version errors, install an
older MSVC toolset side by side, e.g. via the Visual Studio Installer's
"Desktop development with C++" component picker, and pass
`-DCMAKE_CUDA_HOST_COMPILER=<path to that cl.exe>`). If CMake can't
auto-detect your GPU's compute capability, pin it explicitly, e.g.
`-DCMAKE_CUDA_ARCHITECTURES=89` for Ada Lovelace (RTX 40-series).

## Run:

```sh
./build/see-through -m models -i in.png -o out.psd
```

Input is loaded via [stb_image](https://github.com/nothings/stb), so PNG, JPEG,
BMP, TGA, GIF, and PSD are supported — **not WebP**; convert first (e.g.
`ffmpeg -i in.webp in.png`).

`-o` must end in `.psd`. Produces a flat, layered `out.psd` (plus an
`out_depth.psd` companion and an `out.psd.json` metadata sidecar), matching
upstream's `dump_parts_psd`.

Useful flags:

- `--steps 30` / `--res 768` / `--depth-res 768` — quality/speed knobs
  (`768` is the default for both; bump `--res` up to `1280` for sharper
  output at a noticeably higher time cost)
- `--png-dir <dir>` — also export each layer as a separate PNG
- `--seed 42` — for reproducible output
- `--no-split-depth` / `--no-split-lr` — disable the depth-cluster
  front/back split (default target: `hair`) or the left/right
  connected-component split (default targets: `handwear`, `eyewhite`,
  `irides`, `eyelash`, `eyebrow`, `ears`); both are on by default
- `--split-depth-tags tag1,tag2,...` / `--split-lr-tags tag1,tag2,...` —
  override which tags each split applies to (replaces the default list,
  doesn't append to it)
