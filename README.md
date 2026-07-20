# see-through.cpp

A C++/[ggml](https://github.com/ggml-org/ggml) port of
[See-Through](https://github.com/weftspun/see-through) (Shitagaki Lab,
SIGGRAPH 2026, Apache-2.0): single anime illustration → up to 23 fully
inpainted semantic layers + pseudo-depth → layered PSD. Modeled on
[trellis2cpp](https://github.com/weftspun/trellis2cpp): GGUF conversion
first, per-milestone validated ggml graphs, no PyTorch at runtime.

## Status

All numeric milestones are validated against the upstream PyTorch pipeline
(pattern: `gen_reference_*.py` via `uv run` CPU torch → `src/test_*.cpp`
max-abs-diff gate, threshold 5e-2 unless noted):

| Milestone | Test | Result |
|---|---|---|
| UNet1024 alpha head | `test_trans_vae` | 6.5e-4 |
| AutoencoderKL decoder (both VAEs) | `test_sd_vae` | 1.7e-5 / 2.6e-3 |
| Full TransparentVAE decode chain | `test_trans_vae_full` | 1.4e-3 |
| AutoencoderKL encoder (both VAEs) | `test_vae_encode` | 1.4e-3 / 3.0e-2 |
| CLIP text encoders ×3 (id-exact tokenizer) | `test_clip` | 2.0e-2 / 3.4e-2+5.4e-4 / 3.6e-3 |
| UNetFrameCondition blocks (resnet+temb, Transformer3D incl. cross-frame, embeddings) | `test_unet_blocks` | ≤2.7e-3 |
| Full LayerDiff UNet forward (13 frames, group 0) | `test_unet_forward` | 1.5e-3 |
| DPM++ 2M SDE scheduler (30- and 2-step trajectories) | `test_scheduler` | 3.8e-5 (gate 1e-4) |
| Marigold DDIM (v-pred, zero-SNR, trailing) | `test_ddim` | 7.2e-7 (gate 1e-4) |
| apply_layerdiff E2E (512px / 2 steps) | `test_layerdiff_e2e` | 5.9e-3 |
| Marigold depth E2E (256px / 4 steps) | `test_marigold_e2e` | ≤6.1e-3, depth 2.1e-3 |
| LaMa FFC inpainting (custom 2-D FFT op) | `test_lama` | 1e-6 |
| PSD writer | `test_psd` + `check_psd.py` | byte-exact round-trip |

The `see-through` CLI runs the full pipeline on GPU (Vulkan):

```sh
cmake -B build -G Ninja && cmake --build build
./build/see-through -m <model-dir> -i in.png -o out.psd \
    [--seed 42] [--steps 30] [--res 1280] [--depth-res 768] [--threads 8]
```

It produces `out.psd` (depth-ordered RGBA layers), `out_depth.psd` and
`out.psd.json` (xyxy + depth_median per part), compatible with upstream's
`psd2partdicts` reader. Weights (`*.gguf`) live on the
[v0.0.1-dev release](../../releases/tag/v0.0.1-dev); files >2GB are
zstd-compressed split parts (`cat parts | zstd -d -o file.gguf`).

GPU (Vulkan) is required — the CLI and tests pick the first registry GPU
automatically, and error out if none is found. `--device cpu` is not
supported; there is no CPU fallback. Run Vulkan binaries with
`build-vulkan/bin` on PATH (shared ggml). `--png-dir <dir>` additionally
exports per-layer RGBA + depth PNGs in z order with a `layers.json`
manifest; `--debug-dir <dir>` dumps stage stats.

## Development

Braces are required around every `if`/`else`/`for`/`while`/`do` body
(`.clang-tidy`: `readability-braces-around-statements`). Enforced on new
lines only via a pre-commit hook — install it once per clone:

```sh
cp scripts/git-hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

Needs `clang-tidy` on PATH and a `compile_commands.json` in `build-vulkan/`
or `build/` (CMake generates one by default); the hook no-ops with a
warning if either is missing, rather than blocking commits.

## Port notes (documented divergences)

- Upstream `TransparentVAEDecoder.estimate_augmented` has a `break` after the
  first flip/rot combination — the 8-way ensemble is a single pass.
- VAE encodes use the posterior **mean**, not `.sample()` (posterior std is
  ~1e-5; upstream's page-latent path uses unseeded `.sample()`).
- Diffusion noise comes from the port's own RNG (statistically equivalent,
  not bit-equal to torch's); the E2E tests inject recorded upstream draws.
- The hair front/back KMeans uses an exact 1-D 2-means instead of sklearn's
  iterative KMeans (deterministic; only reachable for v2 tag models — v3
  models emit `front hair`/`back hair` directly).
- `ggml_conv_2d` unconditionally rounds activations to f16 in im2col; the
  port builds convs with an im2col matching the weight dtype (the SDXL VAE
  encoder's activations overflow f16 precision). `layerdiff-vae.gguf` and
  `layerdiff-te1.gguf` are f32 for the same reason.
- ggml's gallocr recycles **input** tensor buffers after their last in-graph
  read — every input is re-set before every compute in multi-step loops.
- Low-VRAM knobs at 1280px/24GB: flash attention + per-frame spatial
  transformer chunking + f16 skips + `ggml_conv_2d_direct` for the UNets;
  the decode uses exact im2col tiled over output rows instead (direct conv
  is wrong for the VAE-encoder s2/p0 path and silently zero on Vulkan at
  >= ~1280^2 — both tracked for upstream, see docs/ggml-upstream-issues.md).
- Dependencies are vendored as squashed git subtrees (ggml, psd_sdk,
  rapidcheck): update via `git subtree pull --prefix <dir> <url> <ref> --squash`.
- `third_party/{picoquic,picotls,mbedtls}` (HTTP/3 server, `SEETHROUGH_BUILD_H3_SERVER`)
  are vendored (plain copy, not subtree-tracked) from
  `github.com/fire/webtransportd`, `thirdparty/{picoquic,picotls,mbedtls}`,
  commit `b53faa2d1b94b2d3e3ee7f1591c82d0a7ea2952e`. Licenses: picoquic MIT,
  picotls MIT, mbedtls Apache-2.0. `third_party/cpp-httplib` (the plain
  HTTP/1.1 `see-through-server`) is MIT, from `yhirose/cpp-httplib`.

## GGUF conversion

`convert_diffusers_to_gguf.py` converts any See-Through diffusers component
(tensors keep their state-dict names; component `config.json` — and for text
encoders the tokenizer `vocab.json`/`merges.txt` — travel as KVs):

```sh
python convert_diffusers_to_gguf.py --component layerdiff-unet   # SDXL CrossFrame UNet (f16)
python convert_diffusers_to_gguf.py --component trans-vae        # transparent VAE (UNet1024 head)
python convert_diffusers_to_gguf.py --component marigold-unet    # SD2 depth UNet
python convert_diffusers_to_gguf.py --component layerdiff-vae --ftype 0   # SDXL VAE (f32, see notes)
python convert_diffusers_to_gguf.py --component marigold-vae     # stock SD VAE
python convert_diffusers_to_gguf.py --component layerdiff-te1 --ftype 0   # CLIP-L (f32)
python convert_diffusers_to_gguf.py --component layerdiff-te2    # OpenCLIP-G
python convert_diffusers_to_gguf.py --component marigold-te      # OpenCLIP-H
python convert_lama_to_gguf.py                                   # LaMa inpainter
```
