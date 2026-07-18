# see-through.cpp

A planned C++/[ggml](https://github.com/ggml-org/ggml) port of
[See-Through](https://github.com/weftspun/see-through) (Shitagaki Lab,
SIGGRAPH 2026, Apache-2.0): single anime illustration → up to 23 fully
inpainted semantic layers + pseudo-depth → layered PSD. Modeled on
[trellis2cpp](https://github.com/weftspun/trellis2cpp): GGUF conversion
scripts first, single-file library later, no PyTorch at runtime.

## Status: phase 1 — GGUF conversion

`convert_diffusers_to_gguf.py` converts any See-Through diffusers component
to GGUF v3 (same self-contained writer as trellis2cpp; numpy +
huggingface_hub only, manual safetensors parsing incl. BF16). Tensors keep
their diffusers state-dict names; the component `config.json` travels
verbatim as the `seethrough.<component>.config_json` KV.

```sh
python convert_diffusers_to_gguf.py --component layerdiff-unet   # SDXL CrossFrame UNet
python convert_diffusers_to_gguf.py --component trans-vae        # transparent VAE (UNet1024 head)
python convert_diffusers_to_gguf.py --component marigold-unet    # SD2 depth UNet
python convert_diffusers_to_gguf.py --component layerdiff-vae    # stock SDXL VAE
python convert_diffusers_to_gguf.py --component marigold-vae     # stock SD VAE
```

## Architecture notes (from reading the upstream source)

- **LayerDiff 3D UNet** (`layerdifforg/seethroughv0.0.2_layerdiff3d`,
  `unet/`): a diffusers SDXL `UNet2DConditionModel` where the attention
  down/mid/up blocks are replaced by `CrossFrameAttn*` variants
  (`common/modules/layerdiffuse/layerdiff3d.py`). Layers are batched as
  "frames" (`sample: [B, F, C, H, W]` → reshaped to `B*F`), and attention
  runs **across frames**, video-diffusion style — that is the whole
  multi-layer mechanism. The port can reuse stable-diffusion.cpp's SDXL
  graph with the attention reshaped over the frame axis.
- **TransparentVAE** (`trans_vae/`): LayerDiffuse-style — stock SDXL VAE
  latents, plus a `UNet1024(in=3[+3 rgb_cond], out=4)` decoder head that
  predicts RGBA from (pixels, latent), run as an 8-way flip/rot90 ensemble
  (`estimate_augmented`). Standard conv/norm/attn ops only.
- **Marigold depth** (`24yearsold/seethroughv0.0.1_marigold`): SD2-based
  MarigoldDepthPipeline with the same frame-condition UNet class, one CLIP
  text encoder, stock SD VAE.
- **Text encoders**: stock CLIP-L + OpenCLIP-G (SDXL) — stable-diffusion.cpp
  already implements both; convert with its own tooling when the graph work
  starts.
- **Annotators** (SAM/mmdet/timm taggers) are *not* in scope for the cpp
  port; the pipeline design gates on rf-detr-mcp instead (see
  easy-diffusion-traces `2026-07-17-inverse-render-loop/docs-DESIGN.md`).

## Phase plan

1. **Convert** all five diffusers components to GGUF (this repo, now).
2. **Validate numerics**: reference activations from the Python pipeline
   (docker/WSL2) vs ggml graphs, per component — trellis2cpp's
   `Dockerfile.ref` pattern.
3. **Graphs**: fork stable-diffusion.cpp's SDXL UNet + VAE, add the
   cross-frame attention reshape, UNet1024 alpha head, k-diffusion sampler;
   Marigold is a reduced variant of the same.
4. **PSD writer** + C ABI + server, trellis2cpp-style.
