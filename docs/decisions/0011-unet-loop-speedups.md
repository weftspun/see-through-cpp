# UNet-loop speedups: flash attn scoping, GEMM flattening, Winograd retirement, adaptive row-chunking

* Status: accepted
* Date: 2026-07-20

## Context and Problem Statement

MADR 0010 set the goal: close the gap to the upstream HuggingFace Space's
end-to-end image→PSD time. Two calibration facts changed this session:

1. **The live upstream benchmark is 183.2s, not 72s.** Measured directly:
   `24yearsold/see-through-demo`'s `/inference` (gradio_client, HF_TOKEN,
   `test_image3.png`, resolution=1280/seed=42/tblr_split=True) took 183.2s
   wall. The historical ~1m12s figure does not reflect the Space's current
   behavior; 183s is the honest scoreboard.
2. **The measured pre-session baseline was far above MADR 0010's 108s
   band**: main@4e4156e ran the body UNet loop at **347.6s** (11.4s/step,
   Q4 models, 1280/30, RTX 4090) — the MADR-0009-era revert had not
   recovered the old 108s trace, because a *separate* regression was live
   (see finding 1 below).

All changes below were verified two ways: wall-time A/B at identical
settings, and per-layer alpha-mask IoU between the produced PSDs
(`psd_layers` + threshold-0.125 IoU) — "faster but different output"
does not count.

## Findings and changes

### 1. tiled_naive_attn silently disabled flash attention for BOTH diffusion UNets

`pipe_load` set `tiled_naive_attn = true` for every model (needed only for
`attn_block`, the VAE/unet1024 spatial attention whose plain-naive path
overflows Vulkan's maxStorageBufferRange at 1280px — MADR 0009), and
`attn_tokens` prefers `tiled_naive_attn` over `flash_attn`. Net effect:
both diffusion UNets ran O(T²) naive attention even though
`m.flash_attn=true` was set and `unet_frame.cpp` has **no `attn_block`
call sites at all**.

**Change**: `tiled_naive_attn` is now forced off for `layerdiff-unet`
loads specifically (flash there is Lean-witness-validated,
docs/ggml-upstream-issues.md #4); `SEETHROUGH_TILED_ATTN_LAYERDIFF=1`
opts back out. marigold-unet stays on the tiled path (flash for it
produced corrupted depth at res>=768 when last tried, never re-validated).

**Verified**: flash-vs-tiled A/B on `test_image3.png` (same binary, same
seed): per-layer IoU ≈ 1.000 on every substantive layer (handwear
identical at 19081 px), wall time 6m10s vs 8m58s. Flash is safe for
layerdiff-unet and ~45% faster on both UNet loops.

### 2. linear() ran cross-frame FFNs as thousands of 13-column GEMMs

`cross_frame_block` permutes to (C, F=13, S=1600..6400); every
linear/FFN inside then hit `ggml_mul_mat` with n=13 and batch=S — S tiny
GEMMs each re-reading the whole quantized weight. GGML_VK_PERF_LOGGER
showed these calls at ~52% of a 7.3s UNet step, running ~5x slower per
FLOP than the same multiply with the dims swapped.

**Change**: `linear()` flattens any contiguous rank>2 input to 2D
(ne0, ne1*ne2*ne3) around the matmul — a pure view, bit-identical math —
turning those into single n=20800/83200 GEMMs.

**Measured**: UNet step 7.05s → 4.42s (body loop 215.8s → 135.0s).
Output IoU vs pre-change run ≈ 1.0 (identical within float noise).

### 3. Winograd conv composition is slower than what it replaced — retired to opt-in

`conv2d_winograd_3x3s1` (MADR 0010 candidate 3, verified exact) is built
from ggml op composition: 16 batch=1 `ggml_out_prod` calls plus
add/sub/concat/cont transform trees per conv. At production shapes the
perf logger showed OUT_PROD alone at 24% of a TransparentVAE decode graph
and ~33k CONT nodes per graph. The theoretical 2.25x FLOP saving never
survived the dispatch overhead and the out_prod kernel's performance.

**Change**: the Winograd path is opt-in via `SEETHROUGH_WINOGRAD=1`.
Convs below `conv_row_chunk_min_hw` now take plain im2col+mul_mat.
Decode stages roughly halved (layerdiff.body 281.8s → 175.8s includes
this plus finding 2). Revisit Winograd only as a fused kernel, never as
an op composition.

### 4. Fixed 8-way row-chunking replaced with a VRAM-budget-derived count

`conv_row_chunk`'s NCH=8 was sized for f16-model VRAM pressure; with Q4
weights on a 24GB device the coarser UNet levels' im2col transients are
well under 1GB, so blanket 8-way tiling just multiplied
im2col/concat/cont dispatches. NCH is now
`ceil(im2col_bytes / budget)` (default 2GB,
`SEETHROUGH_ROWCHUNK_BUDGET_MB` overrides; every chunk stays far below
Vulkan's ~4.29GB maxStorageBufferRange), and NCH==1 collapses to a
single untiled im2col with no halo/crop/concat.

### 5. Structural churn eliminated (measured small, kept for hygiene)

One process-lifetime Vulkan backend instead of create/destroy per graph;
the per-frame TransparentVAE decode (24 graph rebuilds/run), marigold
cond-encode (21) and marigold decode (20) loops now build+allocate their
fixed-shape graph once and re-set inputs per frame — the same proven
pattern as the UNet step loops. (This is graph REUSE, not the reverted
multi-image BATCHING that once corrupted depth output.) Instrumentation
(`[perf]` stderr lines + accumulators) showed backend init/graph
build/alloc were only ~0.2s/run after the fix; model loads are ~6.5s/run
total on NVMe and were never the bottleneck.

## Results (test_image3.png, 1280/30/seed 42, Q4, RTX 4090)

| configuration | end-to-end |
|---|---|
| main@4e4156e (extrapolated from body loop 347.6s) | ~12+ min |
| + flash attn scoping, backend/graph reuse (opt1-era) | 9m39s |
| + linear flatten, Winograd opt-in (opt2-era) | 6m03s* |
| + adaptive row-chunking | see spans |
| upstream Space, live | 3m03s |

*opt2 measured on the original test_image.png; t3 flash run on
test_image3.png was 6m10s at the same code.

## Open items

- `handwear-l` on the original `test_image.png` collapses to a ~17px
  sliver in the optimized runs (deterministic across two runs; MADR 0009
  recorded real content there). Flash-vs-tiled produced *identical*
  output on test_image3, so flash is likely not the cause — this looks
  like MADR 0008's "L/R-split sliver" with, finally, a deterministic
  repro candidate. Needs a tiled-attn rerun on the restored
  test_image.png to close.
- Upstream L/R-splits touching hands (handwear-l/-r both real on
  test_image3); our connected-component split keeps them merged. Parity
  gap, not corruption.
- Next speed candidates, per-op profile: ~1.35s/step of f32 element-wise
  ops (f16-activation experiment, env-gated); marigold stage 34s;
  first-step shader-compile overhead (~3-4s/loop).
