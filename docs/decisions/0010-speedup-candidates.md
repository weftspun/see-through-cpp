# Speedup candidates after the quantize_y precision fixes

* Status: proposed
* Date: 2026-07-20

## Context and Problem Statement

The original motivating goal for the Winograd/`out_prod` investigation
(MADR 0004/0005 era) was closing the gap to a ~1m12s upstream HuggingFace
Spaces benchmark; our port ran ~12-14 minutes end to end at the time.

Mid-investigation, a real correctness bug was found and fixed: Vulkan's
`ggml_mul_mat` silently quantizes the activation operand to int8 (`quantize_y`)
whenever the device supports `VK_KHR_shader_integer_dot_product` and the
operand's `(ne0*ne1)%4==0` — true for nearly every conv/linear/attention
shape in this codebase. Fixed by routing through `ggml_out_prod` (no such
fast path) for the plain-F32 case, and — for quantized/F16 weights, where
`out_prod` can't be used at all — an explicit F16 error-feedback residual
pass (`mul_mat_f32_residual` in `ops.cpp`) to recover the precision
`quantize_y` would otherwise have silently taken from the activation side.

`profiling/spans.jsonl` (real per-run trace data, not a micro-benchmark)
shows what these fixes actually cost end to end, tracked by
`layerdiff.body`'s span duration (the main UNet's 30-step diffusion loop,
one trace per full pipeline run, chronological):

| trace (first 8 chars) | layerdiff.body duration |
|---|---|
| `5be534f2` (pre-`out_prod`, original baseline) | **108.2s** |
| `3e4f0541` (`out_prod` conversion landed)     | 484.3s |
| `8b345f32`                                     | 364.3s |
| `3e4342ab`                                     | 428.9s |
| `9cc3e20e`                                     | 374.1s |
| `bb372701`                                     | 360.7s |
| `6068acbe`                                     | 349.8s |
| `a1b35f57`                                     | 343.8s |
| `1ee0cbb8` (+ residual-correction pass)        | **749.2s** |

The `out_prod` conversion alone cost **~3.2-4.5x**. Adding the residual pass
on top of that cost roughly another **~2x** (749s vs the ~350-480s band).
Combined, this session's precision fixes made the single most expensive
stage in the whole pipeline **~7x slower** than the pre-fix baseline — moving
*away* from the original 1m12s target, not toward it.

Meanwhile, MADR 0009 already validated that the **pre-fix** behavior (plain
`ggml_mul_mat`, `quantize_y`'s int8 activation quantization active,
Q4_0-quantized weights) produces a full 1280px/30-step run with 29 correctly
cropped, non-blank, visually coherent layers and a 0.69 mean IoU against
real upstream output — i.e., whatever precision `quantize_y` was costing
was never shown to be visible in the actual deliverable. The precision bug
is real and the fix is real, but its cost may be far out of proportion to
any benefit actually realized in final output quality.

## Candidates, ranked by estimated gain

### 1. Scope the precision fix down to where it's proven to matter (est. ~5-7x on the UNet loop)

Revert `mul_mat_op` (used by every `linear()`/`attn_block()` call) and
`conv_mm`'s general path back to plain `ggml_mul_mat` +
`ggml_mul_mat_set_prec(GGML_PREC_F32)` — the exact fix already validated in
MADR 0009 as sufficient for a coherent, high-IoU production run — for the
bulk of the UNet forward pass. Reserve `out_prod`/the residual-correction
pass (this session's much more expensive fix) only for op sites with a
**demonstrated** precision-sensitivity, analogous to the specific
near-cancelling VAE `mid_block.resnets.1` resnet identified in MADR 0009,
not blanket across every attention/linear/conv call in a 30-step loop.
This is the single biggest lever: it directly targets the ~7x regression
just measured, and only reintroduces precision risk in the one place (large
untiled reductions with real cancellation) shown to actually matter.

**Try this first.** Needs: identify which specific op sites (if any) beyond
the VAE encoder's `mid_block.resnets.1` actually need it — likely none do,
given MADR 0009's IoU numbers already look good without any of this
session's `out_prod`/residual work.

### 2. If (1) reveals real per-site precision needs, keep them scoped per-call, not per-type

Should some other specific layer turn out to need protection (unlikely per
above, but not yet re-verified after reverting), keep the guard keyed to
that exact `pre` name/call site rather than "every quantized-or-F16-weight
matmul in the whole graph" as it is now. Narrower gating keeps most of the
network on the fast path while still protecting the one fragile spot.

### 3. Wire Winograd into `conv_row_chunk`/`direct_conv` (est. up to ~2x on conv-heavy stages)

`conv2d_winograd_3x3s1` (this session's other deliverable, verified exact
against CPU ground truth) is only wired into `conv2d()`'s general im2col
fallback — the two paths that actually dominate cost at production scale
(`conv_row_chunk` for the 1280px VAE, `direct_conv` for the main 160x160
UNet) both bypass it entirely, per the comment already in `ops.cpp`. A
row-chunked Winograd variant (extending the existing row-tiling loop) or a
Winograd-based replacement for `ggml_conv_2d_direct` would give a genuine
~2.25x FLOP reduction (F(2x2,3x3) trades 9 multiplies/output for 4) on the
convolutions that actually cost the most, independent of the quantize_y
question. Lower estimated ceiling than (1), but additive with it and has no
correctness risk (already proven exact).

### 4. Investigate the `ggml_gallocr_needs_realloc`/reallocating-buffers churn (magnitude unconfirmed)

Flagged in the original deep-research pass and never resolved: this message
fires constantly across the tiled VAE encode/decode (once per tile) and CLIP
passes — each is a separate `ggml_gallocr_new`/`ggml_gallocr_alloc_graph`
call, so some repeated allocation overhead here is inherent to the
one-graph-per-tile design, not obviously a bug. The main 30-step UNet loop
itself builds and allocates its graph **once** before the loop (confirmed
in `pipeline.cpp`), so the messages seen during that stage most likely come
from something else sharing the log stream, not per-step reallocation — this
needs to be actually confirmed on a trace, not assumed either way, before
spending time on a fix.

### 5. Re-enable `flash_attn`/`coopmat2` per-model instead of blanket-disabled (magnitude unconfirmed)

`flash_attn` was reverted for **both** UNets after it corrupted
marigold-unet's output at res>=768 (MADR history) even though it was
independently validated safe for layerdiff-unet's shapes. Device reports
`NV_coopmat2` support. Gating flash_attn back on for layerdiff-unet
specifically (distinguishing the two UNets, which the current single
`unet` bool flag can't do) could recover some attention-stage cost without
reintroducing the marigold-unet corruption — but needs its own validation
first, exactly as flagged (and never done) when it was reverted.

## Consequences

Recommend trying (1) first and re-measuring `layerdiff.body` via
`profiling/spans.jsonl` before touching anything else — it's the only
candidate here with hard evidence of a large, currently-live regression
behind it, and reverting scope is far lower-risk than adding anything new.
