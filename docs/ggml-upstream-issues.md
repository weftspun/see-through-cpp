# ggml findings for upstream triage

Observed against vendored ggml v0.17.0 (9be3133) on Windows (clang/lld,
Ninja), CPU = AMD Ryzen 7 3800X, GPU = NVIDIA RTX 4090 (Vulkan backend).
Each item has a local repro in this repo.

## 1. gallocr recycles INPUT tensor buffers within one compute

`ggml_gallocr_free_node` (ggml-alloc.c) only spares tensors flagged
`GGML_TENSOR_FLAG_OUTPUT`; a `GGML_TENSOR_FLAG_INPUT` tensor's buffer is
freed after its last consumer *inside* the graph and reused for later
intermediates. Consequence: in a loop that reuses one allocated graph
(diffusion sampling), inputs set once before the loop are clobbered during
the first compute — the second compute silently reads garbage (in our case
corrupted cross-attention conditioning; step 0 matched the reference at
~1e-3, step 1+ diverged wildly).

Perhaps intended behavior — but worth either documenting loudly on
`ggml_gallocr_alloc_graph` or having FLAG_INPUT also pin the buffer.

Repro: `tests/test_graph_properties.cpp`, property "graph reuse with re-set
inputs matches fresh graph" (the fix — re-setting all inputs each compute —
is what makes it pass; setting them once reproduces the corruption).

## 2. CPU flash_attn_ext diverges from naive attention for n_head >= 2

With q/k/v in the documented layouts (q/k/v `[d, T, H, B]`, v not
transposed, mask null), CPU `ggml_flash_attn_ext` agrees with a naive
soft_max(QK^T)V graph to f16 precision for a single head, but diverges by
~0.3 max-abs for `n_head >= 2` (e.g. d=64, H=2, Tq=13, Tk=20, B=1). The
Vulkan backend agrees with naive for the same shapes/layouts.

Repro: `tests/test_graph_properties.cpp`, property "attn_tokens: flash ==
naive" — currently gated to GPU-only because of this divergence.

## 3. Vulkan ggml_conv_2d_direct silently returns zeros at large spatial sizes

`ggml_conv_2d_direct` (f32 kernel, k3 s1 p1) matches the im2col+mul_mat
path at 512x512 and below; a zero-output symptom was observed at ~1280x1280
during the original decode-stage debugging (which is why decode uses
row-chunked im2col, not direct conv, at pixel resolution). (The op also
differs numerically from im2col for the stride-2/pad-0 + asymmetric-
`ggml_pad` encoder pattern, magnitude ~3.6 — possibly out-of-contract usage,
but it fails silently rather than asserting.)

Note: re-verified via the Lean witness gate (`verify/KernelGate.lean`,
`st_witness_check_flat`) at the exact original shape (1280x1280, C=4, OC=4,
k3 s1 p1) with a fresh random witness — it did NOT reproduce (interval
~0.07, contained). The original repro (deleted `tests/test_graph_properties.
cpp`) never hard-asserted on this case either (informational printf only).
Treat as data/seed-dependent rather than deterministic; the row-chunked
decode path remains the mitigation regardless, so no action is blocked on
this.

## 4. Suspected: ggml_conv_2d_direct wrong at the UNet's 160x160 latent
   level, only with real trained weights, only at res=1280

The full 1280px pipeline collapses every per-tag output layer to a thin
crop (~7% of expected height) at `--steps` >= 8 (steps=4 stays clean, but
that's likely just heavy noise masking the effect, not a real pass).
Bisected by resolution (all multiples of 64, as required by trans-vae's
6-stage decoder — see the `validate_resolution`/`--res` rounding added in
`src/see_through.cpp`): 512, 896, 1088, 1152, 1216 (ZR = res/8 = 64 through
152) all decode cleanly; only res=1280 (ZR=160) — the very next multiple of
64 above the last clean point — collapses. This pins the defect to
something specific about *exactly* 1280, not a gradual size-dependent
drift.

The only resolution-scaling knobs in the UNet are `direct_conv`
(`ggml_conv_2d_direct`) and `flash_attn`, both always-on for GPU regardless
of resolution (`pipeline.cpp`'s `pipe_load`). `direct_conv` operates on the
UNet's three latent levels (ZR, ZR/2, ZR/4); only res=1280 reaches
ZR=160 among tested values. The existing Lean witness gate validates
`direct_conv` at exactly this shape (`unet-direct-160-320-*` cases) and
finds it contained — but only with *synthetic random Gaussian weights*,
never the real trained checkpoint. A real-weight-specific numerical edge
case (e.g. triggered by specific value ranges/magnitudes absent from
N(0,0.5) test data) at this one shape is the leading hypothesis.

Update: `direct_conv` is now **ruled out**. Extended row-chunked im2col to
pass batch through untouched (was hardcoded to 1) and routed it ahead of
`direct_conv` for the UNet's stride-1 convs (`direct_conv` stays only as
the fallback for the stride-2 downsamplers, which row-chunk doesn't
handle). New Lean witness cases (`unet-rowchunk-*-b13`, real batch=13,
all 3 latent levels) confirm this generalization is itself exact (interval
0.0) — so the row-chunk code is correct, not the source of any bug. But a
full 1280px run with this fix in place shows the *same* collapse
(`topwear` still ~854x70). Conclusion: `direct_conv` was never the cause;
the row-chunk swap is kept anyway (independently correct, removes a
latent-shape blind spot for the future) but did not fix this bug.

That leaves `flash_attn` as the only other resolution-scaling knob, and
it has the identical problem `direct_conv` did: disabling it to test in
isolation OOMs immediately (~21.3GB single allocation, matching the
"naive 80x80 spatial attention is ~21GB at 1280px" comment in
`pipeline.cpp`) — no VRAM headroom to run the naive-attention fallback at
production batch/token counts. The existing Lean flash witness cases
(`flash-t6400-*`, `flash-t1600-*`) already happen to match res=1280's real
attention token counts (80x80 and 40x40 levels) with synthetic weights and
pass cleanly, so if `flash_attn` is the cause, it's the same class of
"synthetic-weights-only-validated" blind spot as the `direct_conv`
hypothesis was — untestable without either a Lean witness case backed by
the *real* loaded checkpoint weights (not yet built), or another
VRAM-safe composed replacement for naive attention at this scale.

Status: paused. Both resolution-scaling UNet knobs have been checked as
thoroughly as a simple enable/disable toggle allows; neither confirms nor
excludes `flash_attn`, and `direct_conv` is excluded. Root cause of the
1280px collapse remains open.

## Remediation policy

For any ggml kernel defect that blocks production and has no reasonable
op-composition workaround, the fix is a custom GPU kernel written in Slang
(compiled to SPIR-V, dispatched through ggml custom ops on the Vulkan
backend). Escalation ladder:

1. Compose the computation from validated ggml ops (used for: row-chunked
   im2col replacing the defective large-size Vulkan direct conv).
2. If composition is impossible or too slow for the throughput target,
   write a Slang kernel for exactly the defective op, gated by the same
   property probes as the op it replaces.

Currently nothing qualifies for step 2: the gallocr input-recycling issue
is allocator behavior (kernel-independent), CPU flash divergence is outside
the production path (GPU-primary), and the Vulkan direct-conv defect is
covered by step 1.
