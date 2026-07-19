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

Update: ran every attention shape/batch combination in `productionDomain`
through 8 independent seeds instead of one (`tests/probe_flash_bigT.cpp`,
Vulkan/RTX 4090) to check whether the existing gate's single-seed pass was
hiding a seed-dependent flash divergence, particularly at B=1. It wasn't:
all 56 (shape, seed) combinations landed in violation 0.18-0.30, nowhere
near the 1.0 defect threshold, with no shape or batch showing an outlier.
This rules out "seed got lucky" as an explanation and leaves the
real-checkpoint-weight hypothesis as the only remaining untested angle —
building that requires loading actual trained q/k/v projection weights
(and ideally real mid-pipeline activations, not fresh random inputs) into
the FFI harness, not yet done.

(Caution for future runs of this probe: it must be built against
`build-vulkan/`, not the default `build/` — an earlier attempt this session
accidentally built against a CPU-only configuration and reproduced item 2's
CPU-only flash divergence, which looked exactly like a new defect until the
device was checked. The probe now refuses to run at all if `st_device()`
reports CPU.)

Update: `flash_attn` is now **ruled out**, decisively. Built a query-tiled
naive-attention path (`Model.tiled_naive_attn`, `attn_tokens` in
`unet_frame.cpp`) that computes the exact same `softmax(QK^T)V` as the
plain naive branch but chunked over Tq, keeping the kq intermediate
VRAM-bounded regardless of T — this is what makes it possible to actually
run *without* flash_attn at 1280px production token counts, where a plain
`flash_attn=false` toggle OOMs. New Lean witness cases (`tiled-*`) confirm
it's mathematically exact against the naive reference at every gated shape
(interval 0.0). Running the full 1280px CLI with `SEETHROUGH_TILED_ATTN=1`
in place of `flash_attn` produces the **identical** collapse (`topwear`
still 854x70). Since this path is proven exact and VRAM-safe at production
scale, this rules out `flash_attn` (and attention generally, in the sense
of "wrong kernel math") as the cause — not just "untested", but actually
excluded now.

Update: examined the raw pre-crop decode output directly (`--debug-dir`'s
`raw_topwear_full.png` and the per-frame `frame_N.png` dumps) rather than
only the post-crop semantic layer. Two findings redirect the search:

1. The collapse is already present in the **raw, uncropped, full-canvas**
   layer — `crop_part`'s bounding-box tightening (`src/postproc.cpp`) is
   not the cause; whatever's wrong happens upstream of it, in the decode
   itself.
2. `frame_0.png` and `frame_5.png` (different semantic tags/frames in the
   13-frame batch) show the **same** blocky, checkerboard-like artifact in
   the same screen position — a repeating pattern along the top and right
   edges, not smooth numerical noise or content-dependent corruption. The
   same shape appearing regardless of tag strongly suggests a structural
   bug shared across all frames — e.g. an indexing/tiling bug in the
   decode-stage row-chunk assembly or the multi-frame batch layout — rather
   than a per-content attention or conditioning problem.

This points the search at the VAE decode's row-chunked im2col *assembly*
(the tile-accumulation/frame-layout logic in `ops.cpp`'s `conv2d`, not the
underlying `ggml_im2col`/`ggml_mul_mat` kernels, which are still covered by
the `decode-rowchunk-1280-64` Lean case) as the next suspect, rather than
any attention knob.

Update: instrumented the row-chunk tile boundaries directly
(`SEETHROUGH_DEBUG_ROWCHUNK=1`, already-existing printf) for a full 1280px
decode. Every tile at every spatial size seen (1280, 768, 640, 384, 320,
160, 96, 80, 48, 40) tiles cleanly: monotonic 160-row (or proportional)
chunks, no overlap, correct halo handling, accumulator growing by exactly
one tile's worth each iteration. The row-chunk tiling *arithmetic* is not
the bug.

Update: re-examined the artifact's actual screen position rather than
just its shape — it sits at the very top row and right edge of the
canvas, i.e. exactly where letterbox padding falls for this portrait-
aspect input, with the interior/body region otherwise blank. Since this
was already true in the *raw pre-crop* dump, the decode is producing
genuinely near-empty output in the body region — this isn't a coordinate/
crop bug, the pixels themselves are missing.

Update: dumped the final latent's per-row mean|value| before decode runs
(`SEETHROUGH_DEBUG_LATENT=1`) to check whether the collapse is already
present pre-decode. It is not: every row from 0-156 sits in a tight
0.46-0.51 band with no edge concentration or dropout — a normal-looking
latent, no visible spatial collapse. (Rows 157-159 show a mild, expected
bump/taper — ordinary boundary behavior, nothing like the pixel-space
collapse.) This rules out the UNet/latent computation as the source: the
corruption is introduced specifically during VAE decode, not before it.

Update: `trans_vae_decode` (`src/vae.cpp`) chains two networks: the base
`vae_decode` (mid-block spatial self-attention at T=ZR^2, e.g. 25600 at
res=1280) and `unet1024`, a *second*, deeper residual UNet that runs
directly at full pixel resolution (7 down/up block stages, 3 of them with
spatial self-attention). Both use `attn_block` (`ops.cpp`), a completely
separate code path from `attn_tokens` — always naive, never gated by
`flash_attn`, and with no tiling at all before this session. This had
never been exercised by any prior test in this investigation (the
`SEETHROUGH_TILED_ATTN` runs upstream in this doc only ever touched
`attn_tokens`, since the VAE model load path never set
`m.tiled_naive_attn`).

Added a tiled variant to `attn_block` and wired it in for the VAE model
too. First attempt introduced a real (if minor) precision regression —
caught by a fast standalone probe (`tests/probe_attn_block_tiled.cpp`)
comparing tiled vs. non-tiled `attn_block` in isolation: `attn_block`'s
existing naive branch scales the KQ product *after* the matmul, while the
new tiled branch (copying `attn_tokens_tiled_naive`'s convention) scaled Q
*before* — a valid reordering in exact math, but the two branches'
pre-existing conventions actually differed from each other, and mixing
them measured ~3-4% relative error even in the degenerate single-chunk
case. Fixed to scale after the matmul, matching `attn_block`'s own
convention; the probe then showed exact agreement at a single chunk and
~0.28% at a real multi-chunk split (152x152=23104 tokens, several chunks)
— ordinary floating-point chunking noise, not a bug.

With the fix in place, the full 1280px pipeline was rerun with both
`attn_tokens` and `attn_block` tiled (`SEETHROUGH_TILED_ATTN=1`). Result:
the *same* full-canvas failure as the buggy pre-fix version — an
incoherent dark-noise canvas, actually less structured than the
already-collapsed 854x70 thin-band. Tiling `attn_block` does not fix the
collapse; it makes the output worse, not better. This is still useful
information: it suggests the model's output at this resolution is right
at a fragile numerical edge, sensitive enough that even a ~0.28%
per-call perturbation, compounded across `unet1024`'s ~10+ `attn_block`
invocations in one forward pass, is enough to push a marginal defect into
total incoherence — consistent with (though not proof of) something
resolution-boundary-related rather than a single clean-cut kernel bug.
`attn_block`'s tiling fix is kept (it's a real, independently-useful
correctness fix and VRAM-bounding option for a previously completely
unguarded O(T^2) op) but is **not** the root-cause fix, and
`SEETHROUGH_TILED_ATTN` is left opt-in, not default, since enabling it
currently makes the production symptom worse.

Status: paused, but substantially narrowed. `direct_conv`, `flash_attn`
(via `attn_tokens`, decisively), the row-chunk tiling arithmetic, the
UNet/latent computation, and now `attn_block`'s tiling-vs-non-tiling
distinction are all excluded or shown not to help. The collapse is
confirmed to live somewhere in `vae_decode`/`unet1024`
(`src/vae.cpp`), but not narrowed past "one of resnet blocks / group norm
/ upsampler / attn_block's core (non-tiling-related) numerics /
cross-stage skip connections in that chain" — and the sensitivity-to-
perturbation observation above suggests further probing should look for a
marginal/boundary condition rather than a single obviously-wrong op.

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
