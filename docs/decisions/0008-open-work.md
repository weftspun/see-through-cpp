# Open work

* Status: accepted
* Date: 2026-07-18

## Context and Problem Statement

MADR 0004 and 0007 record completed work. This record tracks everything
still open; items move to 0007 (or a new completed-work record) as they
land on `main`.

## Guiding concept: property-tested DAG quality gates

Every escaped defect so far lived in the pipeline DAG's *configuration
space*, not in any single op: backend (Vulkan/CPU) × knob (flash attention,
direct conv, spatial/temporal chunking, row-chunked im2col) × **shape at
production scale** × reuse pattern (multi-compute graphs). Reference gates
pin one point of that space; property tests must cover the envelope.

Policy:

1. Each subgraph in the DAG (conv, token attention) has a witness case in
   `verify/KernelGate.lean`'s `productionDomain`, spanning the **production
   envelope** — including the 1280px shapes (160×160 latents, 80×80×13-frame
   token counts, 1280² decode) — checked via the `seethrough_c` FFI
   (`st_witness_check_flat`) on the primary device. `verify/QuantDesign.lean`
   applies the same FFI as a design-space search rather than a hard gate
   (see Hardening below).
2. A knob may be enabled in the pipeline only for configurations inside its
   gated envelope. Enabling it for a new shape/backend first extends
   `productionDomain` (this is the invariant that would have blocked the
   direct-conv and flash escapes).
3. Found defects are written up in docs/ggml-upstream-issues.md with a
   local repro. (A `knownDefects` regression-guard list was tried and
   dropped — the historical zero-output symptom didn't reproduce
   deterministically from a fresh random witness at the same shape, so
   asserting it stayed detectable was just flaky, not a real guard.)
4. Visual/e2e checks remain the outermost gate: raw-tensor equality cannot
   catch assembly-order bugs (the ARGB lesson); a stored-preview composite
   is not evidence.

## Checklist

### Validation

- [ ] **Fix the 1280px collapse** (see MADR 0007 for diagnostics completed
      so far — this single bug also causes the blank-face/missing-ears
      symptom, not a separate "content-quality" bug): `direct_conv` and
      `flash_attn` (via `attn_tokens`) are **excluded**. Per-stage
      instrumentation (`SEETHROUGH_DEBUG_DECODE_STAGES=1`) pinned a real,
      **confirmed** bug: `unet1024`'s first spatial self-attention call
      (`down_blocks.4.attentions.0`, `attn_block` in `ops.cpp`) explodes
      from normal values to ~50000 in one step. Root cause: at this
      shape (T=6400, head_dim=8, 32 heads), the naive `(Tk,Tq,H,N)`
      intermediate is 5.24GB — over Vulkan's ~4.295GB
      `maxStorageBufferRange` single-allocation limit, the same boundary
      already implicated elsewhere in this doc. Reproduced with pure
      synthetic weights (not real-checkpoint-specific) and verified
      against independent CPU ground truth: GPU-naive is 295x off, the new
      tiled fix (`Model.tiled_naive_attn`, `SEETHROUGH_TILED_ATTN=1`) is
      1.5% off (normal float tolerance) — a real, confirmed, and fixed
      ggml/Vulkan kernel defect. **However**, the full 1280px CLI still
      produces the same visually-collapsed output even with this fix
      active, so it is not the sole cause — either a second instance of
      the same buffer-overflow class exists elsewhere in the graph, or a
      separate issue coexists at this resolution. See
      docs/ggml-upstream-issues.md item 4's latest status note; next step
      is inspecting per-stage *images* (not just aggregate stats) with the
      fix active, starting from `down4.attn0` itself.
- [ ] Layer-quality polish vs upstream reference: L/R-split slivers at the
      pad boundary, faint head-pass alphas, alpha floor tuning. **Checked,
      not currently reproducible**: audited every L/R-split tag
      (handwear, ears, eyewhite, eyebrow, eyelash, irides) across a fresh
      30-step 512px run on `assets/sample.png` plus the pre-existing
      `assets/final30_layers`/`final_svg_layers` outputs from earlier this
      session — no visible pad-boundary slivers on any of them.
      `assets/upstream_samples/` (referenced when this item was written) no
      longer exists locally, so the specific input/pose that motivated this
      item couldn't be re-tried. Needs either that sample set restored or a
      concrete repro (which image/tag/seed) before a fix can be targeted —
      speculative code changes without a visible symptom aren't worth it.
- [ ] Upstream parity: match our SVG's per-tag layers against upstream's
      output by tag name and compare alpha masks (>0.98 IoU where tags
      match) + depth ordering. Decided against comparing against upstream's
      PSD (no PSD reader in our C++, and we don't want a Python dev-tool
      dependency for this) — needs a C++ tool using ThorVG to parse SVG,
      and a non-PSD upstream reference to compare against (open: what that
      reference format is, since upstream only emits PSD).
- [ ] Full-quality run with the Q4_0-quantized models (currently only
      smoke-tested at 512px/4 steps) — same 1280px/30-step + upstream
      parity bar as the f16 baseline, see MADR 0005
