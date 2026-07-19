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
      `flash_attn` (via `attn_tokens`) are **excluded** for the main
      diffusion UNet. Along the way, found and fixed a real, separate,
      **confirmed** ggml/Vulkan kernel defect: `unet1024`'s first spatial
      self-attention call (`attn_block` in `ops.cpp`, `head_dim=8`, 32
      heads, T=6400) allocates a 5.24GB naive-attention intermediate — over
      Vulkan's ~4.295GB `maxStorageBufferRange` — causing silent
      corruption. Reproduced with synthetic weights and verified against
      independent CPU ground truth (GPU-naive 295x off, the new tiled fix
      1.5% off — normal float tolerance). Kept as opt-in
      (`SEETHROUGH_TILED_ATTN`).
      **However**, this was chasing the wrong stage for the actual visual
      collapse: adding grayscale visualizations to the per-stage taps
      showed `vae_decode`/`unet1024`'s output is flat/featureless at
      res=1280 even with the attn_block fix active, and — critically —
      dumping the raw **latent itself** (before decode runs at all) shows
      it is *already* completely flat at res=1280, while the identical
      frame's latent at res=512 clearly shows the real shape. **The
      collapse is confirmed to originate in the main diffusion UNet's
      forward pass** (`unet_frame_forward`, `unet_frame.cpp`), not
      `vae_decode`/`unet1024` at all — the per-row latent stats check
      earlier in this investigation missed this because a flat field and a
      real image can have similar row-wise averages. Next step: apply the
      same per-stage tap + grayscale-visualization technique to
      `unet_frame_forward`'s resnets/attention/cross-frame blocks to
      bisect where inside the main UNet the signal goes flat. See
      docs/ggml-upstream-issues.md item 4's latest status note.
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
