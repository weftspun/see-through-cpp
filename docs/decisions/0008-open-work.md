# Open work

* Status: accepted
* Date: 2026-07-18

## Context and Problem Statement

MADR 0004, 0007, and 0009 record completed work. This record tracks
everything still open; items move out (to 0009, or a new completed-work
record) as they land on `main`.

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

- [ ] **Blocked on input** — Layer-quality polish vs upstream reference: L/R-split slivers at the
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
      **Second repro attempt, still not reproducible**: tried two more
      upstream test images (`test_image1.png` — simple, hands hidden;
      `test_image2.png` — ornate multi-layer kimono, black background) at
      full production settings (1280px/30 steps). Two apparent leads, both
      false: (1) a cheap 512px/8-step smoke run on `test_image1` showed a
      sharp pad-boundary artifact and hazy legwear content — did not
      reproduce at full 1280px/30-step settings on the same image, so it
      was a low-step/low-res artifact, not a production-scale bug; (2) the
      `bottomwear`/`back_hair` layers from `test_image2` appeared to have
      a solid muddy-purple wash covering most of the canvas instead of a
      transparent background — turned out to be a rendering artifact of
      quickly eyeballing the raw PNG rather than a real defect: every
      "washed" pixel has alpha=0 (confirmed via numpy histogram), and
      compositing properly with `Image.alpha_composite` over white shows
      both layers are actually clean, correctly cropped, full-detail
      content. **Lesson for future debugging**: always alpha-composite
      before judging a layer PNG by eye — a naive viewer can make a
      fully-transparent region with leftover non-zero RGB look like an
      opaque defect. The one remaining oddity (a garbled secondary blob in
      `test_image2`'s `headwear` layer, alongside two clean hair-bow
      shapes) reads as a genuine model-fidelity limit on this character's
      unusually complex layered hair ornaments (chains, tassels) rather
      than an implementation bug — no seam, no alpha-floor cutoff, no
      coordinate error, just softer detail on a hard case. Still no
      actionable repro found across three different characters/poses at
      production settings; this item stays open pending either a restored
      upstream sample set or a user-reported concrete case.
      **Confirmed blocked (2026-07-19)**: no test case is available and
      none of this session's own blind search (3 characters, both cheap
      and production settings) turned up an actionable repro. Stopping
      further speculative search here — the remaining productive path is
      external (a restored `assets/upstream_samples/` set or a concrete
      user report), not more guessing. Revisit if either becomes
      available; no further time should be spent on this item until then.
