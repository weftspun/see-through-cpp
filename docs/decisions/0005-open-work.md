# Open work

* Status: accepted
* Date: 2026-07-18

## Context and Problem Statement

MADR 0004 records the completed close-out of the port. This record tracks
everything still open; items move to 0004's Completed list as they land on
`main`.

## Guiding concept: property-tested DAG quality gates

Every escaped defect so far lived in the pipeline DAG's *configuration
space*, not in any single op: backend (Vulkan/CPU) × knob (flash attention,
direct conv, spatial/temporal chunking, row-chunked im2col) × **shape at
production scale** × reuse pattern (multi-compute graphs). Reference gates
pin one point of that space; property tests must cover the envelope.

Policy:

1. Each subgraph in the DAG (conv, token attention) has a witness case in
   `verify/Verify.lean`'s `productionDomain`, spanning the **production
   envelope** — including the 1280px shapes (160×160 latents, 80×80×13-frame
   token counts, 1280² decode) — checked via the `seethrough_c` FFI
   (`st_witness_check_flat`) on the primary device.
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

- [x] Production-scale probes at every kernel/shape the 1280px CLI can emit
      (direct conv at all 3 UNet latent levels, row-chunked decode at all 3
      pixel levels, flash attention at every token-count/frame-batch
      combination) — `verify/Verify.lean`'s `productionDomain`, 15 cases,
      all contained (VERIFY PASS on Vulkan/RTX 4090). No kernel-level defect
      found at production shapes; the earlier 1280 corruption is not
      explained by conv or attention numerics.
- [ ] Full-quality run on the upstream sample (`common/assets/test_image.png`,
      1280px/30 steps, GPU) with verified per-layer content — last attempt
      died at step 26/30 with no error trace, predates the current binaries;
      needs a fresh run now that the op-level gates are clean
- [ ] If the fresh 1280 run still corrupts despite clean kernel gates, the
      bug is in orchestration (graph assembly/reuse, gallocr input
      recycling across the 30-step loop) rather than any single op —
      narrow with `--debug-dir` frame dumps per step
- [ ] Layer-quality polish vs upstream reference: L/R-split slivers at the
      pad boundary, faint head-pass alphas, alpha floor tuning
- [ ] Upstream parity: `inference_psd.py --save_to_psd` on the same
      seed/input; per-layer alpha-mask IoU (>0.98 where tags match) + depth
      ordering
- [x] `load` vs `load_backend` count difference: benign — the gguf holds
      exactly 248 unique tensors; the host-path count includes ggml's
      internal data-blob tensor

### Hardening

- [x] psd_sdk writer swap: CLOSED as won't-do (YAGNI) — replaced by layered
      SVG output; psd_sdk was never vendored, the minimal PSD writer/gates
      are deleted.
- [x] Lean `verify/` witness gate linked and passing: `seethrough_c`
      rebuilt with the `ST_API` exports, `lake build` links clean, `lake exe
      verify` reports VERIFY PASS over all 15 `productionDomain` cases.

### Repo layout

- [x] models/ for weights (CLI -m default "models"); ggml moved under
      third_party/

### Documentation / upstream

- [x] README: GPU-primary usage, --png-dir/--debug-dir, low-VRAM knob
      notes, subtree vendoring policy
- [x] ggml upstream findings written up with local repros:
      docs/ggml-upstream-issues.md (gallocr input recycling, CPU multi-head
      flash divergence, Vulkan direct-conv zeros)
- [ ] Post-MVP (original plan): C ABI + server, trellis2cpp-style
