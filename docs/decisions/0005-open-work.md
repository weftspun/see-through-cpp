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

1. Each subgraph in the DAG (conv, token attention, transformer chunking,
   decode, graph-reuse) has a rapidcheck property whose generators span the
   **production envelope** — including the 1280px shapes (160×160 latents,
   80×80×13-frame token counts, 1280² decode) — run on the primary device.
2. A knob may be enabled in the pipeline only for configurations inside its
   gated envelope. Enabling it for a new shape/backend first extends the
   gate (this is the invariant that would have blocked the direct-conv and
   flash escapes).
3. Randomized properties are backed by fixed regression probes for every
   defect found (they double as upstream repros —
   docs/ggml-upstream-issues.md).
4. Visual/e2e checks remain the outermost gate: raw-tensor equality cannot
   catch assembly-order bugs (the ARGB lesson); a stored-preview composite
   is not evidence.

## Checklist

### Validation

- [ ] Property-gate the full production envelope per policy above:
      extend generators to 160×160-latent shapes for direct conv (fixed
      probes just added), flash attention at Tq=6400/B=13 (cross-backend:
      GPU flash vs CPU naive), spatial/temporal chunk equivalence at
      S=6400/F=13, row-chunked decode at 1280²
- [ ] Full-quality run on the upstream sample (`common/assets/test_image.png`,
      1280px/30 steps, GPU) with verified per-layer content (in flight)
- [ ] Diagnose the 1280px corruption (decoded 1280 frames are garbage while
      512 is crisp; every GPU knob was gated only at 64×64-latent shapes —
      resolution bisect at 1024 + production-scale probes running)
- [ ] Layer-quality polish vs upstream reference: L/R-split slivers at the
      pad boundary, faint head-pass alphas, alpha floor tuning
- [ ] Upstream parity: `inference_psd.py --save_to_psd` on the same
      seed/input; per-layer alpha-mask IoU (>0.98 where tags match) + depth
      ordering
- [x] `load` vs `load_backend` count difference: benign — the gguf holds
      exactly 248 unique tensors; the host-path count includes ggml's
      internal data-blob tensor

### Hardening

- [ ] Reimplement `write_psd` on psd_sdk behind the same API; keep the
      `test_psd` + `check_psd.py` byte-exact gate; delete the minimal writer
      once green. Also write a real (layer-composited) merged preview so
      viewers don't echo the input page.
- [ ] Property tests linked in the Vulkan build tree (rapidcheck now forced
      static there; final link blocked only by the running pipeline's DLL
      locks — rebuild after it exits)

### Repo layout

- [ ] `models/` for `*.gguf` weights; CLI `-m` default becomes `models`
- [ ] `git mv ggml third_party/ggml` for vendoring consistency (forces one
      full rebuild of both build trees)

### Documentation / upstream

- [x] README: GPU-primary usage, --png-dir/--debug-dir, low-VRAM knob
      notes, subtree vendoring policy
- [x] ggml upstream findings written up with local repros:
      docs/ggml-upstream-issues.md (gallocr input recycling, CPU multi-head
      flash divergence, Vulkan direct-conv zeros)
- [ ] Post-MVP (original plan): C ABI + server, trellis2cpp-style
