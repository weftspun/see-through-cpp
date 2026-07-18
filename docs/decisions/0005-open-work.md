# Open work

* Status: accepted
* Date: 2026-07-18

## Context and Problem Statement

MADR 0004 records the completed close-out of the port. This record tracks
everything still open; items move to 0004's Completed list as they land on
`main`.

## Checklist

### Validation

- [ ] Full-quality run on the upstream sample (`common/assets/test_image.png`,
      1280px/30 steps, GPU) with verified per-layer content (in flight)
- [ ] Diagnose the 1280px layer wash observed on `assets/sample.png`
      (512px layers are crisp; decoded 1280 frames reported healthy stats —
      suspect post-decode assembly at 1280 or resolution-dependent haze)
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
