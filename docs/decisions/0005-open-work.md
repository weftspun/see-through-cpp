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
- [ ] Investigate `load` vs `load_backend` tensor-count difference
      (249 vs 248 on marigold-vae; likely benign duplicate name)

### Hardening

- [ ] Reimplement `write_psd` on psd_sdk behind the same API; keep the
      `test_psd` + `check_psd.py` byte-exact gate; delete the minimal writer
      once green. Also write a real (layer-composited) merged preview so
      viewers don't echo the input page.
- [ ] Property tests linked in the Vulkan build tree (rapidcheck target
      there still fails to link; reconfigure/rebuild once no process locks
      the ggml DLLs)

### Repo layout

- [ ] `models/` for `*.gguf` weights; CLI `-m` default becomes `models`
- [ ] `git mv ggml third_party/ggml` for vendoring consistency (forces one
      full rebuild of both build trees)

### Documentation / upstream

- [ ] README: low-VRAM knobs (direct conv is UNet-only — wrong for the VAE
      encoder s2/p0 path and silently zero on Vulkan >= ~1280^2, hence the
      row-chunked im2col decode), GPU-primary usage incl. `build-vulkan/bin`
      on PATH, `--png-dir`, subtree vendoring policy
- [ ] Minimal repro + issue text for ggml upstream:
      (a) gallocr recycles INPUT tensor buffers after their last in-graph
      read — all inputs must be re-set before every compute of a reused
      graph; (b) CPU flash_attn_ext diverges ~0.3 from naive attention for
      n_head >= 2 (single-head agrees to f16 precision; Vulkan agrees);
      (c) Vulkan ggml_conv_2d_direct silently returns zeros at very large
      spatial sizes (>= ~1280^2; correct at <= 512^2)
- [ ] Post-MVP (original plan): C ABI + server, trellis2cpp-style
