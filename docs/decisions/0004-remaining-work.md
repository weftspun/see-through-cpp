# Remaining work to close out the see-through.cpp port

* Status: accepted
* Date: 2026-07-18 (last updated 2026-07-18)

## Context and Problem Statement

M1–M13 of the port plan are implemented; every component has a numeric gate
against the upstream PyTorch pipeline (see README status table). The CLI runs
end-to-end on CPU and on Vulkan (RTX 4090) with a low-VRAM UNet configuration
(flash attention, per-frame spatial transformer chunking, f16 skip
connections, direct convolutions for the UNets, row-chunked im2col for the
decode). GPU is the primary target; CPU is the explicit fallback.

## Decision Outcome

Track the close-out below; items move from Incomplete to Completed as they
land on `main`.

## Incomplete

### Validation

- [ ] Visual verification of the ARGB-channel-order fix: the current
      full-quality run's per-layer grid must show clean, well-formed layers
      (a stored-preview composite is NOT evidence — psd-tools `composite()`
      returns the embedded merged image unless forced)
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

## Completed

- [x] M1–M13 numeric gates (see README table); Vulkan gates on RTX 4090
- [x] Full-resolution GPU pipeline fits in 24GB: flash attention,
      per-frame spatial transformer chunking, f16 skips, direct conv for
      UNets, row-chunked exact-im2col decode (gated at 2.3e-2)
- [x] Root-caused and fixed the empty-layer failures: (1) direct conv
      corrupted the VAE-encoder path -> scoped to UNets; (2) Vulkan direct
      conv zeros at 1280^2 -> row-chunked im2col decode; (3) decoder output
      is ARGB, not RGBA -> CLI layer assembly reorders (alpha = channel 0)
- [x] `--debug-dir` stage stats (c_concat/eps/latent/per-frame decode)
- [x] `--png-dir`: per-layer RGBA + depth PNGs in z order + layers.json
- [x] GPU is the default device for tests and CLI; CPU is opt-in fallback
- [x] Vendored as subtrees: ggml, psd_sdk, rapidcheck
- [x] Property tests (logic + graph level); catches so far: PackBits
      expansion bound, CPU multi-head flash divergence, and regression
      probes for the direct-conv defect
- [x] Repo layout: scripts/, tests/, gen_reference/ (generators + data),
      reference archive on the v0.0.1-dev release (reference_bins.tar.zst)
- [x] ggml input-buffer gotcha documented in memory + commit history
- [x] MADR set split into 0001-0004

## Consequences

The port is feature-complete once Validation is green; hardening reduces
maintenance risk but does not block use.
