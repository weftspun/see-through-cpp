# Port close-out: completed work record

* Status: accepted
* Date: 2026-07-18

## Context and Problem Statement

M1–M13 of the port plan are implemented; every component has a numeric gate
against the upstream PyTorch pipeline (see README status table). This record
lists what was completed during the close-out; everything still open lives in
MADR 0005.

## Completed

- [x] M1–M13 numeric gates (see README table); Vulkan gates on RTX 4090
- [x] Full-resolution GPU pipeline fits in 24GB: flash attention,
      per-frame spatial transformer chunking, f16 skips, direct conv for
      UNets, row-chunked exact-im2col decode (gated at 2.3e-2)
- [x] Root-caused and fixed the empty-layer failures: (1) direct conv
      corrupted the VAE-encoder path -> scoped to UNets; (2) Vulkan direct
      conv zeros at 1280^2 -> row-chunked im2col decode; (3) decoder output
      is ARGB, not RGBA -> CLI layer assembly reorders (alpha = channel 0)
- [x] Verified crisp per-layer decomposition at 512px on the real sample
      (front/back hair, topwear, face, legwear grids)
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
- [x] MADR set split: 0001-0003 (history), 0004 (this record), 0005 (open)

## Consequences

Open items — including final validation on the upstream sample — are tracked
in MADR 0005.
