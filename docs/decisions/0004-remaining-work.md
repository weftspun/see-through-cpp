# Port close-out: completed work record

* Status: accepted
* Date: 2026-07-18

## Context and Problem Statement

M1–M13 of the port plan are implemented; every component has a numeric gate
against the upstream PyTorch pipeline (see README status table). This record
lists what was completed during the close-out; everything still open lives in
MADR 0007.

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
- [x] `load` vs `load_backend` tensor-count difference: benign — the gguf
      holds exactly 248 unique tensors; the host-path count includes ggml's
      internal data-blob tensor
- [x] psd_sdk writer swap: CLOSED as won't-do (YAGNI) — replaced by layered
      SVG output; psd_sdk was never vendored, the minimal PSD writer/gates
      are deleted
- [x] Property-tested DAG quality gates rebuilt in Lean4: `verify/` (Lake
      package, `plausible-witness-dag` + the `seethrough_c` FFI,
      `st_witness_check_flat`) replaces the earlier rapidcheck-based C++
      property tests. `verify/KernelGate.lean`'s `productionDomain` (15
      cases: direct conv at all 3 UNet latent levels, row-chunked decode at
      all 3 pixel levels, flash attention at every token-count/frame-batch
      combination) reports VERIFY PASS on Vulkan/RTX 4090 — no kernel-level
      defect at any 1280px production shape. Two apparent "defects" found
      along the way were bisected to Vulkan single-allocation buffer-size
      limits in the *naive reference* path (never used in production), not
      real flash-attention divergence; a `knownDefects` regression-guard
      case was tried and dropped after the historical zero-output symptom
      didn't reproduce deterministically from a fresh random witness.
- [x] Fixed `--png-dir`: the CLI parsed the flag but never used it (lost in
      the SVG/chunk-removal refactor) — per-layer z-ordered PNG export now
      writes alongside the SVG.
- [x] Q4_0 quantization landed end-to-end: `convert_diffusers_to_gguf.py
      --ftype 2` quantizes nn.Linear weights only (self/cross-attn, GEGLU
      FF — matching upstream's own bitsandbytes NF4 scope, which also
      skips conv layers), verified bit-exact against ggml's own
      `quantize_row_q4_0_ref`. No C++ changes needed — the native gguf.h
      reader + `ggml_mul_mat` already handle quantized weights on Vulkan.
      `verify/QuantDesign.lean` design-searches 12 real `layerdiff-unet`
      Linear shapes, all within a 15% tolerance (measured ~7.5-10%).
      Requantized layerdiff-unet/marigold-unet/layerdiff-te2/marigold-te:
      12.65GB -> 4.86GB (61.6%); a 512px/4-step smoke run with the
      quantized weights produced coherent layers (clear hair strands,
      clothing) — see MADR 0005.
- [x] `pipe_backend` refuses to silently fall back to CPU when no Vulkan
      device is found and `--device cpu` wasn't explicitly requested —
      GPU is the primary target per project policy, not an implicit-on-
      failure fallback.
- [x] SVG `<image>` elements carry a semantic `id` (sanitized tag name,
      e.g. `id="topwear"`; depth-group counterparts use `id="depth-
      <tag>"`) alongside the existing `data-tag` attribute, so SVG
      consumers (e.g. a future ThorVG-based collation tool) can look
      layers up by id.

## Consequences

Open items — including final validation on the upstream sample — are tracked
in MADR 0007.
