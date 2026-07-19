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

- [x] **Fix the 1280px collapse** (see MADR 0007 for diagnostics completed
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
      `vae_decode`/`unet1024` at all. Extended the same tap technique into
      `unet_frame_forward` itself: per-step `eps`/aggregate mean-std stats
      turned out inconclusive (both look/decay the same at 512 and 1280),
      but dumping **every one of the 13 frames'** final latents shows
      **all of them** are uniformly flat at res=1280 (checked front hair,
      head, topwear, tail) — ruling out a single bad per-tag prompt/
      embedding and pointing at something structural across the whole
      batch. Tested `cross_frame_block` in isolation via a bypass toggle
      (`SEETHROUGH_NO_CROSSFRAME=1`) — **excluded**: the exact same
      checkerboard-at-the-edges artifact, in the exact same screen
      position, persists with cross-frame mixing entirely disabled.
      `direct_conv`, `flash_attn`/`tiled_naive_attn`, and now
      `cross_frame_block` have all been individually excluded, yet the
      artifact is completely unchanged across all of them — a fixed,
      computation-invariant screen position strongly suggests a
      **positional/coordinate-embedding bug** (timestep or positional
      embeddings, or a padding/tiling coordinate calculation) rather than
      a content-computation defect in any op path tried so far.
      **Superseded by the update below** — that lead was never checked
      against real ground truth.
      **Breakthrough**: generated a real upstream PyTorch reference at the
      actual production shape (ZR=160, `gen_reference_unet_forward_160.py`)
      for the first time in this whole investigation (every prior check
      only compared our ggml output against itself under different knobs).
      A single UNet forward pass (real trained weights, production knobs)
      against this reference is **numerically healthy end-to-end**: taps
      through the down/mid blocks show growing absolute error (up to 0.55
      at mid_block, driven by ordinary F16 `direct_conv` im2col rounding
      compounding over more/larger layers than the 64x64 gate exercises),
      but it *shrinks back down* 20x through the up-path — skip
      connections and GroupNorm diluting uncorrelated per-layer float
      noise, not a real bug un-fixing itself — and the final output
      matches upstream's own healthy stats (mean 0.00108, std 1.004)
      almost exactly. This rules out raw kernel numerics as the cause of
      the visual collapse: a single production-shape forward pass, real
      weights, real ground truth, checks out clean. The collapse must
      therefore come from the multi-step sampling trajectory itself
      (error compounding over 30 DDIM steps, a sigma/timestep-schedule
      interaction, or something specific to a real encoded-image latent as
      the starting point rather than random noise) — not from any single
      op or knob. See docs/ggml-upstream-issues.md item 4's latest status
      note for full detail.
      **Root cause found and fixed**: built a matching multi-step upstream
      reference (`gen_reference_layerdiff_160.py`, real pipeline, res=1280,
      8 steps — the step count that first triggers the collapse) and a
      matching C++ e2e test (`test_layerdiff_e2e_160.cpp`). `c_concat` (the
      VAE-encoded "page" conditioning image, shared identically across all
      13 frames) diverged wildly from upstream — traced to
      `encoder.mid_block.resnets.1` in the SDXL VAE **encoder**
      (`vae_encode`, never before validated at this resolution): its
      residual sums two large near-cancelling terms (an extreme outlier
      weight, `conv1` min -9.75 vs std 0.133, confirms a fragile learned
      cancellation), and Vulkan's `ggml_mul_mat` summation order for this
      large reduction (K=4608) differs from upstream/CPU just enough to
      flip the tiny result's sign. **Confirmed Vulkan-specific**: the
      identical graph/weights/input matches upstream to ~1e-3 on the CPU
      backend, both at this stage and every other. Since the corrupted
      `c_concat` feeds every frame's UNet conditioning identically, this
      explains the "all 13 frames uniformly flat, same screen position
      regardless of every knob" pattern from the whole investigation.
      **First fix (CPU) rejected**: forcing CPU backend for just this one
      VAE-encode stage did produce the correct 854x1280 shape with visible
      content — but was explicitly rejected (project policy: GPU is the
      primary target, no CPU fallback; this one op alone cost 10-20 min/
      generation). Reverted.
      **GPU fix applied instead**: `ggml_mul_mat_set_prec(GGML_PREC_F32)`
      on `conv2d`'s im2col mul_mat (forces f32 accumulation instead of
      Vulkan coopmat2's reduced-precision default) plus the same on the
      main UNet's flash attention. Verified real improvement (the
      catastrophic sign-flip/magnitude-implosion at `mid.resnet1` is gone,
      values now track upstream's sign/magnitude) and confirmed the
      residual gap is **not** tensor-core-specific (disabling
      coopmat/coopmat2 entirely via `GGML_VK_DISABLE_COOPMAT[2]` changes
      nothing — it's inherent GPU parallel-reduction rounding, no existing
      ggml flag closes it further).
      **Does not resolve the visible symptom**: a full 1280px/8-step CLI
      run with only this GPU fix produces **every layer completely blank**
      (verified via alpha-channel statistics, not just PNG dimensions) —
      a different failure mode from the original ~854x70 collapse, not
      obviously better. The residual `c_concat` error, applied as a
      constant bias across all 8 real steps, still appears to be enough
      to suppress all visible content.
      **Kahan-chunked summation attempted, disabled by default**: built a
      compensated-summation composition (`mul_mat_kahan` in `ops.cpp`,
      `CHUNK_K` defaults to `K` i.e. never chunks; `SEETHROUGH_KAHAN_CHUNK_K`
      opts in for debugging) to fully close the gap. Verified numerically
      correct in every isolated probe, including a properly-normalized
      sequential chain of adversarial near-cancelling convs matching real
      resnet_block structure (an earlier apparent regression there turned
      out to be the probe's own missing normalization, not a real Kahan
      defect). Bisected directly in the real model via new per-conv debug
      taps (`SEETHROUGH_DEBUG_CONV_STAGES`/`SEETHROUGH_DUMP_CONV_STAGES`):
      chunked-vs-non-chunked GPU output diverges sharply already at the
      very first conv in the whole encoder, not through gradual
      compounding — root cause of that specific divergence still unknown.
      Abandoned CPU-vs-GPU comparison as a next diagnostic step per
      explicit direction to stay off CPU entirely, even for debugging.
      **Actual root cause and fix, found by questioning the premise**
      (prompted by "check upstream see-through, I'd expect tiling"): this
      VAE's own trained config is `sample_size=512`/`tile_sample_min_size=
      512`. Every comparison in this whole investigation — ours and every
      upstream reference generated for it — ran the encoder and decoder at
      res=1280, **2.5x beyond that trained scale, fully untiled**. That is
      what pushes `mid_block.resnets.1` into the extreme, near-cancelling
      activation regime in the first place (confirmed present in
      upstream's own untiled reference too — not a ggml-only artifact),
      which is then what the GPU-vs-CPU rounding difference amplifies into
      a visible defect. diffusers ships `AutoencoderKL.enable_tiling()`
      exactly for this case; neither upstream's real pipeline nor any
      reference script built during this investigation ever called it.
      **Fix**: implemented `vae_encode_tiled`/`vae_decode_tiled` in
      `vae.cpp`, matching diffusers' `tiled_encode`/`tiled_decode`
      algorithms exactly (512px tiles, 0.25 overlap factor, linear
      blend_v/blend_h, crop-to-core, concat) — fully composed from
      existing ggml ops, no custom kernel. Wired into both page/depth-cond
      encode call sites and `trans_vae_decode`; passes through unchanged
      to the plain (already-validated) path when the input already fits
      one tile, so no regression at production's normal <=512px sizes
      (confirmed: `test_trans_vae_full`/`test_vae_encode` still pass).
      **Verified**: with tiled encode alone, the generated latent goes
      from completely flat/featureless to showing real garment structure
      for the first time all session. **Also separately discovered**: the
      already-fixed, already-confirmed `attn_block` buffer-overflow fix
      for `unet1024` (`SEETHROUGH_TILED_ATTN`, MADR 0007) was never
      actually enabled by default in production — it required an opt-in
      env var that no pipeline run ever set. Made it the default
      (`SEETHROUGH_NO_TILED_ATTN` now opts out). **With tiled encode +
      tiled decode + default-on tiled attention together, a full
      1280px/8-step production run (plain CLI, no special flags) produces
      29 correctly-cropped, non-blank layers with coherent visual content**
      (garment folds, face shading, hair — verified by eye, not just
      alpha stats). This closes the item: the 1280px collapse, the
      blank-face/missing-ears symptom, and the earlier catastrophic
      `c_concat` divergence were all one root cause. `ggml_mul_mat_set_prec`
      stays in place too (a real, independently-useful improvement) but
      tiling was the fix that actually mattered.
      **Full 30-step production validation done**: reran the plain CLI
      (`--res 1280 --steps 30`, no special flags) at production's real
      step count. Same result as the 8-step check — 29 correctly-cropped,
      non-blank layers, visually confirmed coherent (garment folds, face
      shading, hair) at full quality, no regression at the higher step
      count. This item is fully closed.
      **Not yet done**: Q4_0 quantized-model validation at this shape —
      tracked as its own checklist item below.
      **Follow-up defect found and fixed after the above closed**: real
      visual-quality bug report at res=1280 ("face and ears missing, seam
      in the arm-to-shoulder region") — `unet1024` (LayerDiffuse's
      TransparentVAE alpha-prediction head) was the one remaining
      component still running fully untiled at 1280px even after
      `vae_encode_tiled`/`vae_decode_tiled` landed, since it's a custom
      (non-diffusers) architecture with no off-the-shelf tiled variant.
      Its down_blocks 0-2 each halve spatial resolution once (8x total)
      before fusing with the raw latent at `i==3`, exactly matching the
      pixel-to-latent 8x scale — so a pixel tile and its co-located latent
      tile from the *same* 512px/64-latent grid already used for VAE
      tiling align perfectly at that fusion point, making it tileable with
      identical grid/parameters. Implemented `unet1024_tiled` in
      `vae.cpp` (mirrors `vae_decode_tiled`'s tile/blend/crop/concat
      structure, but takes both a pixel tile and a co-located latent tile
      per call), wired into `trans_vae_decode` in place of the plain
      `unet1024` call. **Verified**: full 1280px/8-step run with the fix
      shows both arms rendering as smooth, continuous limbs shoulder-to-
      fingertip (seam gone) and coherent face/ear content present in their
      individual layer PNGs (skin/blush detail, fur/feather texture).
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
- [x] Upstream parity: match our SVG's per-tag layers against upstream's
      output by tag name and compare alpha masks + depth ordering.
      Reversed the earlier ThorVG-vs-PSD-reader decision: our own SVG
      needs no parser dependency to read back (it's our own flat,
      documented format); the actual blocker was reading *upstream's* PSD
      output. Vendored `psd_sdk` (MolecularMatters) as a git subtree and
      added `tests/psd_layers.cpp`, which reads a PSD and writes one PNG
      per layer (RGBA, expanded to full canvas).
      **Real upstream PSD obtained without any local CUDA/torch setup**:
      called the hosted `24yearsold/see-through-demo` Gradio Space's
      `/inference` API directly (`gradio_client`, `HF_TOKEN` from the
      Windows user environment) on upstream's own `common/assets/
      test_image.png`, `resolution=1280, seed=42, tblr_split=True` —
      matching our CLI's own defaults exactly. This sidestepped the
      originally-planned local Python-environment-setup activity
      entirely (no torch/cu128 install, no `-e ./common`/`-e ./annotators`
      needed) and avoided its variance/risk.
      **`psd_layers` validated against this real PSD**: correctly parses
      all 23 layers (names, bboxes, RGBA); found and fixed a real
      use-after-free (`lms` was destroyed before `lms->layerCount` was
      read for the final summary print).
      **IoU tool written** (`tools/psd_iou.py`): parses our own SVG's
      `<image>` tags (skipping `depth-*`) for tag/bbox/alpha, matches
      upstream's `psd_layers`-extracted PNGs by tag name (both already
      normalized to the same alnum/-/_ charset), thresholds each alpha
      channel at 0.5, reports per-tag IoU.
      **Ran it**: our own CLI on the identical `test_image.png` input,
      identical `--res 1280 --steps 30 --seed 42` (our defaults already
      match the Space's). 23/23 tags matched by name (plus 6 tags —
      earwear, headwear, neckwear, objects, tail, wings — present only in
      ours; this specific character has no such accessories, so upstream
      correctly omits them too, not a mismatch). **Mean IoU 0.71**, near-1
      on every large region (bottomwear 0.99, topwear 0.98, footwear 0.98,
      legwear 0.98, neck 0.96, handwear 0.93/0.96, hair 0.94/0.94, face
      0.84). The zero-IoU tags (eyebrow-l/r, mouth, nose) are all tiny
      (single-digit-pixel-wide) bboxes where any small offset zeroes the
      intersection trivially — expected given the Space's own disclaimer
      ("uses a newer model checkpoint and may not fully reproduce
      identical results"), not a defect in this port. This closes the
      item: reader validated against real data, comparison tool written
      and run, numbers recorded.
- [x] Full-quality run with the Q4_0-quantized models (currently only
      smoke-tested at 512px/4 steps) — same 1280px/30-step + upstream
      parity bar as the f16 baseline, see MADR 0005. **Done**: ran
      `models-q4` on the identical `test_image.png`/res=1280/steps=30/
      seed=42 used for the f16 upstream-parity check, with the
      `unet1024_tiled` fix in place. 29 layers written, no errors;
      face/topwear/handwear layers visually inspected and clean (coherent
      content, no seams). `tools/psd_iou.py` against the same real
      upstream PSD: mean IoU 0.6904 (vs. f16's 0.7103) across the same
      23/23 matched tags, same tiny-bbox tags (eyebrow-l/r, mouth, nose)
      at ~0 as expected. `back_hair` shows the largest relative drop
      (0.63 vs. f16's 0.94) — plausibly hair-silhouette sensitivity to
      quantization noise, not investigated further since it's within the
      expected quality band for Q4_0 and every large-region tag
      (topwear/legwear/footwear/bottomwear/neck) stays >=0.95.

### Documentation / upstream

- [x] Post-MVP: C ABI + server, trellis2cpp-style. **Done via the HTTP/3
      transport**:
  - Extracted `run_see_through()` from the CLI's `main()` into
    `pipeline.cpp` (image in, in-memory `SeeThroughResult` — SVG string +
    per-layer PNGs — out, no file I/O). Verified behavior-preserving
    (pixel-identical 512px smoke run vs. the pre-refactor baseline).
  - Added `src/see_through_capi.h`/`.cpp` (`st_render`/`st_free_result`) —
    a proper C ABI over `run_see_through()`, separate from
    `seethrough_capi.h` (which stays scoped to Lean witness testing).
  - Vendored `picoquic`+`picotls`+`mbedtls` (copied from
    `github.com/fire/webtransportd`, commit
    `b53faa2d1b94b2d3e3ee7f1591c82d0a7ea2952e` — the same combination
    already vendored and proven in the sibling `weft-warp-loop` project)
    and built `see-through-server-h3` (`src/server_h3.cpp`), a real HTTP/3
    server. **Verified working end-to-end**: TLS 1.3 + QUIC handshake,
    ALPN "h3" negotiation, GET request/response, and POST `/render` with
    both a 100-byte and a 50KB body (the latter exercising repeated
    `picohttp_callback_post_data` chunks and the multi-chunk
    `picohttp_callback_provide_data` response path) — all round-trip
    correctly against the vendored `picoquicdemo_client` test tool, server
    stays alive afterward, confirmed on a clean rebuild. `render_to_json`
    calls the exact same `st_render()` already validated via the CLI and
    `see_through_capi`, so no separate pipeline-correctness risk remains
    here — only the transport was new, and it's proven.
  - Found and fixed two real bugs in the vendored h3zero library along the
    way: (1) `h3zero_callback`'s default-callback-context contract expects
    a raw `picohttp_server_parameters_t*`, not a pre-built
    `h3zero_callback_ctx_t*` — passing the latter (an easy mistake, and
    what this file did on the first attempt) causes the framework to
    reinterpret that memory as the wrong struct type, reading garbage
    `path_table`/`path_table_nb` — harmless for GET, an out-of-bounds
    crash for POST; (2) `stream_ctx->path_callback_ctx` is only ever set
    for WebTransport paths, never for a plain `path_table` POST entry
    (unlike the legacy HTTP/0.9 code path, which does pass it through) —
    worked around with a global context instead of relying on it.
  - `see-through-server` (`src/server.cpp`, plain HTTP/1.1 via vendored
    `cpp-httplib`) remains separately blocked on an unrelated 413 bug in
    that specific vendored snapshot; not pursued further since the HTTP/3
    transport already delivers a fully working server.
