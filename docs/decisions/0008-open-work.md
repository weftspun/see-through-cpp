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
      a content-computation defect in any op path tried so far. This is
      the next angle to investigate. See docs/ggml-upstream-issues.md item
      4's latest status note.
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
      match) + depth ordering. Reversed the earlier ThorVG-vs-PSD-reader
      decision: our own SVG needs no parser dependency to read back (it's
      our own flat, documented format); the actual blocker was reading
      *upstream's* PSD output. Vendored `psd_sdk` (MolecularMatters) as a
      git subtree and added `tests/psd_layers.cpp`, which reads a PSD and
      writes one PNG per layer (RGBA, expanded to full canvas) — the
      reader half is done. Not yet validated against a real upstream PSD
      (none available locally — need to run upstream once to generate
      one) and the actual IoU-comparison step (match by tag name, compute
      per-tag alpha-mask IoU) isn't written yet.
- [ ] Full-quality run with the Q4_0-quantized models (currently only
      smoke-tested at 512px/4 steps) — same 1280px/30-step + upstream
      parity bar as the f16 baseline, see MADR 0005

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
