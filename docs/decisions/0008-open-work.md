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

- [~] Post-MVP: C ABI + server, trellis2cpp-style. **Substantial progress,
      not fully done**:
  - Extracted `run_see_through()` from the CLI's `main()` into
    `pipeline.cpp` (image in, in-memory `SeeThroughResult` — SVG string +
    per-layer PNGs — out, no file I/O). Verified behavior-preserving
    (pixel-identical 512px smoke run vs. the pre-refactor baseline).
  - Added `src/see_through_capi.h`/`.cpp` (`st_render`/`st_free_result`) —
    a proper C ABI over `run_see_through()`, separate from
    `seethrough_capi.h` (which stays scoped to Lean witness testing).
  - Added `see-through-server` (`src/server.cpp`): a plain HTTP/1.1
    `POST /render` server via vendored `cpp-httplib` (single header, MIT).
    **Blocked**: this specific vendored snapshot (master branch,
    2026-dated) 413s on any body over a few bytes — a real bug in that
    build of the library (reproduced down to a 10KB all-`x` payload with
    `Expect:` disabled, ruling out the usual 100-continue cause); not yet
    root-caused or worked around.
  - Per your direction, additionally vendored `picoquic`+`picotls`+
    `mbedtls` (copied from `github.com/fire/webtransportd`, commit
    `b53faa2d1b94b2d3e3ee7f1591c82d0a7ea2952e` — the same combination
    already vendored and proven in the sibling `weft-warp-loop` project)
    and built `see-through-server-h3` (`src/server_h3.cpp`), a real
    HTTP/3 server on this stack. **Verified working**: a full TLS 1.3 +
    QUIC handshake, ALPN "h3" negotiation, and a complete GET
    request/response round-trip all succeed end-to-end against the
    vendored `picoquicdemo_client` test tool. Found and fixed one real
    bug in the vendored h3zero library along the way (`h3zero_common.c`'s
    H3 POST dispatch never initializes `stream_ctx->path_callback_ctx`
    for a plain `path_table` entry, unlike the legacy HTTP/0.9 path,
    which does — worked around with a global instead of relying on the
    framework-supplied context).
  - **Still blocked**: sending an actual `POST /render` crashes the
    server (SIGSEGV) inside the vendored h3zero framework, before
    `render_path_callback` is ever reached (confirmed via a trace printf
    at the top of the callback that never fires, even with
    `DISABLE_DEBUG_PRINTF` off and stderr explicitly flushed) — the
    path_callback_ctx fix above was real but insufficient; there's at
    least one more bug in the vendored POST/QPACK dispatch path, not yet
    isolated. GET works; POST doesn't.
  - Net: the pipeline-extraction and C ABI work is solid and reusable
    regardless of which server transport eventually works. Two server
    binaries exist (`see-through-server` httplib-based,
    `see-through-server-h3` picoquic-based) and neither yet completes a
    real `/render` round trip — httplib blocked on the 413 bug, h3 blocked
    on the POST crash. Next session: pick one transport and root-cause
    its remaining bug (h3's is narrower — GET already proves the
    handshake/transport/TLS stack all work; the bug is specifically in
    POST/QPACK request dispatch).
