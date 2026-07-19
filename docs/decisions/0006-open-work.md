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

- [x] Diagnosed the 1280px layer collapse: bisected by resolution across
      every valid multiple of 64 (512 through 1216 all clean; only 1280
      collapses) — pins it to something specific about exactly the UNet's
      160x160 latent level with *real* trained weights (the Lean witness
      gate only validated that shape with synthetic random weights). Ruled
      out gallocr input-recycling (all inputs already re-set every
      compute) and scheduler numerics (per-step lat mean/std decay
      smoothly, no NaN/blowup). See docs/ggml-upstream-issues.md item 4.
- [x] Tried and ruled out `direct_conv` as the 1280px collapse's cause:
      extended row-chunked im2col to be batch>1-generic and routed it
      ahead of `direct_conv` for the UNet's stride-1 convs at all 3
      latent levels (real engineering, not a flag flip — the old
      row-chunk code hardcoded batch=1). New Lean witness cases
      (`unet-rowchunk-*-b13`) confirm the generalization is itself exact
      at the real batch=13 shape — but a full 1280px run with the fix in
      place shows the *same* collapse. Kept the code (independently
      correct, removes a latent-shape blind spot) but it isn't the fix.
- [ ] Fix the 1280px collapse: `direct_conv` is excluded; `flash_attn` is
      the only remaining resolution-scaling UNet knob, but disabling it
      to test in isolation OOMs the same way `direct_conv` did (~21.3GB
      single alloc, no VRAM headroom for naive attention at production
      batch/token counts). Paused pending either a Lean witness case
      backed by the real loaded checkpoint (not synthetic weights) or a
      VRAM-safe composed replacement for naive attention at this scale —
      see docs/ggml-upstream-issues.md item 4's "Status: paused" note.
- [x] Found and fixed a related bug while bisecting: the CLI crashed
      (unhandled `abort()`, hanging MSVC dialog) on any `--res`/
      `--depth-res` not a multiple of the VAE decoders' required stride
      (64 for trans-vae's 6 stages, 8 for marigold-vae's 3) instead of
      validating input. Now rounds up with a printed note, matching the
      pattern of upstream's own (otherwise unused) `validate_resolution()`.
- [ ] Layer-quality polish vs upstream reference: L/R-split slivers at the
      pad boundary, faint head-pass alphas, alpha floor tuning
- [ ] Upstream parity: match our SVG's per-tag layers against upstream's
      output by tag name and compare alpha masks (>0.98 IoU where tags
      match) + depth ordering. Decided against comparing against upstream's
      PSD (no PSD reader in our C++, and we don't want a Python dev-tool
      dependency for this) — needs a C++ tool using ThorVG to parse SVG,
      and a non-PSD upstream reference to compare against (open: what that
      reference format is, since upstream only emits PSD).
- [ ] Full-quality run with the Q4_0-quantized models (currently only
      smoke-tested at 512px/4 steps) — same 1280px/30-step + upstream
      parity bar as the f16 baseline, see MADR 0005

### Documentation / upstream

- [ ] Post-MVP (original plan): C ABI + server, trellis2cpp-style
