# 1280px collapse: diagnostics completed so far

* Status: accepted
* Date: 2026-07-18

## Context and Problem Statement

The full 1280px pipeline collapses every per-tag output layer to a thin
crop (~7% of expected height) at `--steps` >= 8. This record captures the
diagnostic work completed while narrowing down the cause; the fix itself
remains open (MADR 0008).

## Completed

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
- [x] Found and fixed a related bug while bisecting: the CLI crashed
      (unhandled `abort()`, hanging MSVC dialog) on any `--res`/
      `--depth-res` not a multiple of the VAE decoders' required stride
      (64 for trans-vae's 6 stages, 8 for marigold-vae's 3) instead of
      validating input. Now rounds up with a printed note, matching the
      pattern of upstream's own (otherwise unused) `validate_resolution()`.
- [x] Ruled out `flash_attn` as a *seed-dependent* divergence: multi-seed
      probing (`tests/probe_flash_bigT.cpp`, 8 seeds x every gated
      shape/batch, Vulkan) found no divergence beyond the existing
      single-seed gate — all comfortably contained (worst violation 0.30
      vs. a 1.0 defect threshold). Rules out "the gate got a lucky seed"
      as an explanation, leaving only the real-checkpoint-weight
      hypothesis untested.
- [x] Identified that the blank-face/missing-ears symptom seen in
      `assets/final_svg_composite.png` and `final30_composite.png` is the
      *same* 1280px collapse, not a separate "content-quality" bug: both
      composites were generated at the CLI's default `--res` (1280), and
      re-running at `--res 512` produces correct face and ears layers. The
      face/ears layers come from the same head-crop second pass
      (`see_through.cpp`'s "head crop + pass 2") as topwear etc., so they
      collapse identically at res=1280 (e.g. a 1280px run's `face` layer
      comes out 853x87, the same thin-band shape as the already-documented
      `topwear` collapse).

## Consequences

The fix itself (the last remaining resolution-scaling UNet knob,
`flash_attn`, is untestable in isolation without OOMing, and multi-seed
synthetic-weight probing found nothing) is tracked as the top item of
MADR 0008.
