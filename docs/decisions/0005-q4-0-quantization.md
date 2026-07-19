# 5. Q4_0 quantization for nn.Linear weights

Status: accepted (2026-07-18)

## Context and Problem Statement

Upstream ships an NF4-quantized pipeline (`inference_psd_quantized.py`,
bitsandbytes `Linear4bit`) for 8GB-class GPUs. We wanted the equivalent
VRAM win without adopting bitsandbytes or reopening the low-VRAM knob
surface (flash attention + direct conv + row-chunked decode, already gated
by `verify/KernelGate.lean`). What quantization format, what scope, and how
is it validated?

## Considered Options

1. Port bitsandbytes' NF4 format bit-for-bit.
2. Use ggml's own native quantized types (Q4_0/Q4_K/…) instead of NF4.
3. Skip quantization; rely on the existing flash/direct-conv/row-chunk
   low-VRAM path alone.

## Decision Outcome

Option 2, specifically **Q4_0** (ggml's simplest 4-bit format: 32-element
blocks, one f16 scale + packed nibbles) applied to **nn.Linear weights
only** — self/cross-attention projections and GEGLU FF, matching upstream's
own scope (bitsandbytes `Linear4bit` doesn't touch conv layers either, and
conv kernels' innermost dim of 3 can't form 32-element blocks anyway; norms
and biases stay f32/f16, they're a rounding error's worth of VRAM).

Why Q4_0 over Q4_K: Q4_K's superblock format (6-bit hierarchical scales)
would need a from-scratch numpy implementation with more surface for bugs;
Q4_0's block layout is simple enough to implement and bit-verify against
ggml's own `quantize_row_q4_0_ref` in one sitting. If accuracy proves
insufficient at full 1280px/30-step scale, upgrading specific tensors to
Q4_K is a contained follow-up, not a redesign.

Why not NF4: it's bitsandbytes' format, requiring either vendoring
bitsandbytes' CUDA-only kernels (defeats the Vulkan-primary goal entirely)
or reimplementing NF4's non-uniform quantization levels from scratch for a
format ggml has no native support for. Q4_0 gets the same order-of-magnitude
VRAM win with a format ggml already runs natively on every backend
(CPU/Vulkan/CUDA), no new kernel code.

### Loading needs zero C++ changes

`Model::load_backend` already uses ggml's own `gguf_init_from_file` (real
GGUF reader, not a hand-rolled one), and `linear()` is a plain
`ggml_mul_mat(w, x)` — ggml's mul_mat handles quantized `src0` natively on
both CPU and Vulkan (this is exactly llama.cpp's use case). The only new
code is in the Python converter (`quantize_q4_0`, verified bit-exact
against the C reference) and a witness-gate case to characterize the
resulting error.

### Validation: a design search, not a fixed-point gate

Reference-gate byte-exactness doesn't apply to a lossy format by
definition, so this isn't a pass/fail regression gate like
`verify/KernelGate.lean`. Instead `verify/QuantDesign.lean` **searches**
whether Q4_0 error stays within an accepted design tolerance (atol=1e-3,
rtol=0.15) across the 12 real nn.Linear shapes `layerdiff-unet`'s SDXL
CrossFrame UNet uses (self/cross-attn at C∈{320,640,1280}, cross-attn K/V
from the 2048-dim text context, GEGLU ff.net.0/net.2) — all 12 landed at
~7.5-10% actual error, comfortably inside tolerance. The `seethrough_c` FFI
gained a `linear_quant` case (op 2) for this: build an f32 reference weight
and a Q4_0 candidate via ggml's own reference quantizer, run both through
`linear()`, and report the interval violation exactly like the kernel gate
does for conv/attention.

### Results

Requantized layerdiff-unet (8.15GB→2.78GB), marigold-unet
(2.43GB→1.50GB), layerdiff-te2 (1.39GB→0.39GB), marigold-te
(0.68GB→0.19GB) — 12.65GB→4.86GB, 61.6% smaller. VAEs and `layerdiff-te1`
are untouched (VAEs are conv-dominated, little to gain; te1 was already
pinned to f32 for accuracy). A 512px/4-step smoke run through a
`models-q4/` weight set produced coherent layers (clear hair strands,
recognizable clothing) — real end-to-end confirmation, not just the
isolated FFI probe.

## Consequences

Not yet run at full 1280px/30-step scale or checked against upstream IoU
parity — tracked in MADR 0007. If a specific shape proves too lossy at full
scale despite passing the design search (design tolerance was chosen from a
synthetic random-data measurement, not derived from the real weight
distributions), the fix is Q4_K for that tensor alone, not abandoning
quantization wholesale.
