# ggml findings for upstream triage

Observed against vendored ggml v0.17.0 (9be3133) on Windows (clang/lld,
Ninja), CPU = AMD Ryzen 7 3800X, GPU = NVIDIA RTX 4090 (Vulkan backend).
Each item has a local repro in this repo.

## 1. gallocr recycles INPUT tensor buffers within one compute

`ggml_gallocr_free_node` (ggml-alloc.c) only spares tensors flagged
`GGML_TENSOR_FLAG_OUTPUT`; a `GGML_TENSOR_FLAG_INPUT` tensor's buffer is
freed after its last consumer *inside* the graph and reused for later
intermediates. Consequence: in a loop that reuses one allocated graph
(diffusion sampling), inputs set once before the loop are clobbered during
the first compute — the second compute silently reads garbage (in our case
corrupted cross-attention conditioning; step 0 matched the reference at
~1e-3, step 1+ diverged wildly).

Perhaps intended behavior — but worth either documenting loudly on
`ggml_gallocr_alloc_graph` or having FLAG_INPUT also pin the buffer.

Repro: `tests/test_graph_properties.cpp`, property "graph reuse with re-set
inputs matches fresh graph" (the fix — re-setting all inputs each compute —
is what makes it pass; setting them once reproduces the corruption).

## 2. CPU flash_attn_ext diverges from naive attention for n_head >= 2

With q/k/v in the documented layouts (q/k/v `[d, T, H, B]`, v not
transposed, mask null), CPU `ggml_flash_attn_ext` agrees with a naive
soft_max(QK^T)V graph to f16 precision for a single head, but diverges by
~0.3 max-abs for `n_head >= 2` (e.g. d=64, H=2, Tq=13, Tk=20, B=1). The
Vulkan backend agrees with naive for the same shapes/layouts.

Repro: `tests/test_graph_properties.cpp`, property "attn_tokens: flash ==
naive" — currently gated to GPU-only because of this divergence.

## 3. Vulkan ggml_conv_2d_direct silently returns zeros at large spatial sizes

`ggml_conv_2d_direct` (f32 kernel, k3 s1 p1) matches the im2col+mul_mat
path at 512x512 and below, but at ~1280x1280 the Vulkan backend returns
all-zero output with no error or assert. (The op also differs numerically
from im2col for the stride-2/pad-0 + asymmetric-`ggml_pad` encoder pattern,
magnitude ~3.6 — possibly out-of-contract usage, but it fails silently
rather than asserting.)

Repro: `tests/test_graph_properties.cpp`, fixed regression block
"direct conv at 1280^2 is not zero" (run with a Vulkan build).
