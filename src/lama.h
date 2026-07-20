// LaMa inpainting (AnimeMangaInpainting lama_large_512px): FFCResNetGenerator
// 4->3, ngf 64, 3 downsamplings, 18 FFC bottleneck blocks (local 128 /
// global 384, enable_lfu=false, use_mpe=false), BatchNorm folded at graph
// build. Two upstream pieces ggml has no Vulkan kernel for at all --
// GGML_OP_CUSTOM (the 2-D FFT) and GGML_OP_PAD_REFLECT_1D (reflect padding,
// used by every conv here) -- are both replaced with compositions of
// standard, Vulkan-supported ops (mul_mat-based DFT; get_rows-based
// reflect-index gather) so the whole graph runs on GPU. Weights: lama.gguf
// ("model.N.*").
#pragma once

#include "ops.h"

#include <cstdint>
#include <vector>

// one-time fixup after loading lama.gguf (any backend): the FourierUnit's
// conv_layer.weight was trained expecting real/imag channels interleaved
// (2c=real(c), 2c+1=imag(c)) on both its input and output axes; permutes
// both axes (plus the correspondingly-indexed .fu.bn.*/conv_layer.bias) to
// the block layout (real block then imag block) this file's matmul-based
// rfft2/irfft2 use instead. Idempotent-unsafe: call exactly once per load.
void lama_prepare_gpu_weights(Model & m);

// auxiliary graph-input tensors lama_inpaint_graph() creates -- opaque to
// the caller beyond passing this struct to lama_set_extra_inputs() after
// graph allocation, the same way img/mask are set.
struct LamaExtraInputs {
    // DFT basis (see lama.cpp for the exact math): Ar/Ai (W,Wh) forward row,
    // Br/Bi (H,H) forward/inverse col, Dr/Di (Wh,W) weighted inverse row --
    // W,H here are the FFT's own axis (the global branch's spatial size,
    // img's W/8), not img's own W/H.
    ggml_tensor * fft_ar, * fft_ai, * fft_br, * fft_bi, * fft_dr, * fft_di;

    // reflect-pad index gathers: ggml_get_rows requires the index tensor's
    // ne[1]/ne[2] to exactly match the gathered tensor's channel/batch dims
    // (no broadcast), so one is needed per distinct (length, pad, channels)
    // combination actually used in the graph -- built lazily during
    // lama_inpaint_graph and cached by that key so repeated call sites at
    // the same shape reuse one tensor.
    struct PadIdx { ggml_tensor * t; int64_t len; int pad; int64_t channels; };
    std::vector<PadIdx> pad_idxs;
};

// img (W,H,3,1) in [0,1], mask (W,H,1,1) in {0,1} (1 = hole). Returns the
// composited result pred*mask + img*(1-mask), (W,H,3,1) in [0,1].
// W and H must be equal and a multiple of 64 (3 stride-2 downsamples ahead
// of the FFT, so the global branch's spatial size -- where the FFT runs --
// is W/8 and must itself be even for the real-FFT Nyquist-bin handling).
ggml_tensor * lama_inpaint_graph(Model & m, ggml_tensor * img, ggml_tensor * mask, LamaExtraInputs * extra);

// fills every tensor in `extra` with its CPU-computed data. Call after
// ggml_gallocr_alloc_graph succeeds, same as any other graph input.
void lama_set_extra_inputs(const LamaExtraInputs & extra, int64_t img_W);
