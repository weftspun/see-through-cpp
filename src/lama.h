// LaMa inpainting (AnimeMangaInpainting lama_large_512px): FFCResNetGenerator
// 4->3, ngf 64, 3 downsamplings, 18 FFC bottleneck blocks (local 128 /
// global 384, enable_lfu=false, use_mpe=false), BatchNorm folded at graph
// build, 2-D real FFT as a custom CPU op. Weights: lama.gguf ("model.N.*").
#pragma once

#include "ops.h"

// img (W,H,3,1) in [0,1], mask (W,H,1,1) in {0,1} (1 = hole). Returns the
// composited result pred*mask + img*(1-mask), (W,H,3,1) in [0,1].
// W and H must be equal and a multiple of 8 (upstream pads to square/64).
ggml_tensor * lama_inpaint_graph(Model & m, ggml_tensor * img, ggml_tensor * mask);
