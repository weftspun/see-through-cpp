// Winograd F(2x2,3x3) convolution, built from existing ggml primitive ops
// (im2col, add/sub/scale, concat, out_prod) rather than a custom op/shader --
// every primitive already has a correct, Vulkan-accelerated implementation.
// See src/winograd.cpp's header comment for the derivation and why the small
// G/B/A transforms are unrolled explicitly rather than expressed as matmuls.
#pragma once

#include "ops.h"

// 3x3 stride-1 pad-1 dilation-1 conv + bias, matching conv2d()'s contract for
// that specific case.
//
// Returns nullptr (falls through, caller should use conv2d() instead) when
// the shape doesn't fit the F(2x2,3x3) tiling: weight isn't 3x3, batch != 1,
// or W/H aren't both even.
ggml_tensor * conv2d_winograd_3x3s1(Model & m, ggml_tensor * x, const std::string & pre);
