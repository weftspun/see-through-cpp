// C FFI for driving see-through graph checks from external verifiers —
// designed for Lean/Lake `plausible-witness-dag` (the caller supplies the
// deterministic walk + candidate predicate; this library executes one
// candidate configuration deterministically on the primary device and
// returns a numeric verdict).
//
// Witness space: an op kind + shape + knob bits. The verdict is the
// per-element interval violation of the candidate kernel configuration
// against the reference configuration (<= 1.0 means contained in the
// f16-accumulation error interval; > 1.0 is a defect witness; < 0 is an
// execution error).
#pragma once

#include <stdint.h>

#if defined(_WIN32)
#  if defined(ST_BUILD_DLL)
#    define ST_API __declspec(dllexport)
#  else
#    define ST_API __declspec(dllimport)
#  endif
#else
#  define ST_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct st_case {
    const char * op;      // "conv2d" | "attn" | "linear_quant"
    int32_t w, h, c, oc;  // conv2d: spatial + channels; linear_quant: c=in, oc=out
    int32_t stride;       // conv2d: 1 or 2 (pad fixed at 1, k3)
    int32_t heads, tq, tk, batch;   // attn: head count (dim 64), tokens;
                                    // linear_quant: tq = token count (batch of rows)
    int32_t direct;       // conv2d knob: ggml_conv_2d_direct
    int32_t rowchunk;     // conv2d knob: row-chunked im2col
    int32_t flash;        // attn knob: ggml_flash_attn_ext
    uint64_t seed;        // deterministic weight/input generation
} st_case;

// Executes the candidate (knobs as given) and the reference (knobs off) on
// the primary device and returns the worst per-element interval violation
// (tol = atol + rtol_auto * max|ref|, rtol_auto scaled by reduction length).
ST_API double st_witness_check(const st_case * c);

// primary device name ("Vulkan0 (...)" or "CPU"), static storage
ST_API const char * st_device(void);

// Flat-scalar variant for FFI hosts without struct marshalling (Lean 4
// @[extern]): op 0 = conv2d, 1 = attn, 2 = linear_quant (c=in, oc=out,
// tq=token count; candidate weight is ggml's own Q4_0, reference is f32);
// knobs bit0 = direct, bit1 = rowchunk, bit2 = flash (ignored for op 2).
// Deterministic in all arguments.
ST_API double st_witness_check_flat(uint32_t op, uint32_t w, uint32_t h, uint32_t c,
                             uint32_t oc, uint32_t stride, uint32_t heads,
                             uint32_t tq, uint32_t tk, uint32_t batch,
                             uint32_t knobs, uint64_t seed);

#ifdef __cplusplus
}
#endif
