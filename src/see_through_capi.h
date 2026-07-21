// C ABI for the see-through pipeline: image bytes in, layered per-layer PNGs
// out. Distinct from seethrough_capi.h, which is scoped to
// Lean/plausible-witness-dag kernel witness testing (st_witness_check*) --
// this one drives the real production pipeline (run_see_through in
// pipeline.cpp), for embedding in other processes (e.g. a server) without
// going through the CLI's argv/file-I/O surface.
#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(ST2_BUILD_DLL)
#    define ST2_API __declspec(dllexport)
#  else
#    define ST2_API __declspec(dllimport)
#  endif
#else
#  define ST2_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct st_layer {
    char * tag;           // e.g. "topwear"; null-terminated, owned by the result
    uint8_t * png;         // PNG-encoded RGBA bytes, owned by the result
    size_t png_len;
} st_layer;

typedef struct st_render_result {
    st_layer * layers;      // z-ordered back-to-front
    size_t num_layers;
} st_render_result;

// Renders one encoded image (anything stb_image can decode: PNG/JPEG/etc.)
// into z-ordered layers. `device` is "auto" or "vulkan" (both select the first
// GPU; hard error if none found) -- GPU-only, "cpu" is rejected outright.
// Returns 0 on success; nonzero on failure (bad image, unsupported device,
// model load failure, or a pipeline stage returning false) with *out left
// zero-initialized. On success, the caller must release *out via
// st_free_result exactly once.
ST2_API int st_render(const char * model_dir, const uint8_t * image_data, size_t image_len,
                      int steps, int res, int depth_res, uint64_t seed, const char * device,
                      st_render_result * out);

ST2_API void st_free_result(st_render_result * r);

#ifdef __cplusplus
}
#endif
