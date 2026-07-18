// Shared scaffolding for the per-milestone validation binaries: reference
// reader ({i32 ndim, i64 dims[ndim], f32 data} records), graph-context setup,
// CPU compute, and max-abs-diff comparison against a reference array.
#pragma once

#include "ops.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// SEETHROUGH_DEVICE=vulkan selects the first GPU backend from the registry
// (falls back to CPU when the build has none)
static ggml_backend_t st_backend_init() {
    // GPU is the primary target; CPU only when forced or unavailable
    const char * dev = getenv("SEETHROUGH_DEVICE");
    if (!(dev && strcmp(dev, "cpu") == 0)) {
        ggml_backend_dev_t d = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (d) {
            fprintf(stderr, "device: %s\n", ggml_backend_dev_name(d));
            return ggml_backend_dev_init(d, nullptr);
        }
    }
    return ggml_backend_cpu_init();
}

// load weights to the right place for the selected device;
// SEETHROUGH_FLASH=1 enables ggml_flash_attn_ext in token attention
static bool st_load(Model & m, const char * path) {
    const char * fa = getenv("SEETHROUGH_FLASH");
    if (fa && fa[0] == '1') m.flash_attn = true;
    const char * ch = getenv("SEETHROUGH_CHUNK");
    if (ch && ch[0] == '1') m.direct_conv = true;
    const char * rw = getenv("SEETHROUGH_ROWCHUNK");
    if (rw && rw[0] == '1') m.conv_row_chunk = true;
    const char * dev = getenv("SEETHROUGH_DEVICE");
    if (!(dev && strcmp(dev, "cpu") == 0)) {
        ggml_backend_dev_t d = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (d) return m.load_backend(path, ggml_backend_dev_buffer_type(d));
    }
    return m.load(path);
}

struct NpyArray {
    std::vector<int64_t> shape;   // numpy order (outermost first)
    std::vector<float>   data;
};

static bool read_ref(const char * path, std::vector<NpyArray> & out, size_t expect) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    for (;;) {
        int32_t ndim = 0;
        if (fread(&ndim, 4, 1, f) != 1) break;
        if (ndim <= 0 || ndim > 8) { fclose(f); return false; }
        NpyArray arr;
        arr.shape.resize(ndim);
        if (fread(arr.shape.data(), 8, ndim, f) != (size_t) ndim) { fclose(f); return false; }
        int64_t n = 1;
        for (int64_t d : arr.shape) n *= d;
        arr.data.resize(n);
        if (fread(arr.data.data(), 4, n, f) != (size_t) n) { fclose(f); return false; }
        out.push_back(std::move(arr));
    }
    fclose(f);
    return out.size() == expect;
}

// allocate a graph metadata context on m.ctx_g
static void init_graph_ctx(Model & m, size_t max_nodes) {
    size_t meta = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    ggml_init_params ip = { meta, nullptr, /*no_alloc*/ true };
    m.ctx_g = ggml_init(ip);
}

// build+alloc+compute all `outs` in one CPU graph; inputs must be set via the
// callback after allocation (gallocr owns the buffers)
template <typename SetInputs>
static bool compute_cpu_multi(Model & m, const std::vector<ggml_tensor *> & outs,
                              size_t max_nodes, SetInputs set_inputs, int n_threads = 8) {
    ggml_backend_t backend = st_backend_init();
    if (ggml_backend_is_cpu(backend)) ggml_backend_cpu_set_n_threads(backend, n_threads);
    ggml_cgraph * gf = ggml_new_graph_custom(m.ctx_g, max_nodes, false);
    for (ggml_tensor * out : outs) ggml_build_forward_expand(gf, out);
    printf("graph: %d nodes\n", ggml_graph_n_nodes(gf));
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) { fprintf(stderr, "alloc failed\n"); return false; }
    set_inputs();
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n"); return false;
    }
    return true;
}

template <typename SetInputs>
static bool compute_cpu(Model & m, ggml_tensor * out, size_t max_nodes, SetInputs set_inputs,
                        int n_threads = 8) {
    return compute_cpu_multi(m, { out }, max_nodes, set_inputs, n_threads);
}

// max-abs-diff gate; returns process exit code
static int compare_ref(ggml_tensor * out, const NpyArray & ref, double threshold = 5e-2) {
    std::vector<float> y(ggml_nelements(out));
    ggml_backend_tensor_get(out, y.data(), 0, y.size() * 4);
    double max_abs = 0, sum_abs = 0;
    size_t max_i = 0;
    for (size_t i = 0; i < y.size(); i++) {
        double d = fabs((double) y[i] - (double) ref.data[i]);
        if (d > max_abs) { max_abs = d; max_i = i; }
        sum_abs += d;
    }
    printf("output %lld elems: max_abs_diff=%.6f mean_abs_diff=%.6f (argmax %zu: %f vs ref %f)\n",
           (long long) y.size(), max_abs, sum_abs / y.size(), max_i,
           (double) y[max_i], (double) ref.data[max_i]);
    printf("%s\n", max_abs < threshold ? "VALIDATION PASS" : "VALIDATION FAIL");
    return max_abs < threshold ? 0 : 1;
}
