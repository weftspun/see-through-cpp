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
#include <vector>

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

// build+alloc+compute `out` on CPU; inputs must be set via the callback after
// allocation (gallocr owns the buffers)
template <typename SetInputs>
static bool compute_cpu(Model & m, ggml_tensor * out, size_t max_nodes, SetInputs set_inputs,
                        int n_threads = 8) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    ggml_cgraph * gf = ggml_new_graph_custom(m.ctx_g, max_nodes, false);
    ggml_build_forward_expand(gf, out);
    printf("graph: %d nodes\n", ggml_graph_n_nodes(gf));
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) { fprintf(stderr, "alloc failed\n"); return false; }
    set_inputs();
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n"); return false;
    }
    return true;
}

// max-abs-diff gate; returns process exit code
static int compare_ref(ggml_tensor * out, const NpyArray & ref, double threshold = 5e-2) {
    std::vector<float> y(ggml_nelements(out));
    ggml_backend_tensor_get(out, y.data(), 0, y.size() * 4);
    double max_abs = 0, sum_abs = 0;
    for (size_t i = 0; i < y.size(); i++) {
        double d = fabs((double) y[i] - (double) ref.data[i]);
        if (d > max_abs) max_abs = d;
        sum_abs += d;
    }
    printf("output %lld elems: max_abs_diff=%.6f mean_abs_diff=%.6f\n",
           (long long) y.size(), max_abs, sum_abs / y.size());
    printf("%s\n", max_abs < threshold ? "VALIDATION PASS" : "VALIDATION FAIL");
    return max_abs < threshold ? 0 : 1;
}
