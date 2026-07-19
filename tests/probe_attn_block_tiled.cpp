// Correctness check for attn_block (ops.cpp) against an independent CPU
// ground truth -- comparing tiled-vs-untiled on the SAME (possibly broken)
// GPU backend only proves they differ, not which one is right. This runs
// both variants on GPU and on CPU (forced backend) with byte-identical
// weights/input and reports each against the CPU result as ground truth.
//   probe_attn_block_tiled <W> <C> <head_dim> <gn_groups>
#include "ops.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static ggml_backend_t make_backend(bool use_cpu) {
    if (!use_cpu) {
        ggml_backend_dev_t d = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (d) { return ggml_backend_dev_init(d, nullptr); }
    }
    return ggml_backend_cpu_init();
}

int main(int argc, char ** argv) {
    const int W = argc > 1 ? atoi(argv[1]) : 152;
    const int H = W;
    const int C = argc > 2 ? atoi(argv[2]) : 128;
    const int head_dim = argc > 3 ? atoi(argv[3]) : 0;
    const int gn_groups = argc > 4 ? atoi(argv[4]) : 32;
    fprintf(stderr, "T = %d, C = %d, head_dim = %d, gn_groups = %d\n", W * H, C, head_dim, gn_groups);

    std::vector<float> xin((size_t) W * H * C);
    {
        std::mt19937_64 rng(42);
        std::normal_distribution<float> nrm(0.0f, 0.5f);
        for (float & v : xin) v = nrm(rng);
    }

    auto run = [&](bool use_cpu, bool tiled) {
        Model m;
        m.gn_groups = gn_groups; m.gn_eps = 1e-5f; m.head_dim = head_dim;
        m.tiled_naive_attn = tiled;
        ggml_init_params ipw = { 16u * 1024 * 1024, nullptr, true };
        ggml_context * ctx_w = ggml_init(ipw);
        m.ctx_w.push_back(ctx_w);

        std::mt19937_64 rng(42);
        std::normal_distribution<float> nrm(0.0f, 0.5f);
        auto mkweight = [&](const std::string & name, std::initializer_list<int64_t> ne) {
            std::vector<int64_t> d(ne);
            ggml_tensor * t = ggml_new_tensor(ctx_w, GGML_TYPE_F32, (int) d.size(), d.data());
            m.weights[name] = t;
        };
        mkweight("a.group_norm.weight", { C });
        mkweight("a.group_norm.bias", { C });
        mkweight("a.to_q.weight", { C, C });
        mkweight("a.to_k.weight", { C, C });
        mkweight("a.to_v.weight", { C, C });
        mkweight("a.to_out.0.weight", { C, C });
        mkweight("a.to_out.0.bias", { C });

        ggml_backend_t backend = make_backend(use_cpu);
        fprintf(stderr, "  [%s tiled=%d] device: %s\n", use_cpu ? "CPU" : "GPU", tiled,
                ggml_backend_name(backend));
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(
            ctx_w, ggml_backend_get_default_buffer_type(backend));
        m.bufs.push_back(buf);
        // same weight values every call: same rng seed, same generation order
        for (const char * n : { "a.group_norm.weight", "a.group_norm.bias", "a.to_q.weight",
                                 "a.to_k.weight", "a.to_v.weight", "a.to_out.0.weight", "a.to_out.0.bias" }) {
            ggml_tensor * t = m.weights[n];
            size_t ne = ggml_nelements(t);
            std::vector<float> vals(ne);
            for (float & v : vals) v = nrm(rng);
            ggml_backend_tensor_set(t, vals.data(), 0, ne * 4);
        }

        size_t meta = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false);
        ggml_init_params ip = { meta, nullptr, true };
        m.ctx_g = ggml_init(ip);
        ggml_tensor * xt = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, W, H, C, 1);
        ggml_set_input(xt);
        ggml_tensor * out = attn_block(m, xt, "a");
        ggml_set_output(out);
        ggml_cgraph * gf = ggml_new_graph_custom(m.ctx_g, 4096, false);
        ggml_build_forward_expand(gf, out);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        std::vector<float> res;
        if (ggml_gallocr_alloc_graph(alloc, gf)) {
            ggml_backend_tensor_set(xt, xin.data(), 0, xin.size() * 4);
            if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
                res.resize(ggml_nelements(out));
                ggml_backend_tensor_get(out, res.data(), 0, res.size() * 4);
            }
        }
        ggml_gallocr_free(alloc);
        ggml_free(m.ctx_g);
        m.ctx_g = nullptr;
        ggml_backend_free(backend);
        return res;
    };

    auto cpu_ref = run(true, false);
    auto gpu_naive = run(false, false);
    auto gpu_tiled = run(false, true);

    auto compare = [&](const char * label, const std::vector<float> & v) {
        if (v.empty() || v.size() != cpu_ref.size()) {
            printf("%-12s execution error (n=%zu vs cpu n=%zu)\n", label, v.size(), cpu_ref.size());
            return;
        }
        double worst = 0, amp = 0;
        for (float x : cpu_ref) { amp = std::max(amp, (double) fabsf(x)); }
        for (size_t i = 0; i < v.size(); i++) { worst = std::max(worst, (double) fabsf(v[i] - cpu_ref[i])); }
        printf("%-12s vs CPU ground truth: max|cpu|=%.6f max_abs_diff=%.6f rel=%.6f\n",
               label, amp, worst, worst / (amp + 1e-9));
    };
    compare("gpu_naive", gpu_naive);
    compare("gpu_tiled", gpu_tiled);
    return 0;
}
