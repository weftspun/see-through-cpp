// Fast, direct correctness check for attn_block's new tiled_naive_attn
// path (src/ops.cpp) at a T large enough to force multiple chunks, since
// the full-pipeline 1280px test regressed even at a previously-clean
// resolution (1216) once attn_block tiling was enabled -- isolating
// whether the bug is in the tiling itself or elsewhere in the pipeline.
#include "ops.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static ggml_backend_t pick_backend() {
    ggml_backend_dev_t d = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (d) { fprintf(stderr, "device: %s\n", ggml_backend_dev_name(d)); return ggml_backend_dev_init(d, nullptr); }
    fprintf(stderr, "device: CPU\n");
    return ggml_backend_cpu_init();
}

int main(int argc, char ** argv) {
    const int W = argc > 1 ? atoi(argv[1]) : 152;   // ZR at res=1216
    const int H = W, C = 128;
    fprintf(stderr, "T = %d\n", W * H);

    Model m;
    m.gn_groups = 32; m.gn_eps = 1e-6f; m.head_dim = 0;
    ggml_init_params ipw = { 16u * 1024 * 1024, nullptr, true };
    ggml_context * ctx_w = ggml_init(ipw);
    m.ctx_w.push_back(ctx_w);

    std::mt19937_64 rng(42);
    std::normal_distribution<float> nrm(0.0f, 0.5f);
    auto mkweight = [&](const std::string & name, std::initializer_list<int64_t> ne) {
        std::vector<int64_t> d(ne);
        ggml_tensor * t = ggml_new_tensor(ctx_w, GGML_TYPE_F32, (int) d.size(), d.data());
        m.weights[name] = t;
        return t;
    };
    mkweight("a.group_norm.weight", { C });
    mkweight("a.group_norm.bias", { C });
    mkweight("a.to_q.weight", { C, C });
    mkweight("a.to_k.weight", { C, C });
    mkweight("a.to_v.weight", { C, C });
    mkweight("a.to_out.0.weight", { C, C });
    mkweight("a.to_out.0.bias", { C });

    ggml_backend_t backend = pick_backend();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx_w, ggml_backend_get_default_buffer_type(backend));
    m.bufs.push_back(buf);
    for (auto & kv : m.weights) {
        size_t n = ggml_nelements(kv.second);
        std::vector<float> vals(n);
        for (float & v : vals) v = nrm(rng);
        ggml_backend_tensor_set(kv.second, vals.data(), 0, n * 4);
    }

    std::vector<float> xin((size_t) W * H * C);
    for (float & v : xin) v = nrm(rng);

    auto run = [&](bool tiled) {
        m.tiled_naive_attn = tiled;
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
        return res;
    };

    auto ref = run(false);
    auto cand = run(true);
    if (ref.empty() || cand.empty() || ref.size() != cand.size()) {
        fprintf(stderr, "execution error: ref=%zu cand=%zu\n", ref.size(), cand.size());
        return 1;
    }
    double worst = 0, amp = 0;
    for (float v : ref) amp = std::max(amp, (double) fabsf(v));
    for (size_t i = 0; i < ref.size(); i++) worst = std::max(worst, (double) fabsf(ref[i] - cand[i]));
    printf("max|ref|=%.6f max_abs_diff=%.6f rel=%.6f\n", amp, worst, worst / (amp + 1e-9));
    return 0;
}
