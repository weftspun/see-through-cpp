#include "seethrough_capi.h"

#include "ops.h"
#include "unet_frame.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-common.h"   // block_q4_0 layout (private ggml header)
#include "ggml-quants.h"   // quantize_row_q4_0_ref (private ggml header)

#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// deterministic execution of one op configuration on the primary device
// ---------------------------------------------------------------------------

static ggml_backend_t capi_backend() {
    ggml_backend_dev_t d = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (d) return ggml_backend_dev_init(d, nullptr);
    return ggml_backend_cpu_init();
}

const char * st_device(void) {
    static std::string name;
    if (name.empty()) {
        ggml_backend_dev_t d = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        name = d ? ggml_backend_dev_name(d) : "CPU";
    }
    return name.c_str();
}

namespace {

struct Fixture {
    Model m;
    ggml_context * ctx_w = nullptr;
    std::mt19937_64 rng;
    std::vector<std::pair<ggml_tensor *, std::vector<float>>> pending;
    bool committed = false;

    explicit Fixture(uint64_t seed) : rng(seed) {
        ggml_init_params ip = { 16u * 1024 * 1024, nullptr, /*no_alloc*/ true };
        ctx_w = ggml_init(ip);
        m.ctx_w.push_back(ctx_w);
    }
    ggml_tensor * weight(const std::string & name, std::initializer_list<int64_t> ne) {
        std::vector<int64_t> d(ne);
        ggml_tensor * t = ggml_new_tensor(ctx_w, GGML_TYPE_F32, (int) d.size(), d.data());
        std::normal_distribution<float> n(0.0f, 0.5f);
        std::vector<float> vals(ggml_nelements(t));
        for (float & x : vals) x = n(rng);
        pending.emplace_back(t, std::move(vals));
        m.weights[name] = t;
        return t;
    }
    void commit() {
        if (committed) return;
        ggml_backend_t probe = capi_backend();
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(
            ctx_w, ggml_backend_get_default_buffer_type(probe));
        m.bufs.push_back(buf);
        for (auto & pv : pending) {
            ggml_backend_tensor_set(pv.first, pv.second.data(), 0, pv.second.size() * 4);
        }
        pending.clear();
        ggml_backend_free(probe);
        committed = true;
    }
    std::vector<float> randvec(size_t n) {
        std::normal_distribution<float> d(0.0f, 1.0f);
        std::vector<float> v(n);
        for (float & x : v) x = d(rng);
        return v;
    }
};

template <typename Build, typename SetInputs>
std::vector<float> run1(Fixture & fx, Build build, SetInputs set_inputs) {
    fx.commit();
    Model & m = fx.m;
    const size_t max_nodes = 4096;
    size_t meta = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    ggml_init_params ip = { meta, nullptr, true };
    m.ctx_g = ggml_init(ip);
    ggml_tensor * out = build();
    ggml_set_output(out);
    ggml_backend_t backend = capi_backend();
    ggml_cgraph * gf = ggml_new_graph_custom(m.ctx_g, max_nodes, false);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    std::vector<float> res;
    if (ggml_gallocr_alloc_graph(alloc, gf)) {
        set_inputs();
        if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
            res.resize(ggml_nelements(out));
            ggml_backend_tensor_get(out, res.data(), 0, res.size() * 4);
        }
    }
    ggml_gallocr_free(alloc);
    ggml_backend_free(backend);
    ggml_free(m.ctx_g);
    m.ctx_g = nullptr;
    return res;
}

// interval containment with the f16-accumulation error model (see
// tests/test_graph_properties history): tol = atol + rtol * max|ref|
double interval_violation(const std::vector<float> & a, const std::vector<float> & b,
                          double atol, double rtol) {
    if (a.empty() || a.size() != b.size()) return -1.0;
    double amp = 0;
    for (float v : b) amp = std::max(amp, fabs((double) v));
    const double tol = atol + rtol * amp;
    double worst = 0;
    for (size_t i = 0; i < a.size(); i++) {
        worst = std::max(worst, fabs((double) a[i] - (double) b[i]) / tol);
    }
    return worst;
}

std::vector<float> conv_run(Fixture & fx, const st_case & c, bool direct, bool rowchunk,
                            const std::vector<float> & x) {
    fx.m.direct_conv = direct;
    fx.m.conv_row_chunk = rowchunk;
    // production floor is 40*40 for the UNet's latent-space convs (see
    // pipeline.cpp's pipe_load); the default 256*256 only covers decode's
    // pixel-space sizes, so a batch>1 UNet-shape witness needs this lowered
    fx.m.conv_row_chunk_min_hw = 40 * 40;
    const int32_t batch = c.batch > 0 ? c.batch : 1;
    ggml_tensor * xt = nullptr;
    auto r = run1(fx,
        [&]() {
            xt = ggml_new_tensor_4d(fx.m.ctx_g, GGML_TYPE_F32, c.w, c.h, c.c, batch);
            ggml_set_input(xt);
            return conv2d(fx.m, xt, "w", c.stride, 1);
        },
        [&]() { ggml_backend_tensor_set(xt, x.data(), 0, x.size() * 4); });
    fx.m.direct_conv = false;
    fx.m.conv_row_chunk = false;
    return r;
}

std::vector<float> attn_run(Fixture & fx, const st_case & c, bool flash, bool tiled,
                            const std::vector<float> & q, const std::vector<float> & kv) {
    fx.m.flash_attn = flash;
    fx.m.tiled_naive_attn = tiled;
    const int C = 64 * c.heads;
    ggml_tensor * qt = nullptr, * kt = nullptr;
    auto r = run1(fx,
        [&]() {
            qt = ggml_new_tensor_3d(fx.m.ctx_g, GGML_TYPE_F32, C, c.tq, c.batch);
            kt = ggml_new_tensor_3d(fx.m.ctx_g, GGML_TYPE_F32, C, c.tk, c.batch);
            ggml_set_input(qt);
            ggml_set_input(kt);
            return attn_tokens(fx.m, qt, kt, "a", c.heads);
        },
        [&]() {
            ggml_backend_tensor_set(qt, q.data(), 0, q.size() * 4);
            ggml_backend_tensor_set(kt, kv.data(), 0, kv.size() * 4);
        });
    fx.m.flash_attn = false;
    fx.m.tiled_naive_attn = false;
    return r;
}

// linear() with either the f32 reference weight or a Q4_0 candidate,
// swapped in the Fixture's weight map (both point at the same in/out shape)
std::vector<float> linear_run(Fixture & fx, const st_case & c, bool quantized,
                              ggml_tensor * f32w, ggml_tensor * qw,
                              const std::vector<float> & x) {
    fx.m.weights["w.weight"] = quantized ? qw : f32w;
    ggml_tensor * xt = nullptr;
    auto r = run1(fx,
        [&]() {
            xt = ggml_new_tensor_2d(fx.m.ctx_g, GGML_TYPE_F32, f32w->ne[0], c.tq);
            ggml_set_input(xt);
            return linear(fx.m, xt, "w");
        },
        [&]() { ggml_backend_tensor_set(xt, x.data(), 0, x.size() * 4); });
    return r;
}

} // namespace

double st_witness_check_flat(uint32_t op, uint32_t w, uint32_t h, uint32_t c,
                             uint32_t oc, uint32_t stride, uint32_t heads,
                             uint32_t tq, uint32_t tk, uint32_t batch,
                             uint32_t knobs, uint64_t seed) {
    st_case sc = {};
    sc.op = op == 0 ? "conv2d" : (op == 1 ? "attn" : "linear_quant");
    sc.w = (int32_t) w; sc.h = (int32_t) h; sc.c = (int32_t) c; sc.oc = (int32_t) oc;
    sc.stride = (int32_t) stride;
    sc.heads = (int32_t) heads; sc.tq = (int32_t) tq; sc.tk = (int32_t) tk;
    sc.batch = (int32_t) batch;
    sc.direct = (knobs & 1) != 0;
    sc.rowchunk = (knobs & 2) != 0;
    sc.flash = (knobs & 4) != 0;
    sc.tiled = (knobs & 8) != 0;
    sc.seed = seed;
    return st_witness_check(&sc);
}

double st_witness_check(const st_case * c) {
    if (!c || !c->op) return -1.0;
    if (strcmp(c->op, "conv2d") == 0) {
        if (c->w < 3 || c->h < 3 || c->c < 1 || c->oc < 1 ||
            (c->stride != 1 && c->stride != 2)) return -1.0;
        Fixture fx(c->seed);
        fx.weight("w.weight", { 3, 3, c->c, c->oc });
        fx.weight("w.bias", { c->oc });
        const int32_t batch = c->batch > 0 ? c->batch : 1;
        std::vector<float> x = fx.randvec((size_t) c->w * c->h * c->c * batch);
        auto ref = conv_run(fx, *c, false, false, x);
        auto cand = conv_run(fx, *c, c->direct != 0, c->rowchunk != 0, x);
        // budget scales with the 9*C-term reduction
        return interval_violation(cand, ref, 1e-3, 2.0 * sqrt(9.0 * c->c) / 1024.0);
    }
    if (strcmp(c->op, "attn") == 0) {
        if (c->heads < 1 || c->tq < 1 || c->tk < 1 || c->batch < 1) return -1.0;
        Fixture fx(c->seed);
        const int C = 64 * c->heads;
        for (const char * n : { "a.to_q", "a.to_k", "a.to_v", "a.to_out.0" }) {
            fx.weight(std::string(n) + ".weight", { C, C });
        }
        fx.weight("a.to_out.0.bias", { C });
        std::vector<float> q = fx.randvec((size_t) C * c->tq * c->batch);
        std::vector<float> kv = fx.randvec((size_t) C * c->tk * c->batch);
        auto ref = attn_run(fx, *c, false, false, q, kv);
        auto cand = attn_run(fx, *c, c->flash != 0, c->tiled != 0, q, kv);
        return interval_violation(cand, ref, 1e-3, 8.0 / 1024.0);
    }
    if (strcmp(c->op, "linear_quant") == 0) {
        // Q4_0 blocks are 32 contiguous elements of the input (row/in_features)
        // dimension — the same constraint the GGUF converter enforces
        if (c->c < 32 || c->c % 32 != 0 || c->oc < 1 || c->tq < 1) return -1.0;
        Fixture fx(c->seed);
        fx.weight("w.weight", { c->c, c->oc });
        fx.weight("w.bias", { c->oc });
        fx.commit();

        ggml_tensor * f32w = fx.m.weights["w.weight"];
        std::vector<float> wvals(ggml_nelements(f32w));
        ggml_backend_tensor_get(f32w, wvals.data(), 0, wvals.size() * sizeof(float));

        ggml_init_params ipq = { ggml_tensor_overhead() + 1024, nullptr, true };
        ggml_context * ctx_q = ggml_init(ipq);
        ggml_tensor * qw = ggml_new_tensor_2d(ctx_q, GGML_TYPE_Q4_0, f32w->ne[0], f32w->ne[1]);
        ggml_backend_t probe = capi_backend();
        ggml_backend_buffer_t qbuf = ggml_backend_alloc_ctx_tensors_from_buft(
            ctx_q, ggml_backend_get_default_buffer_type(probe));
        ggml_backend_free(probe);
        fx.m.bufs.push_back(qbuf);
        fx.m.ctx_w.push_back(ctx_q);

        std::vector<uint8_t> qraw(ggml_nbytes(qw));
        quantize_row_q4_0_ref(wvals.data(), reinterpret_cast<block_q4_0 *>(qraw.data()),
                              (int64_t) wvals.size());
        ggml_backend_tensor_set(qw, qraw.data(), 0, qraw.size());

        std::vector<float> x = fx.randvec((size_t) c->c * c->tq);
        auto ref = linear_run(fx, *c, false, f32w, qw, x);
        auto cand = linear_run(fx, *c, true, f32w, qw, x);
        // Q4_0's per-block max element error is amax_block/16 (half the
        // quant step d=amax/8) — an inherent ~6.25% relative-error floor,
        // not a rounding artifact that shrinks with the reduction length
        // (signal and quantization noise both scale with sqrt(in_features)
        // for random weights, so relative error stays roughly flat). 0.15
        // gives headroom above the ~0.10 measured on a synthetic probe
        // while still catching shapes that degrade further.
        return interval_violation(cand, ref, 1e-3, 0.15);
    }
    return -1.0;
}
