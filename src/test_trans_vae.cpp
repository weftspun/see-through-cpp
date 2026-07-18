// Milestone 1 of see-through.cpp: the TransparentVAE decoder head (UNet1024)
// as a ggml graph, validated numerically against a PyTorch reference.
//
//   test_trans_vae <trans-vae.gguf> <reference_trans_vae.npz>
//
// Architecture (see-through common/modules/layerdiffuse/vae.py, UNet1024):
//   conv_in 3->32; 7 down blocks (32,32,64,128,256,512,512), plain x4 then
//   attn x3, 2 resnets each, conv downsample except last; latent (4ch, /8)
//   injected via 1x1 conv before down block 3; attn mid block; 7 up blocks
//   (3 resnets each, skip concat, nearest-2x upsample except last);
//   GroupNorm(4)/SiLU everywhere, attention head dim 8, no time embedding.
// Weights keep their diffusers names under the "decoder.model." prefix.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ── reference reader: per array {i32 ndim, i64 dims[ndim], f32 data} ─────────
struct NpyArray {
    std::vector<int64_t> shape;   // numpy order (outermost first)
    std::vector<float>   data;
};

static bool read_ref(const char * path, std::vector<NpyArray> & out) {
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
    return out.size() == 3;
}

// ── model ────────────────────────────────────────────────────────────────────
struct Model {
    ggml_context * ctx_w = nullptr;   // weights (from gguf)
    ggml_context * ctx_g = nullptr;   // graph (no_alloc)
    std::map<std::string, ggml_tensor *> weights;

    ggml_tensor * get(const std::string & name) const {
        auto it = weights.find("decoder.model." + name);
        if (it == weights.end()) { fprintf(stderr, "missing tensor: decoder.model.%s\n", name.c_str()); exit(1); }
        return it->second;
    }
    bool has(const std::string & name) const {
        return weights.count("decoder.model." + name) != 0;
    }
};

static const int   GN_GROUPS = 4;
static const float GN_EPS    = 1e-5f;
static const int   HEAD_DIM  = 8;

static ggml_tensor * bias4d(ggml_context * ctx, ggml_tensor * b) {
    return ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);  // (1,1,C,1) for broadcast
}

static ggml_tensor * conv2d(Model & m, ggml_tensor * x, const std::string & pre,
                            int stride = 1, int pad = 1) {
    ggml_context * ctx = m.ctx_g;
    x = ggml_conv_2d(ctx, m.get(pre + ".weight"), x, stride, stride, pad, pad, 1, 1);
    return ggml_add(ctx, x, bias4d(ctx, m.get(pre + ".bias")));
}

static ggml_tensor * group_norm_affine(Model & m, ggml_tensor * x, const std::string & pre) {
    ggml_context * ctx = m.ctx_g;
    x = ggml_group_norm(ctx, x, GN_GROUPS, GN_EPS);
    x = ggml_mul(ctx, x, bias4d(ctx, m.get(pre + ".weight")));
    return ggml_add(ctx, x, bias4d(ctx, m.get(pre + ".bias")));
}

// diffusers ResnetBlock2D with temb=None, output_scale_factor=1
static ggml_tensor * resnet_block(Model & m, ggml_tensor * x, const std::string & pre) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * h = group_norm_affine(m, x, pre + ".norm1");
    h = ggml_silu(ctx, h);
    h = conv2d(m, h, pre + ".conv1");
    h = group_norm_affine(m, h, pre + ".norm2");
    h = ggml_silu(ctx, h);
    h = conv2d(m, h, pre + ".conv2");
    ggml_tensor * sc = x;
    if (m.has(pre + ".conv_shortcut.weight")) {
        sc = conv2d(m, x, pre + ".conv_shortcut", 1, 0);  // 1x1
    }
    return ggml_add(ctx, h, sc);
}

static ggml_tensor * linear(Model & m, ggml_tensor * x /*(C_in, T)*/, const std::string & pre) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * y = ggml_mul_mat(ctx, m.get(pre + ".weight"), x);  // (C_out, T)
    return ggml_add(ctx, y, m.get(pre + ".bias"));                    // (C_out) bcast over T
}

// diffusers Attention: spatial self-attention, residual_connection=true
static ggml_tensor * attn_block(Model & m, ggml_tensor * x, const std::string & pre) {
    ggml_context * ctx = m.ctx_g;
    const int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2];
    const int64_t T = W * H;
    const int64_t n_head = C / HEAD_DIM;

    ggml_tensor * h = group_norm_affine(m, x, pre + ".group_norm");
    // (W,H,C,1) -> tokens (C,T)
    h = ggml_reshape_2d(ctx, h, T, C);
    h = ggml_cont(ctx, ggml_transpose(ctx, h));                     // (C, T)

    ggml_tensor * q = linear(m, h, pre + ".to_q");                  // (C, T)
    ggml_tensor * k = linear(m, h, pre + ".to_k");
    ggml_tensor * v = linear(m, h, pre + ".to_v");

    q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, q, HEAD_DIM, n_head, T), 0, 2, 1, 3)); // (d,T,H)
    k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, k, HEAD_DIM, n_head, T), 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, HEAD_DIM, n_head, T), 1, 2, 0, 3)); // (T,d,H)

    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);                     // (T,T,H)
    kq = ggml_soft_max(ctx, ggml_scale(ctx, kq, 1.0f / sqrtf((float) HEAD_DIM)));
    ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);                   // (d,T,H)
    kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));       // (d,H,T)
    kqv = ggml_reshape_2d(ctx, kqv, C, T);                          // (C, T)

    ggml_tensor * out = linear(m, kqv, pre + ".to_out.0");          // (C, T)
    // back to spatial and residual
    out = ggml_cont(ctx, ggml_transpose(ctx, out));                 // (T, C)
    out = ggml_reshape_4d(ctx, out, W, H, C, 1);
    return ggml_add(ctx, out, x);
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s trans-vae.gguf reference.npz\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);

    // load weights
    Model m;
    gguf_init_params gp = { /*no_alloc*/ false, /*ctx*/ &m.ctx_w };
    gguf_context * g = gguf_init_from_file(argv[1], gp);
    if (!g) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
    for (ggml_tensor * t = ggml_get_first_tensor(m.ctx_w); t; t = ggml_get_next_tensor(m.ctx_w, t)) {
        m.weights[ggml_get_name(t)] = t;
    }
    printf("weights: %zu tensors\n", m.weights.size());

    // load reference (x, latent, y in file order)
    std::vector<NpyArray> ref;
    if (!read_ref(argv[2], ref)) { fprintf(stderr, "failed to read %s\n", argv[2]); return 1; }
    const NpyArray & rx = ref[0], & rl = ref[1], & ry = ref[2];
    const int64_t RES = rx.shape[3], IN_CH = rx.shape[1];
    printf("reference: x(1,%lld,%lld,%lld) latent(1,%lld,%lld,%lld) y(1,%lld,%lld,%lld)\n",
           (long long) rx.shape[1], (long long) rx.shape[2], (long long) rx.shape[3],
           (long long) rl.shape[1], (long long) rl.shape[2], (long long) rl.shape[3],
           (long long) ry.shape[1], (long long) ry.shape[2], (long long) ry.shape[3]);

    printf("building graph...\n");
    // graph context (metadata only)
    size_t graph_meta = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false);
    ggml_init_params ip = { graph_meta, nullptr, /*no_alloc*/ true };
    m.ctx_g = ggml_init(ip);
    ggml_context * ctx = m.ctx_g;

    ggml_tensor * x      = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, RES, RES, IN_CH, 1);
    ggml_tensor * latent = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, RES / 8, RES / 8, 4, 1);
    ggml_set_input(x);
    ggml_set_input(latent);

    // ── forward ──────────────────────────────────────────────────────────────
    ggml_tensor * sample_latent = conv2d(m, latent, "latent_conv_in", 1, 0); // 1x1
    ggml_tensor * sample = conv2d(m, x, "conv_in");

    std::vector<ggml_tensor *> res_stack = { sample };
    const bool down_attn[7] = { false, false, false, false, true, true, true };
    for (int i = 0; i < 7; i++) {
        char pre[64];
        if (i == 3) sample = ggml_add(ctx, sample, sample_latent);
        for (int r = 0; r < 2; r++) {
            snprintf(pre, sizeof(pre), "down_blocks.%d.resnets.%d", i, r);
            sample = resnet_block(m, sample, pre);
            if (down_attn[i]) {
                snprintf(pre, sizeof(pre), "down_blocks.%d.attentions.%d", i, r);
                sample = attn_block(m, sample, pre);
            }
            res_stack.push_back(sample);
        }
        snprintf(pre, sizeof(pre), "down_blocks.%d.downsamplers.0.conv", i);
        if (m.has(std::string(pre) + ".weight")) {
            sample = conv2d(m, sample, pre, 2, 1);
            res_stack.push_back(sample);
        }
    }

    sample = resnet_block(m, sample, "mid_block.resnets.0");
    sample = attn_block(m, sample, "mid_block.attentions.0");
    sample = resnet_block(m, sample, "mid_block.resnets.1");

    const bool up_attn[7] = { true, true, true, false, false, false, false };
    for (int i = 0; i < 7; i++) {
        char pre[64];
        for (int r = 0; r < 3; r++) {
            ggml_tensor * res = res_stack.back(); res_stack.pop_back();
            sample = ggml_concat(ctx, sample, res, 2);  // channel dim
            snprintf(pre, sizeof(pre), "up_blocks.%d.resnets.%d", i, r);
            sample = resnet_block(m, sample, pre);
            if (up_attn[i]) {
                snprintf(pre, sizeof(pre), "up_blocks.%d.attentions.%d", i, r);
                sample = attn_block(m, sample, pre);
            }
        }
        snprintf(pre, sizeof(pre), "up_blocks.%d.upsamplers.0.conv", i);
        if (m.has(std::string(pre) + ".weight")) {
            sample = ggml_upscale(ctx, sample, 2, GGML_SCALE_MODE_NEAREST);
            sample = conv2d(m, sample, pre);
        }
    }

    sample = group_norm_affine(m, sample, "conv_norm_out");
    sample = ggml_silu(ctx, sample);
    sample = conv2d(m, sample, "conv_out");
    ggml_set_output(sample);

    // ── compute ──────────────────────────────────────────────────────────────
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 8);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(gf, sample);
    printf("graph: %d nodes\n", ggml_graph_n_nodes(gf));

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) { fprintf(stderr, "alloc failed\n"); return 1; }

    ggml_backend_tensor_set(x,      rx.data.data(), 0, rx.data.size() * 4);
    ggml_backend_tensor_set(latent, rl.data.data(), 0, rl.data.size() * 4);

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n"); return 1;
    }

    // ── compare ──────────────────────────────────────────────────────────────
    std::vector<float> y(ggml_nelements(sample));
    ggml_backend_tensor_get(sample, y.data(), 0, y.size() * 4);

    double max_abs = 0, sum_abs = 0;
    for (size_t i = 0; i < y.size(); i++) {
        double d = fabs((double) y[i] - (double) ry.data[i]);
        if (d > max_abs) max_abs = d;
        sum_abs += d;
    }
    printf("output %lld elems: max_abs_diff=%.6f mean_abs_diff=%.6f\n",
           (long long) y.size(), max_abs, sum_abs / y.size());
    printf("%s\n", max_abs < 5e-2 ? "VALIDATION PASS" : "VALIDATION FAIL");
    return max_abs < 5e-2 ? 0 : 1;
}
