// Milestone 3 of see-through.cpp: the full TransparentVAE decode chain as one
// ggml graph — SDXL VAE decode -> pixel*0.5+0.5 -> UNet1024 alpha head ->
// clip(0,1) — validated against upstream TransparentVAEDecoder.forward.
// (Upstream's estimate_augmented breaks after the first flip/rot combination,
// so the "8-way ensemble" is a single identity pass and no augmentation loop
// is needed.)
//
//   test_trans_vae_full <layerdiff-vae.gguf> <trans-vae.gguf> <reference_trans_vae_full.bin>
//
// Both sub-graphs are the ones already validated standalone (test_sd_vae,
// test_trans_vae); this composes them, with GroupNorm(32, eps 1e-6) /
// full-channel-head attention on the VAE side and GroupNorm(4, eps 1e-5) /
// head-dim-8 attention on the UNet1024 side.

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
    return out.size() == 2;
}

// ── model: weights merged from both ggufs (names do not collide) ─────────────
struct Model {
    std::vector<ggml_context *> ctx_w;   // one per gguf
    ggml_context * ctx_g = nullptr;      // graph (no_alloc)
    std::map<std::string, ggml_tensor *> weights;

    // per-subgraph normalization/attention parameters, set before building
    int   gn_groups = 32;
    float gn_eps    = 1e-6f;
    int   head_dim  = 0;      // 0 = one head of dim C

    bool load(const char * path) {
        ggml_context * c = nullptr;
        gguf_init_params gp = { /*no_alloc*/ false, /*ctx*/ &c };
        gguf_context * g = gguf_init_from_file(path, gp);
        if (!g) return false;
        for (ggml_tensor * t = ggml_get_first_tensor(c); t; t = ggml_get_next_tensor(c, t)) {
            weights[ggml_get_name(t)] = t;
        }
        ctx_w.push_back(c);
        return true;
    }
    ggml_tensor * get(const std::string & name) const {
        auto it = weights.find(name);
        if (it == weights.end()) { fprintf(stderr, "missing tensor: %s\n", name.c_str()); exit(1); }
        return it->second;
    }
    bool has(const std::string & name) const {
        return weights.count(name) != 0;
    }
};

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
    x = ggml_group_norm(ctx, x, m.gn_groups, m.gn_eps);
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
    const int64_t hd = m.head_dim > 0 ? m.head_dim : C;
    const int64_t n_head = C / hd;

    ggml_tensor * h = group_norm_affine(m, x, pre + ".group_norm");
    // (W,H,C,1) -> tokens (C,T)
    h = ggml_reshape_2d(ctx, h, T, C);
    h = ggml_cont(ctx, ggml_transpose(ctx, h));                     // (C, T)

    ggml_tensor * q = linear(m, h, pre + ".to_q");                  // (C, T)
    ggml_tensor * k = linear(m, h, pre + ".to_k");
    ggml_tensor * v = linear(m, h, pre + ".to_v");

    q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, q, hd, n_head, T), 0, 2, 1, 3)); // (d,T,H)
    k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, k, hd, n_head, T), 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, hd, n_head, T), 1, 2, 0, 3)); // (T,d,H)

    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);                     // (T,T,H)
    kq = ggml_soft_max(ctx, ggml_scale(ctx, kq, 1.0f / sqrtf((float) hd)));
    ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);                   // (d,T,H)
    kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));       // (d,H,T)
    kqv = ggml_reshape_2d(ctx, kqv, C, T);                          // (C, T)

    ggml_tensor * out = linear(m, kqv, pre + ".to_out.0");          // (C, T)
    // back to spatial and residual
    out = ggml_cont(ctx, ggml_transpose(ctx, out));                 // (T, C)
    out = ggml_reshape_4d(ctx, out, W, H, C, 1);
    return ggml_add(ctx, out, x);
}

// diffusers AutoencoderKL decode (see test_sd_vae.cpp)
static ggml_tensor * vae_decode(Model & m, ggml_tensor * z) {
    ggml_context * ctx = m.ctx_g;
    m.gn_groups = 32; m.gn_eps = 1e-6f; m.head_dim = 0;

    ggml_tensor * sample = conv2d(m, z, "post_quant_conv", 1, 0);   // 1x1
    sample = conv2d(m, sample, "decoder.conv_in");

    sample = resnet_block(m, sample, "decoder.mid_block.resnets.0");
    sample = attn_block(m, sample, "decoder.mid_block.attentions.0");
    sample = resnet_block(m, sample, "decoder.mid_block.resnets.1");

    for (int i = 0; i < 4; i++) {
        char pre[64];
        for (int r = 0; r < 3; r++) {
            snprintf(pre, sizeof(pre), "decoder.up_blocks.%d.resnets.%d", i, r);
            sample = resnet_block(m, sample, pre);
        }
        snprintf(pre, sizeof(pre), "decoder.up_blocks.%d.upsamplers.0.conv", i);
        if (m.has(std::string(pre) + ".weight")) {
            sample = ggml_upscale(ctx, sample, 2, GGML_SCALE_MODE_NEAREST);
            sample = conv2d(m, sample, pre);
        }
    }

    sample = group_norm_affine(m, sample, "decoder.conv_norm_out");
    sample = ggml_silu(ctx, sample);
    return conv2d(m, sample, "decoder.conv_out");
}

// UNet1024 alpha head (see test_trans_vae.cpp); weights under P
static ggml_tensor * unet1024(Model & m, ggml_tensor * x, ggml_tensor * latent) {
    ggml_context * ctx = m.ctx_g;
    m.gn_groups = 4; m.gn_eps = 1e-5f; m.head_dim = 8;
    const std::string P = "decoder.model.";

    ggml_tensor * sample_latent = conv2d(m, latent, P + "latent_conv_in", 1, 0); // 1x1
    ggml_tensor * sample = conv2d(m, x, P + "conv_in");

    std::vector<ggml_tensor *> res_stack = { sample };
    const bool down_attn[7] = { false, false, false, false, true, true, true };
    for (int i = 0; i < 7; i++) {
        char pre[64];
        if (i == 3) sample = ggml_add(ctx, sample, sample_latent);
        for (int r = 0; r < 2; r++) {
            snprintf(pre, sizeof(pre), "down_blocks.%d.resnets.%d", i, r);
            sample = resnet_block(m, sample, P + pre);
            if (down_attn[i]) {
                snprintf(pre, sizeof(pre), "down_blocks.%d.attentions.%d", i, r);
                sample = attn_block(m, sample, P + pre);
            }
            res_stack.push_back(sample);
        }
        snprintf(pre, sizeof(pre), "down_blocks.%d.downsamplers.0.conv", i);
        if (m.has(P + pre + ".weight")) {
            sample = conv2d(m, sample, P + pre, 2, 1);
            res_stack.push_back(sample);
        }
    }

    sample = resnet_block(m, sample, P + "mid_block.resnets.0");
    sample = attn_block(m, sample, P + "mid_block.attentions.0");
    sample = resnet_block(m, sample, P + "mid_block.resnets.1");

    const bool up_attn[7] = { true, true, true, false, false, false, false };
    for (int i = 0; i < 7; i++) {
        char pre[64];
        for (int r = 0; r < 3; r++) {
            ggml_tensor * res = res_stack.back(); res_stack.pop_back();
            sample = ggml_concat(ctx, sample, res, 2);  // channel dim
            snprintf(pre, sizeof(pre), "up_blocks.%d.resnets.%d", i, r);
            sample = resnet_block(m, sample, P + pre);
            if (up_attn[i]) {
                snprintf(pre, sizeof(pre), "up_blocks.%d.attentions.%d", i, r);
                sample = attn_block(m, sample, P + pre);
            }
        }
        snprintf(pre, sizeof(pre), "up_blocks.%d.upsamplers.0.conv", i);
        if (m.has(P + pre + ".weight")) {
            sample = ggml_upscale(ctx, sample, 2, GGML_SCALE_MODE_NEAREST);
            sample = conv2d(m, sample, P + pre);
        }
    }

    sample = group_norm_affine(m, sample, P + "conv_norm_out");
    sample = ggml_silu(ctx, sample);
    return conv2d(m, sample, P + "conv_out");
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s layerdiff-vae.gguf trans-vae.gguf reference.bin\n", argv[0]);
        return 1;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);

    Model m;
    if (!m.load(argv[1]) || !m.load(argv[2])) { fprintf(stderr, "failed to load weights\n"); return 1; }
    printf("weights: %zu tensors (both ggufs)\n", m.weights.size());

    // load reference (z, y in file order)
    std::vector<NpyArray> ref;
    if (!read_ref(argv[3], ref)) { fprintf(stderr, "failed to read %s\n", argv[3]); return 1; }
    const NpyArray & rz = ref[0], & ry = ref[1];
    const int64_t ZRES = rz.shape[3];
    printf("reference: z(1,%lld,%lld,%lld) y(1,%lld,%lld,%lld)\n",
           (long long) rz.shape[1], (long long) rz.shape[2], (long long) rz.shape[3],
           (long long) ry.shape[1], (long long) ry.shape[2], (long long) ry.shape[3]);

    printf("building graph...\n");
    size_t graph_meta = ggml_tensor_overhead() * 32768 + ggml_graph_overhead_custom(32768, false);
    ggml_init_params ip = { graph_meta, nullptr, /*no_alloc*/ true };
    m.ctx_g = ggml_init(ip);
    ggml_context * ctx = m.ctx_g;

    ggml_tensor * z = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ZRES, ZRES, 4, 1);
    ggml_set_input(z);

    // ── forward: TransparentVAEDecoder.forward, single pass ──────────────────
    ggml_tensor * pixel = vae_decode(m, z);
    pixel = ggml_scale_bias(ctx, pixel, 0.5f, 0.5f);        // [-1,1] -> [0,1]
    ggml_tensor * y_out = unet1024(m, pixel, z);
    y_out = ggml_clamp(ctx, y_out, 0.0f, 1.0f);             // estimate_augmented clip
    ggml_set_output(y_out);

    // ── compute ──────────────────────────────────────────────────────────────
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 8);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, false);
    ggml_build_forward_expand(gf, y_out);
    printf("graph: %d nodes\n", ggml_graph_n_nodes(gf));

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) { fprintf(stderr, "alloc failed\n"); return 1; }

    ggml_backend_tensor_set(z, rz.data.data(), 0, rz.data.size() * 4);

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n"); return 1;
    }

    // ── compare ──────────────────────────────────────────────────────────────
    std::vector<float> y(ggml_nelements(y_out));
    ggml_backend_tensor_get(y_out, y.data(), 0, y.size() * 4);

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
