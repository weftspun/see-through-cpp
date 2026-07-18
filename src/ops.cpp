#include "ops.h"

#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

bool Model::load(const char * path) {
    ggml_context * c = nullptr;
    gguf_init_params gp = { /*no_alloc*/ false, /*ctx*/ &c };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) return false;
    for (int64_t i = 0; i < gguf_get_n_kv(g); i++) {
        const char * key = gguf_get_key(g, i);
        if (gguf_get_kv_type(g, i) == GGUF_TYPE_STRING) {
            std::string k = key;
            for (const char * suf : { ".config_json", ".vocab_json", ".merges_txt" }) {
                size_t n = strlen(suf);
                if (k.size() > n && k.compare(k.size() - n, n, suf) == 0) {
                    config_json[k] = gguf_get_val_str(g, i);
                }
            }
        }
    }
    for (ggml_tensor * t = ggml_get_first_tensor(c); t; t = ggml_get_next_tensor(c, t)) {
        weights[ggml_get_name(t)] = t;
    }
    ctx_w.push_back(c);
    gguf_free(g);
    return true;
}

ggml_tensor * Model::get(const std::string & name) const {
    auto it = weights.find(name);
    if (it == weights.end()) { fprintf(stderr, "missing tensor: %s\n", name.c_str()); exit(1); }
    return it->second;
}

bool Model::has(const std::string & name) const {
    return weights.count(name) != 0;
}

ggml_tensor * bias4d(ggml_context * ctx, ggml_tensor * b) {
    return ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
}

ggml_tensor * conv2d(Model & m, ggml_tensor * x, const std::string & pre, int stride, int pad) {
    ggml_context * ctx = m.ctx_g;
    // ggml_conv_2d unconditionally rounds activations to f16 in im2col; for
    // f32 weights (SDXL VAE encoder has activations too large for that) keep
    // the whole conv in f32 instead
    ggml_tensor * w = m.get(pre + ".weight");
    ggml_type it = w->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_tensor * im = ggml_im2col(ctx, w, x, stride, stride, pad, pad, 1, 1, true, it);
    ggml_tensor * r = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im, im->ne[0], im->ne[3] * im->ne[2] * im->ne[1]),
        ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1] * w->ne[2], w->ne[3]));
    r = ggml_reshape_4d(ctx, r, im->ne[1], im->ne[2], im->ne[3], w->ne[3]);
    r = ggml_cont(ctx, ggml_permute(ctx, r, 0, 1, 3, 2));
    return ggml_add(ctx, r, bias4d(ctx, m.get(pre + ".bias")));
}

ggml_tensor * group_norm_affine(Model & m, ggml_tensor * x, const std::string & pre) {
    ggml_context * ctx = m.ctx_g;
    x = ggml_group_norm(ctx, x, m.gn_groups, m.gn_eps);
    x = ggml_mul(ctx, x, bias4d(ctx, m.get(pre + ".weight")));
    return ggml_add(ctx, x, bias4d(ctx, m.get(pre + ".bias")));
}

ggml_tensor * layer_norm_affine(Model & m, ggml_tensor * x, const std::string & pre, float eps) {
    ggml_context * ctx = m.ctx_g;
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, m.get(pre + ".weight"));
    return ggml_add(ctx, x, m.get(pre + ".bias"));
}

ggml_tensor * linear(Model & m, ggml_tensor * x, const std::string & pre) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * y = ggml_mul_mat(ctx, m.get(pre + ".weight"), x);
    return ggml_add(ctx, y, m.get(pre + ".bias"));
}

ggml_tensor * resnet_block(Model & m, ggml_tensor * x, const std::string & pre) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * h = group_norm_affine(m, x, pre + ".norm1");
    h = ggml_silu(ctx, h);
    h = conv2d(m, h, pre + ".conv1");
    h = group_norm_affine(m, h, pre + ".norm2");
    h = ggml_silu(ctx, h);
    h = conv2d(m, h, pre + ".conv2");
    ggml_tensor * sc = x;
    if (m.has(pre + ".conv_shortcut.weight")) {
        sc = conv2d(m, x, pre + ".conv_shortcut", 1, 0);
    }
    return ggml_add(ctx, h, sc);
}

ggml_tensor * attn_block(Model & m, ggml_tensor * x, const std::string & pre) {
    ggml_context * ctx = m.ctx_g;
    const int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2];
    const int64_t T = W * H;
    const int64_t hd = m.head_dim > 0 ? m.head_dim : C;
    const int64_t n_head = C / hd;

    ggml_tensor * h = group_norm_affine(m, x, pre + ".group_norm");
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
    out = ggml_cont(ctx, ggml_transpose(ctx, out));                 // (T, C)
    out = ggml_reshape_4d(ctx, out, W, H, C, 1);
    return ggml_add(ctx, out, x);
}
