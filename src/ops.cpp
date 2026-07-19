#include "ops.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>

Model::~Model() {
    for (ggml_backend_buffer * b : bufs) ggml_backend_buffer_free(b);
    for (ggml_context * c : ctx_w) ggml_free(c);
}

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

bool Model::load_backend(const char * path, ggml_backend_buffer_type * buft) {
    ggml_context * c = nullptr;
    gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ &c };
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
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(c, buft);
    if (!buf) { gguf_free(g); return false; }
    bufs.push_back(buf);

    FILE * f = fopen(path, "rb");
    if (!f) { gguf_free(g); return false; }
    const size_t data_off = gguf_get_data_offset(g);
    std::vector<uint8_t> staging;
    for (ggml_tensor * t = ggml_get_first_tensor(c); t; t = ggml_get_next_tensor(c, t)) {
        int64_t idx = gguf_find_tensor(g, ggml_get_name(t));
        if (idx < 0) { fclose(f); gguf_free(g); return false; }
        size_t nbytes = ggml_nbytes(t);
        staging.resize(nbytes);
#ifdef _WIN32
        int seek_rc = _fseeki64(f, (long long) (data_off + gguf_get_tensor_offset(g, idx)), SEEK_SET);
#else
        int seek_rc = fseeko(f, (off_t) (data_off + gguf_get_tensor_offset(g, idx)), SEEK_SET);
#endif
        if (seek_rc != 0 || fread(staging.data(), 1, nbytes, f) != nbytes) {
            fclose(f); gguf_free(g); return false;
        }
        ggml_backend_tensor_set(t, staging.data(), 0, nbytes);
        weights[ggml_get_name(t)] = t;
    }
    fclose(f);
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

void debug_tap(Model & m, const std::string & name, ggml_tensor * t) {
    if (!m.debug_capture) { return; }
    ggml_set_output(t);
    m.debug_taps.push_back({ name, { t->ne[0], t->ne[1], t->ne[2], t->ne[3] }, t });
}

ggml_tensor * bias4d(ggml_context * ctx, ggml_tensor * b) {
    return ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
}

ggml_tensor * conv2d(Model & m, ggml_tensor * x, const std::string & pre, int stride, int pad) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * w = m.get(pre + ".weight");
    // decode-stage / UNet low-VRAM mode: exact im2col numerics, tiled over
    // output rows with a halo so every transient stays ~NCH times smaller.
    // Only for the stride-1 pad-1 k3 convs at big spatial sizes. Batch (N)
    // is passed through untouched -- im2col/mul_mat/concat are already
    // batch-generic, this just tiles the row dimension. Tried before
    // direct_conv: the UNet enables both (direct_conv as the fallback for
    // the stride-2 downsampler convs this path doesn't handle).
    if (m.conv_row_chunk && stride == 1 && pad == 1 &&
        w->ne[0] == 3 && x->ne[0] * x->ne[1] >= m.conv_row_chunk_min_hw) {
        const int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], N = x->ne[3];
        const int64_t NCH = 8, TS = (H + NCH - 1) / NCH;
        bool rc = m.conv_row_chunk;
        m.conv_row_chunk = false;
        ggml_tensor * acc = nullptr;
        for (int64_t y0 = 0; y0 < H; y0 += TS) {
            const int64_t y1 = std::min(y0 + TS, H);
            const int64_t top = y0 > 0 ? y0 - 1 : 0;           // halo rows
            const int64_t bot = y1 < H ? y1 + 1 : H;
            ggml_tensor * slab = ggml_view_4d(ctx, x, W, bot - top, C, N,
                                              x->nb[1], x->nb[2], x->nb[3],
                                              top * x->nb[1]);
            slab = ggml_cont(ctx, slab);
            ggml_tensor * ys;
            if (y0 > 0 && y1 < H) {
                // interior: horizontal pad only — output rows match exactly
                ggml_tensor * im = ggml_im2col(ctx, w, slab, 1, 1, 1, 0, 1, 1, true,
                                               w->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32);
                ggml_tensor * r = ggml_mul_mat(ctx,
                    ggml_reshape_2d(ctx, im, im->ne[0], im->ne[3] * im->ne[2] * im->ne[1]),
                    ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1] * w->ne[2], w->ne[3]));
                r = ggml_reshape_4d(ctx, r, im->ne[1], im->ne[2], im->ne[3], w->ne[3]);
                ys = ggml_cont(ctx, ggml_permute(ctx, r, 0, 1, 3, 2));
            } else {
                // edge tiles: full pad then crop the halo rows back out
                ggml_tensor * im = ggml_im2col(ctx, w, slab, 1, 1, 1, 1, 1, 1, true,
                                               w->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32);
                ggml_tensor * r = ggml_mul_mat(ctx,
                    ggml_reshape_2d(ctx, im, im->ne[0], im->ne[3] * im->ne[2] * im->ne[1]),
                    ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1] * w->ne[2], w->ne[3]));
                r = ggml_reshape_4d(ctx, r, im->ne[1], im->ne[2], im->ne[3], w->ne[3]);
                ggml_tensor * full = ggml_cont(ctx, ggml_permute(ctx, r, 0, 1, 3, 2));
                ys = ggml_cont(ctx, ggml_view_4d(ctx, full, W, y1 - y0, w->ne[3], N,
                                                 full->nb[1], full->nb[2], full->nb[3],
                                                 (y0 - top) * full->nb[1]));
            }
            if (getenv("SEETHROUGH_DEBUG_ROWCHUNK")) {
                fprintf(stderr, "[rowchunk] H=%lld W=%lld C=%lld TS=%lld y0=%lld y1=%lld top=%lld bot=%lld "
                                "ys.ne=(%lld,%lld,%lld,%lld) acc.ne=(%lld,%lld,%lld,%lld)\n",
                        (long long) H, (long long) W, (long long) C, (long long) TS,
                        (long long) y0, (long long) y1, (long long) top, (long long) bot,
                        (long long) ys->ne[0], (long long) ys->ne[1], (long long) ys->ne[2], (long long) ys->ne[3],
                        acc ? (long long) acc->ne[0] : -1, acc ? (long long) acc->ne[1] : -1,
                        acc ? (long long) acc->ne[2] : -1, acc ? (long long) acc->ne[3] : -1);
            }
            acc = acc ? ggml_concat(ctx, acc, ys, 1) : ys;
        }
        m.conv_row_chunk = rc;
        if (m.has(pre + ".bias")) acc = ggml_add(ctx, acc, bias4d(ctx, m.get(pre + ".bias")));
        return acc;
    }
    // low-VRAM mode: direct conv — the im2col transients dominate the 1280px
    // graphs (up to 5.7GB each). Fallback for shapes row-chunk doesn't cover
    // (stride-2 downsamplers, or below conv_row_chunk_min_hw).
    if (m.direct_conv) {
        ggml_tensor * r = ggml_conv_2d_direct(ctx, w, x, stride, stride, pad, pad, 1, 1);
        if (m.has(pre + ".bias")) r = ggml_add(ctx, r, bias4d(ctx, m.get(pre + ".bias")));
        return r;
    }
    // ggml_conv_2d unconditionally rounds activations to f16 in im2col; for
    // f32 weights (SDXL VAE encoder has activations too large for that) keep
    // the whole conv in f32 instead
    ggml_type it = w->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_tensor * im = ggml_im2col(ctx, w, x, stride, stride, pad, pad, 1, 1, true, it);
    ggml_tensor * r = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im, im->ne[0], im->ne[3] * im->ne[2] * im->ne[1]),
        ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1] * w->ne[2], w->ne[3]));
    r = ggml_reshape_4d(ctx, r, im->ne[1], im->ne[2], im->ne[3], w->ne[3]);
    r = ggml_cont(ctx, ggml_permute(ctx, r, 0, 1, 3, 2));
    if (m.has(pre + ".bias")) r = ggml_add(ctx, r, bias4d(ctx, m.get(pre + ".bias")));
    return r;
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
    if (m.has(pre + ".bias")) y = ggml_add(ctx, y, m.get(pre + ".bias"));
    return y;
}

ggml_tensor * resnet_block(Model & m, ggml_tensor * x, const std::string & pre,
                           ggml_tensor * temb) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * h = group_norm_affine(m, x, pre + ".norm1");
    h = ggml_silu(ctx, h);
    h = conv2d(m, h, pre + ".conv1");
    if (temb) {
        ggml_tensor * t = linear(m, ggml_silu(ctx, temb), pre + ".time_emb_proj");
        h = ggml_add(ctx, h, ggml_reshape_4d(ctx, t, 1, 1, t->ne[0], t->ne[1]));
    }
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
    const int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], N = x->ne[3];
    const int64_t T = W * H;
    const int64_t hd = m.head_dim > 0 ? m.head_dim : C;
    const int64_t n_head = C / hd;

    ggml_tensor * h = group_norm_affine(m, x, pre + ".group_norm");
    h = ggml_reshape_3d(ctx, h, T, C, N);
    h = ggml_cont(ctx, ggml_permute(ctx, h, 1, 0, 2, 3));           // (C, T, N)

    ggml_tensor * q = linear(m, h, pre + ".to_q");                  // (C, T, N)
    ggml_tensor * k = linear(m, h, pre + ".to_k");
    ggml_tensor * v = linear(m, h, pre + ".to_v");

    q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, q, hd, n_head, T, N), 0, 2, 1, 3)); // (d,T,H,N)
    k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, k, hd, n_head, T, N), 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, hd, n_head, T, N), 1, 2, 0, 3)); // (T,d,H,N)

    ggml_tensor * kqv;
    if (m.tiled_naive_attn) {
        // attn_block's spatial self-attention is always-naive, always
        // O(T^2), and completely untiled -- at the VAE decoder's mid_block
        // (T = latent_res^2, e.g. 25600 at res=1280/ZR=160), this is the
        // one unguarded quadratic-in-T op in the whole decode chain (conv2d
        // is O(T) and row-chunk-tiled/Lean-gated; attn_tokens has the
        // flash_attn/tiled_naive_attn knobs). Suspected site of the paused
        // 1280px collapse (docs/ggml-upstream-issues.md #4) since neither
        // was ever exercised here before -- reuses the same query-tiling
        // idea as attn_tokens_tiled_naive.
        const int64_t target_bytes = 512LL * 1024 * 1024;
        const int64_t per_token = std::max<int64_t>(T * n_head * N * (int64_t) sizeof(float), 1);
        int64_t chunk = std::min<int64_t>(T, std::max<int64_t>(1, target_bytes / per_token));
        ggml_tensor * acc = nullptr;
        for (int64_t t0 = 0; t0 < T; t0 += chunk) {
            const int64_t t1 = std::min(t0 + chunk, T);
            ggml_tensor * qc = ggml_cont(ctx, ggml_view_4d(ctx, q, hd, t1 - t0, n_head, N,
                                                           q->nb[1], q->nb[2], q->nb[3],
                                                           t0 * q->nb[1]));
            ggml_tensor * kqc = ggml_mul_mat(ctx, k, qc);           // (T, chunk, H, N)
            kqc = ggml_soft_max(ctx, ggml_scale(ctx, kqc, 1.0f / sqrtf((float) hd)));
            ggml_tensor * chunk_kqv = ggml_mul_mat(ctx, v, kqc);    // (d, chunk, H, N)
            acc = acc ? ggml_concat(ctx, acc, chunk_kqv, 1) : chunk_kqv;
        }
        kqv = acc;
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx, k, q);                     // (T,T,H,N)
        kq = ggml_soft_max(ctx, ggml_scale(ctx, kq, 1.0f / sqrtf((float) hd)));
        kqv = ggml_mul_mat(ctx, v, kq);                                 // (d,T,H,N)
    }
    kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));       // (d,H,T,N)
    kqv = ggml_reshape_3d(ctx, kqv, C, T, N);                       // (C, T, N)

    ggml_tensor * out = linear(m, kqv, pre + ".to_out.0");          // (C, T, N)
    out = ggml_cont(ctx, ggml_permute(ctx, out, 1, 0, 2, 3));       // (T, C, N)
    out = ggml_reshape_4d(ctx, out, W, H, C, N);
    return ggml_add(ctx, out, x);
}
