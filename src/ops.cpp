#include "ops.h"
#include "winograd.h"

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

// Vulkan's ggml_mul_mat has a silent int8 fast path (quantize_y) that
// quantizes the activation operand to Q8_1 whenever the device supports
// VK_KHR_shader_integer_dot_product and that operand's (ne0*ne1)%4==0 --
// true for nearly every conv/linear/attention shape here.
// ggml_mul_mat_set_prec(GGML_PREC_F32) doesn't disable it, only selects a
// variant of the already-quantized pipeline. An out_prod-based dodge (and,
// for quantized/f16 weights where out_prod can't run at all, an explicit
// F16 error-feedback residual pass) was tried and measured exact against a
// CPU ground truth -- but per docs/decisions/0010-speedup-candidates.md,
// applying it blanket to every matmul in the whole network cost ~7x on the
// main UNet loop (108s -> 749s, real trace data), while MADR 0009 already
// showed plain mul_mat + PREC_F32 (quantize_y's precision loss included)
// produces a full, visually-coherent, 0.69-mean-IoU production run. The
// precision loss is real but was never shown to be visible in output;
// paying 7x for it isn't justified without a demonstrated regression.
// Reverted to the MADR-0009-proven behavior. See git history / that MADR
// for the out_prod/residual approach if a specific op site is later shown
// to need it.
static ggml_tensor * mul_mat_f32(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    ggml_tensor * r = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_set_prec(r, GGML_PREC_F32);
    return r;
}

// ggml-vulkan's quantize_y fast path (see mul_mat_f32's comment above)
// requires ggml_is_contiguous(src1) -- the SECOND ggml_mul_mat operand,
// i.e. the activation, not the weight (ggml-vulkan.cpp's
// ggml_vk_mul_mat_q_f16, `quantize_y = ... && ggml_is_contiguous(src1) &&
// ...`). Deliberately break that contiguity -- zero-pad ne[0] by one
// element then view back the original ne[0], making nb[1] one element
// wider than a tightly-packed row would be -- so the backend falls back to
// its existing (already-correct, no int8 detour) dequant kernel instead.
// Cheap: one extra column of zeros (ne[1] elements) and a view, not a full
// permute+cont+out_prod composition, and works for any operand type
// (unlike mul_mat_op, which needs an f32 upcast and has no Vulkan f16
// OUT_PROD support at all).
static ggml_tensor * break_contig(ggml_context * ctx, ggml_tensor * b) {
    // ggml-vulkan's PAD kernel, like OUT_PROD, has no f16 dispatch -- and
    // quantize_y only ever triggers for an f32 activation operand anyway
    // (ggml_vk_mul_mat_q_f16's condition includes src1->type ==
    // GGML_TYPE_F32), so upcasting here isn't paying for precision we
    // didn't already need: b is f32 in every call site quantize_y would
    // otherwise engage on.
    if (b->type != GGML_TYPE_F32) b = ggml_cast(ctx, b, GGML_TYPE_F32);
    ggml_tensor * padded = ggml_pad(ctx, b, 1, 0, 0, 0);   // (ne0+1, ne1, ne2, ne3), zero-filled
    return ggml_view_4d(ctx, padded, b->ne[0], b->ne[1], b->ne[2], b->ne[3],
                        padded->nb[1], padded->nb[2], padded->nb[3], 0);
}

static bool use_conv_noquant() {
    static int cached = -1;
    if (cached < 0) { cached = getenv("SEETHROUGH_CONV_NOQUANT") ? 1 : 0; }
    return cached != 0;
}

static ggml_tensor * mul_mat_noquant(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    ggml_tensor * r = ggml_mul_mat(ctx, a, break_contig(ctx, b));
    ggml_mul_mat_set_prec(r, GGML_PREC_F32);
    return r;
}

// See unet_frame.cpp's mul_mat_op/SEETHROUGH_ATTN_OUTPROD comment: same
// out_prod-based exact replacement for ggml_mul_mat(a,b), gated separately
// here (SEETHROUGH_CONV_OUTPROD) to A/B this specific site.
static ggml_tensor * mul_mat_op(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    // ggml-vulkan's OUT_PROD kernel only supports f32 operands (no f16 or
    // quantized-type dispatch, unlike mul_mat's fused dequant paths) --
    // upcast whichever operand isn't already f32 before permuting.
    if (a->type != GGML_TYPE_F32) a = ggml_cast(ctx, a, GGML_TYPE_F32);
    if (b->type != GGML_TYPE_F32) b = ggml_cast(ctx, b, GGML_TYPE_F32);
    ggml_tensor * a_op = ggml_cont(ctx, ggml_permute(ctx, a, 1, 0, 2, 3));
    ggml_tensor * b_op = ggml_cont(ctx, ggml_permute(ctx, b, 1, 0, 2, 3));
    return ggml_out_prod(ctx, a_op, b_op);
}

static bool use_conv_outprod() {
    static int cached = -1;
    if (cached < 0) { cached = getenv("SEETHROUGH_CONV_OUTPROD") ? 1 : 0; }
    return cached != 0;
}

static ggml_tensor * conv_mm(ggml_context * ctx, ggml_tensor * im, ggml_tensor * w) {
    ggml_tensor * im2d = ggml_reshape_2d(ctx, im, im->ne[0], im->ne[3] * im->ne[2] * im->ne[1]);  // [K, N*OH*OW]
    ggml_tensor * w2d  = ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1] * w->ne[2], w->ne[3]);        // [K, OC]
    ggml_tensor * r = use_conv_outprod() ? mul_mat_op(ctx, w2d, im2d)
                    : use_conv_noquant() ? mul_mat_noquant(ctx, w2d, im2d)
                    : mul_mat_f32(ctx, w2d, im2d); // [OC, N*OH*OW]
    r = ggml_reshape_4d(ctx, r, w->ne[3], im->ne[1], im->ne[2], im->ne[3]);            // [OC,OW,OH,N]
    return ggml_cont(ctx, ggml_permute(ctx, r, 2, 0, 1, 3));                            // -> [OW,OH,OC,N]
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
        // Chunk count sized to the actual im2col transient, not a fixed 8:
        // the fixed count dates from f16-model VRAM pressure; with Q4
        // weights the finest UNet level's transient is ~4GB and the coarser
        // levels are well under 1GB, so blanket 8-way tiling just multiplied
        // im2col/concat/cont dispatches (measured via GGML_VK_PERF_LOGGER:
        // conv overhead ops were ~0.5s of a 4.1s UNet step). Budget default
        // 2GB keeps every chunk comfortably under Vulkan's ~4.29GB
        // maxStorageBufferRange; NCH==1 collapses to a single untiled
        // im2col+mul_mat with no halo/crop/concat at all.
        // SEETHROUGH_CONV_F16=1: f16 im2col even for quantized weights --
        // halves im2col traffic and the GEMM's activation reads. MADR
        // 0009's production-shape tap analysis already characterized f16
        // conv-activation rounding in this UNet as ordinary noise that
        // shrinks 20x through the up-path; opt-in until IoU-verified here.
        const ggml_type imt = (w->type == GGML_TYPE_F16 || !getenv("SEETHROUGH_NO_CONV_F16"))
                                  ? GGML_TYPE_F16 : GGML_TYPE_F32;
        int64_t budget_mb = 2048;
        if (const char * e = getenv("SEETHROUGH_ROWCHUNK_BUDGET_MB")) budget_mb = atoll(e);
        const int64_t im2col_bytes = W * H * N * C * 9 * (imt == GGML_TYPE_F16 ? 2 : 4);
        const int64_t NCH = std::max<int64_t>(1, (im2col_bytes + budget_mb * 1024 * 1024 - 1) / (budget_mb * 1024 * 1024));
        const int64_t TS = (H + NCH - 1) / NCH;
        if (NCH == 1) {
            ggml_tensor * im = ggml_im2col(ctx, w, x, 1, 1, 1, 1, 1, 1, true, imt);
            ggml_tensor * r = conv_mm(ctx, im, w);
            if (m.has(pre + ".bias")) r = ggml_add(ctx, r, bias4d(ctx, m.get(pre + ".bias")));
            return r;
        }
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
                ys = conv_mm(ctx, im, w);
            } else {
                // edge tiles: full pad then crop the halo rows back out
                ggml_tensor * im = ggml_im2col(ctx, w, slab, 1, 1, 1, 1, 1, 1, true,
                                               w->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32);
                ggml_tensor * full = conv_mm(ctx, im, w);
                ys = ggml_cont(ctx, ggml_view_4d(ctx, full, W, y1 - y0, w->ne[3], N,
                                                 full->nb[1], full->nb[2], full->nb[3],
                                                 (y0 - top) * full->nb[1]));
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
    // Winograd F(2x2,3x3): only valid for the specific stride-1 pad-1 3x3
    // case (conv2d_winograd_3x3s1 hardcodes that receptive field in its own
    // im2col patch extraction, doesn't re-check stride/pad itself). Not
    // wired into the conv_row_chunk/direct_conv branches above: this
    // implementation processes the whole tensor in one shot (no row
    // tiling), so using it there would reintroduce the VRAM blowup those
    // paths exist to avoid at large (1280px) spatial sizes -- that needs a
    // row-chunked Winograd variant or replacing ggml_conv_2d_direct
    // specifically, neither done yet.
    //
    // OPT-IN only (SEETHROUGH_WINOGRAD=1): verified exact, but measured
    // SLOWER than plain im2col+mul_mat at production shapes. Its
    // composition (16 batch=1 ggml_out_prod calls + add/sub/concat/cont
    // transform trees per conv) made OUT_PROD alone 24% of a TransparentVAE
    // decode graph and exploded node counts (~33k CONTs/graph) -- the
    // theoretical 2.25x FLOP saving never survived the op-count and
    // out_prod-kernel overhead (GGML_VK_PERF_LOGGER, 2026-07-20). Revisit
    // only as a fused kernel, not an op composition.
    if (getenv("SEETHROUGH_WINOGRAD") &&
        stride == 1 && pad == 1 && w->ne[0] == 3 && w->ne[1] == 3) {
        if (ggml_tensor * r = conv2d_winograd_3x3s1(m, x, pre)) { return r; }
    }
    // ggml_conv_2d unconditionally rounds activations to f16 in im2col; for
    // f32 weights (SDXL VAE encoder has activations too large for that) keep
    // the whole conv in f32 instead
    ggml_type it = (w->type == GGML_TYPE_F16 || !getenv("SEETHROUGH_NO_CONV_F16"))
                       ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_tensor * im = ggml_im2col(ctx, w, x, stride, stride, pad, pad, 1, 1, true, it);
    ggml_tensor * r = conv_mm(ctx, im, w);
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
    ggml_tensor * w = m.get(pre + ".weight");
    // A linear is pointwise over every position past ne0, so a contiguous
    // higher-rank input is exactly its 2D flattening -- and the flattened
    // GEMM is dramatically faster when ne1 is small with a large ne2/ne3
    // "batch": cross_frame_block runs every linear/FFN at (C, F=13,
    // S=1600..6400), which ggml_mul_mat executes as S tiny 13-column GEMMs,
    // re-reading the whole (possibly quantized) weight S times. Measured
    // via GGML_VK_PERF_LOGGER at production shape: the n=13/batch=1600
    // variant of the same multiply runs ~5x slower per FLOP than its
    // n=1600/batch=13 sibling, and these calls were ~52% of every UNet
    // step. Flattening is a pure view (no copy, no math change): identical
    // element order in, identical element order out.
    const bool flatten = ggml_is_contiguous(x) && (x->ne[2] > 1 || x->ne[3] > 1);
    const int64_t ne1 = x->ne[1], ne2 = x->ne[2], ne3 = x->ne[3];
    if (flatten) x = ggml_reshape_2d(ctx, x, x->ne[0], ne1 * ne2 * ne3);
    // GGML_PREC_F32 was adopted for the VAE's fragile near-cancelling
    // resnet reduction and flash attention (MADR 0009) -- it was never
    // shown necessary for the UNet transformer linears, where f32
    // accumulation halves coopmat2 GEMM throughput. Experiment gate:
    // SEETHROUGH_LINEAR_FAST=1 uses the backend-default precision here
    // (conv2d and attention keep their PREC_F32 guards regardless).
    ggml_tensor * y;
    if (m.linear_fast || getenv("SEETHROUGH_LINEAR_FAST")) {
        y = ggml_mul_mat(ctx, w, x);
    } else if (use_conv_outprod()) {
        y = mul_mat_op(ctx, w, x);
    } else if (use_conv_noquant()) {
        y = mul_mat_noquant(ctx, w, x);
    } else {
        y = mul_mat_f32(ctx, w, x);
    }
    if (flatten) y = ggml_reshape_4d(ctx, y, y->ne[0], ne1, ne2, ne3);
    if (m.has(pre + ".bias")) y = ggml_add(ctx, y, m.get(pre + ".bias"));
    return y;
}

ggml_tensor * resnet_block(Model & m, ggml_tensor * x, const std::string & pre,
                           ggml_tensor * temb) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * h = group_norm_affine(m, x, pre + ".norm1");
    h = ggml_silu(ctx, h);
    h = conv2d(m, h, pre + ".conv1");
    debug_tap(m, pre + ".conv1_out", h);
    if (temb) {
        ggml_tensor * t = linear(m, ggml_silu(ctx, temb), pre + ".time_emb_proj");
        h = ggml_add(ctx, h, ggml_reshape_4d(ctx, t, 1, 1, t->ne[0], t->ne[1]));
    }
    h = group_norm_affine(m, h, pre + ".norm2");
    h = ggml_silu(ctx, h);
    h = conv2d(m, h, pre + ".conv2");
    debug_tap(m, pre + ".conv2_out", h);
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
            ggml_tensor * kqc = mul_mat_f32(ctx, k, qc);             // (T, chunk, H, N)
            kqc = ggml_soft_max(ctx, ggml_scale(ctx, kqc, 1.0f / sqrtf((float) hd)));
            ggml_tensor * chunk_kqv = mul_mat_f32(ctx, v, kqc);      // (d, chunk, H, N)
            acc = acc ? ggml_concat(ctx, acc, chunk_kqv, 1) : chunk_kqv;
        }
        kqv = acc;
    } else {
        ggml_tensor * kq = mul_mat_f32(ctx, k, q);                       // (T,T,H,N)
        kq = ggml_soft_max(ctx, ggml_scale(ctx, kq, 1.0f / sqrtf((float) hd)));
        kqv = mul_mat_f32(ctx, v, kq);                                   // (d,T,H,N)
    }
    kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));       // (d,H,T,N)
    kqv = ggml_reshape_3d(ctx, kqv, C, T, N);                       // (C, T, N)

    ggml_tensor * out = linear(m, kqv, pre + ".to_out.0");          // (C, T, N)
    out = ggml_cont(ctx, ggml_permute(ctx, out, 1, 0, 2, 3));       // (T, C, N)
    out = ggml_reshape_4d(ctx, out, W, H, C, N);
    return ggml_add(ctx, out, x);
}
