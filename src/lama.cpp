#include "lama.h"

#include "ggml-backend.h"

#include <cmath>
#include <cstring>
#include <map>
#include <tuple>
#include <vector>

// ---------------------------------------------------------------------------
// DFT-via-matmul: mathematically exact replacement for a 2-D real FFT (same
// linear transform an FFT computes, just via a dense basis matrix instead of
// a butterfly network) -- lets the whole LaMa graph, FFT included, run on
// any ggml backend via plain ggml_mul_mat/add/sub/permute, none of which
// need a custom op.
//
// Forward (rfft2): real x (W,H,C,N) -> complex (fr,fi), each (Wh,H,C,N),
// Wh=W/2+1, ortho-normalized (scale 1/sqrt(W*H)):
//   row transform (W axis, real -> truncated complex):
//     Xr = Ar^T x, Xi = Ai^T x      Ar[j,k]=cos(2pi jk/W), Ai[j,k]=-sin(2pi jk/W)
//   col transform (H axis, full complex DFT), applied to the permuted
//   (H,Wh,C,N) tensor so H is the contracted (ne0) axis:
//     Yr = Br^T Xr - Bi^T Xi, Yi = Br^T Xi + Bi^T Xr
//     Br[j,k]=cos(2pi jk/H), Bi[j,k]=-sin(2pi jk/H)
//
// Inverse (irfft2): complex (Wh,H,C,N) -> real (W,H,C,N):
//   col transform (H axis, inverse = conjugate of the forward basis, i.e.
//   same Br but -Bi):
//     Tr = Br^T Sr + Bi^T Si, Ti = Br^T Si - Bi^T Sr
//   row transform (Wh -> W axis): a closed-form real-output inverse that
//   folds in the Hermitian-symmetric mirror terms algebraically (derived
//   from x[w] = Re{sum_k S_full[k] exp(i2pi kw/W)} with S_full built from
//   the half-spectrum S by conjugate-mirroring) instead of literally
//   constructing the mirrored half and running a further complex DFT --
//   avoids needing any gather/scatter op:
//     x = Dr^T Tr - Di^T Ti
//     Dr[k,w]=weight(k)cos(2pi kw/W), Di[k,w]=weight(k)sin(2pi kw/W)
//     weight(0)=weight(Wh-1)=1 (W even -- true here, self-paired Nyquist
//     bin), weight(k)=2 otherwise (each interior bin's mirror pair
//     contributes an equal real part, per the conjugate-mirror identity
//     Re{conj(S[m])e^{-i2pi mw/W}} = Re{S[m]e^{i2pi mw/W}}).
// ---------------------------------------------------------------------------

static ggml_tensor * mm_t(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    return ggml_mul_mat(ctx, a, b);
}

// x real (W,H,C,N) -> fr,fi each (Wh,H,C,N)
static void rfft2_mm(ggml_context * ctx, ggml_tensor * x,
                     ggml_tensor * Ar, ggml_tensor * Ai, ggml_tensor * Br, ggml_tensor * Bi,
                     ggml_tensor ** fr_out, ggml_tensor ** fi_out) {
    ggml_tensor * Xr = mm_t(ctx, Ar, x);   // (Wh,H,C,N)
    ggml_tensor * Xi = mm_t(ctx, Ai, x);
    ggml_tensor * Xr_t = ggml_cont(ctx, ggml_permute(ctx, Xr, 1, 0, 2, 3));   // (H,Wh,C,N)
    ggml_tensor * Xi_t = ggml_cont(ctx, ggml_permute(ctx, Xi, 1, 0, 2, 3));
    ggml_tensor * Yr_t = ggml_sub(ctx, mm_t(ctx, Br, Xr_t), mm_t(ctx, Bi, Xi_t));
    ggml_tensor * Yi_t = ggml_add(ctx, mm_t(ctx, Br, Xi_t), mm_t(ctx, Bi, Xr_t));
    ggml_tensor * Yr = ggml_cont(ctx, ggml_permute(ctx, Yr_t, 1, 0, 2, 3));   // (Wh,H,C,N)
    ggml_tensor * Yi = ggml_cont(ctx, ggml_permute(ctx, Yi_t, 1, 0, 2, 3));
    const float scale = 1.0f / sqrtf((float) (x->ne[0] * x->ne[1]));
    *fr_out = ggml_scale(ctx, Yr, scale);
    *fi_out = ggml_scale(ctx, Yi, scale);
}

// f (Wh,H,2C,N) block layout (first C channels real, next C imag -- see
// lama_prepare_gpu_weights: conv_layer's output-channel axis and .fu.bn are
// deinterleaved to match) -> real (W,H,C,N)
static ggml_tensor * irfft2_mm(ggml_context * ctx, ggml_tensor * f, int64_t W,
                               ggml_tensor * Br, ggml_tensor * Bi, ggml_tensor * Dr, ggml_tensor * Di) {
    const int64_t Wh = f->ne[0], H = f->ne[1], C = f->ne[2] / 2, N = f->ne[3];
    // plain contiguous half-split (unlike the old interleaved layout, no
    // doubled/strided view needed here)
    ggml_tensor * Sr = ggml_cont(ctx, ggml_view_4d(ctx, f, Wh, H, C, N,
                                                   f->nb[1], f->nb[2], f->nb[3], 0));
    ggml_tensor * Si = ggml_cont(ctx, ggml_view_4d(ctx, f, Wh, H, C, N,
                                                   f->nb[1], f->nb[2], f->nb[3], C * f->nb[2]));
    ggml_tensor * Sr_t = ggml_cont(ctx, ggml_permute(ctx, Sr, 1, 0, 2, 3));   // (H,Wh,C,N)
    ggml_tensor * Si_t = ggml_cont(ctx, ggml_permute(ctx, Si, 1, 0, 2, 3));
    ggml_tensor * Tr_t = ggml_add(ctx, mm_t(ctx, Br, Sr_t), mm_t(ctx, Bi, Si_t));
    ggml_tensor * Ti_t = ggml_sub(ctx, mm_t(ctx, Br, Si_t), mm_t(ctx, Bi, Sr_t));
    ggml_tensor * Tr = ggml_cont(ctx, ggml_permute(ctx, Tr_t, 1, 0, 2, 3));   // (Wh,H,C,N)
    ggml_tensor * Ti = ggml_cont(ctx, ggml_permute(ctx, Ti_t, 1, 0, 2, 3));
    ggml_tensor * x = ggml_sub(ctx, mm_t(ctx, Dr, Tr), mm_t(ctx, Di, Ti));    // (W,H,C,N)
    const float scale = 1.0f / sqrtf((float) (W * H));
    return ggml_scale(ctx, x, scale);
}

// reflect-index data for a (len,pad) pair -- see reflect_pad_axis0's
// comment for the exact index formula (matches ggml_compute_forward_pad_
// reflect_1d, ggml-cpu/ops.cpp, bit-for-bit).
static std::vector<int32_t> reflect_pad_index_data(int64_t len, int p) {
    std::vector<int32_t> idx(len + 2 * p);
    for (int64_t k = 0; k < (int64_t) idx.size(); k++) {
        int64_t src;
        if (k < p) src = p - k;
        else if (k < p + len) src = k - p;
        else src = 2 * len - 2 - k + p;
        idx[k] = (int32_t) src;
    }
    return idx;
}

// ggml_get_rows requires idx's ne[1] to exactly equal the gathered tensor's
// channel count (no broadcast) -- tile the base index pattern once per
// channel to match the (len+2p, channels, 1) shape get_pad_idx() created.
static void set_pad_idx(const LamaExtraInputs::PadIdx & p) {
    std::vector<int32_t> base = reflect_pad_index_data(p.len, p.pad);
    std::vector<int32_t> full((size_t) base.size() * (size_t) p.channels);
    for (int64_t c = 0; c < p.channels; c++) {
        std::copy(base.begin(), base.end(), full.begin() + (size_t) c * base.size());
    }
    ggml_backend_tensor_set(p.t, full.data(), 0, full.size() * 4);
}

void lama_set_extra_inputs(const LamaExtraInputs & extra, int64_t img_W) {
    // FFT basis: W here is the global-branch spatial size (img_W/8,
    // matching lama_inpaint_graph's Wg), not the full image width.
    const int64_t W = img_W / 8;
    const int64_t Wh = W / 2 + 1, H = W;
    const double pi = 3.14159265358979323846;

    std::vector<float> Ar((size_t) W * Wh), Ai((size_t) W * Wh);
    for (int64_t j = 0; j < W; j++) {
        for (int64_t k = 0; k < Wh; k++) {
            double th = 2.0 * pi * (double) j * (double) k / (double) W;
            Ar[(size_t) k * W + j] = (float) cos(th);
            Ai[(size_t) k * W + j] = (float) (-sin(th));
        }
    }
    std::vector<float> Br((size_t) H * H), Bi((size_t) H * H);
    for (int64_t j = 0; j < H; j++) {
        for (int64_t k = 0; k < H; k++) {
            double th = 2.0 * pi * (double) j * (double) k / (double) H;
            Br[(size_t) k * H + j] = (float) cos(th);
            Bi[(size_t) k * H + j] = (float) (-sin(th));
        }
    }
    std::vector<float> Dr((size_t) Wh * W), Di((size_t) Wh * W);
    for (int64_t k = 0; k < Wh; k++) {
        double weight = (k == 0 || (W % 2 == 0 && k == Wh - 1)) ? 1.0 : 2.0;
        for (int64_t w = 0; w < W; w++) {
            double th = 2.0 * pi * (double) k * (double) w / (double) W;
            Dr[(size_t) w * Wh + k] = (float) (weight * cos(th));
            Di[(size_t) w * Wh + k] = (float) (weight * sin(th));
        }
    }
    ggml_backend_tensor_set(extra.fft_ar, Ar.data(), 0, Ar.size() * 4);
    ggml_backend_tensor_set(extra.fft_ai, Ai.data(), 0, Ai.size() * 4);
    ggml_backend_tensor_set(extra.fft_br, Br.data(), 0, Br.size() * 4);
    ggml_backend_tensor_set(extra.fft_bi, Bi.data(), 0, Bi.size() * 4);
    ggml_backend_tensor_set(extra.fft_dr, Dr.data(), 0, Dr.size() * 4);
    ggml_backend_tensor_set(extra.fft_di, Di.data(), 0, Di.size() * 4);

    for (const auto & p : extra.pad_idxs) { set_pad_idx(p); }
}

// ---------------------------------------------------------------------------
// one-time GPU-weight fixup
// ---------------------------------------------------------------------------

// permutes one axis of a tensor from interleaved (2c=real(c),2c+1=imag(c))
// to block order (first C indices real, next C imag) -- axis 2 (input
// channels) on conv_layer.weight matches rfft2_mm's (fr,fi) concat order;
// axis 3 (output channels) on conv_layer.weight, and axis 0 on the 1-D
// .fu.bn.{weight,bias,running_mean,running_var}/conv_layer.bias tensors
// (all indexed by that same output-channel space), makes conv_layer's
// output itself land in block order too, so irfft2_mm's input can be a
// plain contiguous half-split instead of a strided every-other-channel view.
static void deinterleave_axis(ggml_tensor * t, int axis) {
    const int64_t ne[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
    int64_t es[4]; es[0] = 1; for (int i = 1; i < 4; i++) es[i] = es[i - 1] * ne[i - 1];
    const int64_t C = ne[axis] / 2;
    const size_t n = (size_t) ne[0] * ne[1] * ne[2] * ne[3];
    std::vector<float> buf(n), out(n);
    // Model::load() (host RAM, e.g. the SEETHROUGH_DEVICE=cpu test path)
    // leaves t->buffer null -- t->data is a plain, directly-readable CPU
    // pointer there. Only load_backend() (e.g. Vulkan VRAM) tensors need
    // ggml_backend_tensor_get/set, which assert on a null buffer.
    if (t->buffer) { ggml_backend_tensor_get(t, buf.data(), 0, n * 4); }
    else { std::memcpy(buf.data(), t->data, n * 4); }
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    int64_t idx[4] = { i0, i1, i2, i3 };
                    int64_t c = idx[axis];
                    idx[axis] = (c % 2 == 0) ? c / 2 : C + c / 2;
                    size_t src = (size_t) i0 * es[0] + (size_t) i1 * es[1] + (size_t) i2 * es[2] + (size_t) i3 * es[3];
                    size_t dst = (size_t) idx[0] * es[0] + (size_t) idx[1] * es[1] + (size_t) idx[2] * es[2] + (size_t) idx[3] * es[3];
                    out[dst] = buf[src];
                }
            }
        }
    }
    if (t->buffer) { ggml_backend_tensor_set(t, out.data(), 0, n * 4); }
    else { std::memcpy(t->data, out.data(), n * 4); }
}

void lama_prepare_gpu_weights(Model & m) {
    for (int i = 5; i <= 22; i++) {
        for (const char * sub : { "conv1", "conv2" }) {
            std::string pre = "model." + std::to_string(i) + "." + sub + ".ffc.convg2g.fu";
            if (m.has(pre + ".conv_layer.weight")) {
                deinterleave_axis(m.get(pre + ".conv_layer.weight"), 2);   // input channels
                deinterleave_axis(m.get(pre + ".conv_layer.weight"), 3);   // output channels
            }
            if (m.has(pre + ".conv_layer.bias")) { deinterleave_axis(m.get(pre + ".conv_layer.bias"), 0); }
            for (const char * p : { ".bn.weight", ".bn.bias", ".bn.running_mean", ".bn.running_var" }) {
                if (m.has(pre + p)) { deinterleave_axis(m.get(pre + p), 0); }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// graph builders (conv/norm/pad -- unchanged from the original)
// ---------------------------------------------------------------------------

// ggml has no Vulkan kernel for GGML_OP_PAD_REFLECT_1D at all (only CPU) --
// replace with an index gather (GGML_OP_GET_ROWS, Vulkan-supported) against
// a precomputed reflect-index tensor, matching ggml_compute_forward_pad_
// reflect_1d's exact index math (ops.cpp, ggml-cpu): output position k in
// [0,L+2p) maps to source index p-k (k<p), k-p (p<=k<p+L), or 2L-2-k+p
// (k>=p+L) -- see reflect_pad_index_data().
//
// ggml_get_rows asserts idx->ne[1]==a->ne[2] and idx->ne[2]==a->ne[3] (no
// broadcast across channels/batch), so the index tensor's shape depends on
// the channel count at each call site -- built lazily and cached by
// (length,pad,channels) so repeated call sites at the same shape share one
// tensor instead of allocating a fresh one every time.
struct PadCache {
    ggml_context * ctx;
    LamaExtraInputs * extra;
    std::map<std::tuple<int64_t, int, int64_t>, ggml_tensor *> cache;
};

static ggml_tensor * get_pad_idx(PadCache & pc, int64_t len, int pad, int64_t channels) {
    auto key = std::make_tuple(len, pad, channels);
    auto it = pc.cache.find(key);
    if (it != pc.cache.end()) return it->second;
    ggml_tensor * t = ggml_new_tensor_3d(pc.ctx, GGML_TYPE_I32, len + 2 * pad, channels, 1);
    ggml_set_input(t);
    pc.cache[key] = t;
    pc.extra->pad_idxs.push_back({ t, len, pad, channels });
    return t;
}

static ggml_tensor * reflect_pad_axis0(ggml_context * ctx, ggml_tensor * x, PadCache & pc, int pad) {
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));   // pad axis -> ne1
    ggml_tensor * idx = get_pad_idx(pc, xt->ne[1], pad, xt->ne[2]);
    ggml_tensor * g = ggml_get_rows(ctx, xt, idx);
    return ggml_cont(ctx, ggml_permute(ctx, g, 1, 0, 2, 3));
}

static ggml_tensor * rpad(ggml_context * ctx, ggml_tensor * x, PadCache & pc, int pad) {
    x = reflect_pad_axis0(ctx, x, pc, pad);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    x = reflect_pad_axis0(ctx, x, pc, pad);   // square tensors throughout: same pad for both axes
    return ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
}

// eval-mode BatchNorm as per-channel affine
static ggml_tensor * batch_norm(Model & m, ggml_tensor * x, const std::string & pre) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * var = ggml_scale_bias(ctx, m.get(pre + ".running_var"), 1.0f, 1e-5f);
    ggml_tensor * scale = ggml_div(ctx, m.get(pre + ".weight"), ggml_sqrt(ctx, var));
    ggml_tensor * shift = ggml_sub(ctx, m.get(pre + ".bias"),
                                   ggml_mul(ctx, m.get(pre + ".running_mean"), scale));
    x = ggml_mul(ctx, x, bias4d(ctx, scale));
    return ggml_add(ctx, x, bias4d(ctx, shift));
}

// bundles the DFT basis + shared PadCache (the auxiliary inputs needed
// repeatedly inside the 18-block loop) so helpers don't need a growing
// individual-tensor parameter list
struct FFCCtx {
    ggml_tensor * Ar, * Ai, * Br, * Bi, * Dr, * Di;
    PadCache * pc;
};

// reflect-padded conv (LaMa's padding_mode='reflect'); pad=0 means no
// padding (1x1 convs)
static ggml_tensor * rconv(Model & m, ggml_tensor * x, const std::string & pre,
                           PadCache & pc, int pad, int stride = 1) {
    if (pad > 0) x = rpad(m.ctx_g, x, pc, pad);
    return conv2d(m, x, pre, stride, 0);
}

// SpectralTransform (stride 1, no LFU): conv1 -> FourierUnit -> conv2(x + fu)
static ggml_tensor * spectral_transform(Model & m, ggml_tensor * g, const std::string & pre, const FFCCtx & fc) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * x = conv2d(m, g, pre + ".conv1.0", 1, 0);
    x = ggml_relu(ctx, batch_norm(m, x, pre + ".conv1.1"));

    ggml_tensor * fr, * fi;
    rfft2_mm(ctx, x, fc.Ar, fc.Ai, fc.Br, fc.Bi, &fr, &fi);
    ggml_tensor * f = ggml_concat(ctx, fr, fi, 2);   // block layout: matches
                                                      // lama_prepare_gpu_weights
    f = conv2d(m, f, pre + ".fu.conv_layer", 1, 0);
    f = ggml_relu(ctx, batch_norm(m, f, pre + ".fu.bn"));
    f = irfft2_mm(ctx, f, x->ne[0], fc.Br, fc.Bi, fc.Dr, fc.Di);

    return conv2d(m, ggml_add(ctx, x, f), pre + ".conv2", 1, 0);
}

// FFC_BN_ACT at ratio 0.75 (l 128 / g 384) inside a resnet block
static void ffc_bn_relu(Model & m, ggml_tensor *& l, ggml_tensor *& g, const std::string & pre, const FFCCtx & fc) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * out_l = ggml_add(ctx, rconv(m, l, pre + ".ffc.convl2l", *fc.pc, 1),
                                        rconv(m, g, pre + ".ffc.convg2l", *fc.pc, 1));
    ggml_tensor * out_g = ggml_add(ctx, rconv(m, l, pre + ".ffc.convl2g", *fc.pc, 1),
                                        spectral_transform(m, g, pre + ".ffc.convg2g", fc));
    l = ggml_relu(ctx, batch_norm(m, out_l, pre + ".bn_l"));
    g = ggml_relu(ctx, batch_norm(m, out_g, pre + ".bn_g"));
}

// torch ConvTranspose2d k3 s2 p1 outpad1: p0 result cropped by 1 on each
// leading edge -> exactly 2x the input size
static ggml_tensor * up_conv(Model & m, ggml_tensor * x, const std::string & pre) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * w = m.get(pre + ".weight");
    ggml_tensor * y = ggml_conv_transpose_2d_p0(ctx, w, x, 2);       // (2W+1, 2H+1, OC, N)
    const int64_t W2 = 2 * x->ne[0], H2 = 2 * x->ne[1];
    y = ggml_cont(ctx, ggml_view_4d(ctx, y, W2, H2, y->ne[2], y->ne[3],
                                    y->nb[1], y->nb[2], y->nb[3],
                                    y->nb[0] + y->nb[1]));
    return ggml_add(ctx, y, bias4d(ctx, m.get(pre + ".bias")));
}

ggml_tensor * lama_inpaint_graph(Model & m, ggml_tensor * img, ggml_tensor * mask, LamaExtraInputs * extra) {
    ggml_context * ctx = m.ctx_g;
    const int64_t W = img->ne[0];   // full image size (square, see lama.h)

    // global branch spatial size after 3 stride-2 downsamples (stem is
    // stride 1): W/8, square.
    const int64_t Wg = W / 8, Whg = Wg / 2 + 1;
    FFCCtx fc;
    fc.Ar = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Wg, Whg);
    fc.Ai = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Wg, Whg);
    fc.Br = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Wg, Wg);
    fc.Bi = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Wg, Wg);
    fc.Dr = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Whg, Wg);
    fc.Di = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Whg, Wg);
    for (ggml_tensor * t : { fc.Ar, fc.Ai, fc.Br, fc.Bi, fc.Dr, fc.Di }) ggml_set_input(t);
    extra->fft_ar = fc.Ar; extra->fft_ai = fc.Ai; extra->fft_br = fc.Br;
    extra->fft_bi = fc.Bi; extra->fft_dr = fc.Dr; extra->fft_di = fc.Di;

    PadCache pc{ ctx, extra, {} };
    fc.pc = &pc;

    ggml_tensor * inv_mask = ggml_scale_bias(ctx, mask, -1.0f, 1.0f);      // 1-mask
    ggml_tensor * x = ggml_concat(ctx, ggml_mul(ctx, img, inv_mask), mask, 2);

    // stem: reflect(3) + conv k7 + BN + ReLU
    x = rconv(m, x, "model.1.ffc.convl2l", pc, 3);
    x = ggml_relu(ctx, batch_norm(m, x, "model.1.bn_l"));

    // downsamples: k3 s2 reflect p1; the last one splits local/global
    x = rconv(m, x, "model.2.ffc.convl2l", pc, 1, 2);
    x = ggml_relu(ctx, batch_norm(m, x, "model.2.bn_l"));
    x = rconv(m, x, "model.3.ffc.convl2l", pc, 1, 2);
    x = ggml_relu(ctx, batch_norm(m, x, "model.3.bn_l"));
    ggml_tensor * l = rconv(m, x, "model.4.ffc.convl2l", pc, 1, 2);
    ggml_tensor * g = rconv(m, x, "model.4.ffc.convl2g", pc, 1, 2);
    l = ggml_relu(ctx, batch_norm(m, l, "model.4.bn_l"));
    g = ggml_relu(ctx, batch_norm(m, g, "model.4.bn_g"));

    // 18 FFC resnet blocks
    for (int i = 5; i <= 22; i++) {
        ggml_tensor * id_l = l, * id_g = g;
        ffc_bn_relu(m, l, g, "model." + std::to_string(i) + ".conv1", fc);
        ffc_bn_relu(m, l, g, "model." + std::to_string(i) + ".conv2", fc);
        l = ggml_add(ctx, l, id_l);
        g = ggml_add(ctx, g, id_g);
    }

    x = ggml_concat(ctx, l, g, 2);

    // upsamples: ConvTranspose k3 s2 p1 op1 + BN + ReLU (indices 24/27/30)
    for (int i = 24; i <= 30; i += 3) {
        x = up_conv(m, x, "model." + std::to_string(i));
        x = ggml_relu(ctx, batch_norm(m, x, "model." + std::to_string(i + 1)));
    }

    // head: reflect(3) + conv k7 + sigmoid (x is back at full W here, same
    // shape as the stem input -- get_pad_idx() will reuse that cached tensor)
    x = rpad(ctx, x, pc, 3);
    x = conv2d(m, x, "model.34", 1, 0);
    x = ggml_sigmoid(ctx, x);

    // composite: pred*mask + img*(1-mask)
    return ggml_add(ctx, ggml_mul(ctx, x, mask), ggml_mul(ctx, img, inv_mask));
}
