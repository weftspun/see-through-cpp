#include "lama.h"

#include <cmath>
#include <complex>
#include <vector>

// ---------------------------------------------------------------------------
// mixed-radix complex FFT (double), recursive Cooley-Tukey with naive DFT for
// prime radices — sizes here are bottleneck dims (multiples of 8, <= ~200)
// ---------------------------------------------------------------------------

using cd = std::complex<double>;

static const double PI = 3.14159265358979323846;

static int smallest_factor(int n) {
    for (int p = 2; p * p <= n; p++) {
        if (n % p == 0) return p;
    }
    return n;
}

// out[k] = sum_j in[j*is] * exp(sign*2πi jk/n), sign = -1 fwd / +1 inv
static void fft1d(int n, const cd * in, int is, cd * out, bool inv) {
    if (n == 1) { out[0] = in[0]; return; }
    const int p = smallest_factor(n), m = n / p;
    for (int r = 0; r < p; r++) {
        fft1d(m, in + (size_t) r * is, is * p, out + (size_t) r * m, inv);
    }
    const double sign = inv ? 2.0 * PI / n : -2.0 * PI / n;
    std::vector<cd> t(p);
    for (int k1 = 0; k1 < m; k1++) {
        for (int r = 0; r < p; r++) t[r] = out[(size_t) r * m + k1];
        for (int k2 = 0; k2 < p; k2++) {
            cd s = 0;
            const int64_t k = k1 + (int64_t) k2 * m;
            for (int r = 0; r < p; r++) {
                double a = sign * (double) ((int64_t) r * k % n);
                s += t[r] * cd(cos(a), sin(a));
            }
            out[(size_t) k2 * m + k1] = s;
        }
    }
}

// forward rfft2 with ortho norm: x (W,H,C,N) -> (W/2+1, H, 2C, N) where
// channel 2c = real, 2c+1 = imag (the torch stack/permute/view layout)
static void rfft2_cb(ggml_tensor * dst, int ith, int nth, void * /*ud*/) {
    const ggml_tensor * a = dst->src[0];
    const int64_t W = a->ne[0], H = a->ne[1], C = a->ne[2], N = a->ne[3];
    const int64_t Wh = W / 2 + 1;
    const double scale = 1.0 / sqrt((double) (W * H));

    std::vector<cd> row(W), rowf(W);
    std::vector<cd> col(H), colf(H);
    std::vector<cd> spec(Wh * H);

    for (int64_t idx = ith; idx < C * N; idx += nth) {
        const int64_t c = idx % C, n = idx / C;
        const char * base = (const char *) a->data + c * a->nb[2] + n * a->nb[3];
        for (int64_t h = 0; h < H; h++) {
            const float * src = (const float *) (base + h * a->nb[1]);
            for (int64_t w = 0; w < W; w++) row[w] = cd(src[w], 0.0);
            fft1d((int) W, row.data(), 1, rowf.data(), false);
            for (int64_t w = 0; w < Wh; w++) spec[h * Wh + w] = rowf[w];
        }
        for (int64_t w = 0; w < Wh; w++) {
            for (int64_t h = 0; h < H; h++) col[h] = spec[h * Wh + w];
            fft1d((int) H, col.data(), 1, colf.data(), false);
            for (int64_t h = 0; h < H; h++) spec[h * Wh + w] = colf[h] * scale;
        }
        float * dre = (float *) ((char *) dst->data + (2 * c) * dst->nb[2] + n * dst->nb[3]);
        float * dim = (float *) ((char *) dst->data + (2 * c + 1) * dst->nb[2] + n * dst->nb[3]);
        for (int64_t h = 0; h < H; h++) {
            for (int64_t w = 0; w < Wh; w++) {
                dre[h * Wh + w] = (float) spec[h * Wh + w].real();
                dim[h * Wh + w] = (float) spec[h * Wh + w].imag();
            }
        }
    }
}

// inverse: (W/2+1, H, 2C, N) -> real (W,H,C,N), ortho norm
static void irfft2_cb(ggml_tensor * dst, int ith, int nth, void * /*ud*/) {
    const ggml_tensor * a = dst->src[0];
    const int64_t W = dst->ne[0], H = dst->ne[1], C = dst->ne[2], N = dst->ne[3];
    const int64_t Wh = W / 2 + 1;
    const double scale = 1.0 / sqrt((double) (W * H));

    std::vector<cd> col(H), colf(H);
    std::vector<cd> row(W), rowf(W);
    std::vector<cd> spec(Wh * H);

    for (int64_t idx = ith; idx < C * N; idx += nth) {
        const int64_t c = idx % C, n = idx / C;
        const float * sre = (const float *) ((const char *) a->data + (2 * c) * a->nb[2] + n * a->nb[3]);
        const float * sim = (const float *) ((const char *) a->data + (2 * c + 1) * a->nb[2] + n * a->nb[3]);
        // inverse transform over H for each retained column
        for (int64_t w = 0; w < Wh; w++) {
            for (int64_t h = 0; h < H; h++) {
                col[h] = cd(sre[h * Wh + w], sim[h * Wh + w]);
            }
            fft1d((int) H, col.data(), 1, colf.data(), true);
            for (int64_t h = 0; h < H; h++) spec[h * Wh + w] = colf[h];
        }
        // rows are Hermitian: extend and inverse transform over W
        float * out = (float *) ((char *) dst->data + c * dst->nb[2] + n * dst->nb[3]);
        for (int64_t h = 0; h < H; h++) {
            for (int64_t w = 0; w < Wh; w++) row[w] = spec[h * Wh + w];
            for (int64_t w = Wh; w < W; w++) row[w] = std::conj(spec[h * Wh + (W - w)]);
            fft1d((int) W, row.data(), 1, rowf.data(), true);
            for (int64_t w = 0; w < W; w++) out[h * W + w] = (float) (rowf[w].real() * scale);
        }
    }
}

// ---------------------------------------------------------------------------
// graph builders
// ---------------------------------------------------------------------------

static ggml_tensor * rfft2(ggml_context * ctx, ggml_tensor * x) {
    ggml_tensor * args[1] = { x };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0] / 2 + 1, x->ne[1], 2 * x->ne[2],
                          x->ne[3], args, 1, rfft2_cb, GGML_N_TASKS_MAX, nullptr);
}

static ggml_tensor * irfft2(ggml_context * ctx, ggml_tensor * x, int64_t W, int64_t H) {
    ggml_tensor * args[1] = { x };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, W, H, x->ne[2] / 2, x->ne[3],
                          args, 1, irfft2_cb, GGML_N_TASKS_MAX, nullptr);
}

static ggml_tensor * rpad(Model & m, ggml_tensor * x, int p) {
    ggml_context * ctx = m.ctx_g;
    x = ggml_pad_reflect_1d(ctx, x, p, p);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    x = ggml_pad_reflect_1d(ctx, x, p, p);
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

// reflect-padded conv (LaMa's padding_mode='reflect')
static ggml_tensor * rconv(Model & m, ggml_tensor * x, const std::string & pre,
                           int pad, int stride = 1) {
    if (pad > 0) x = rpad(m, x, pad);
    return conv2d(m, x, pre, stride, 0);
}

// SpectralTransform (stride 1, no LFU): conv1 -> FourierUnit -> conv2(x + fu)
static ggml_tensor * spectral_transform(Model & m, ggml_tensor * g, const std::string & pre) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * x = conv2d(m, g, pre + ".conv1.0", 1, 0);
    x = ggml_relu(ctx, batch_norm(m, x, pre + ".conv1.1"));

    ggml_tensor * f = rfft2(ctx, x);
    f = conv2d(m, f, pre + ".fu.conv_layer", 1, 0);
    f = ggml_relu(ctx, batch_norm(m, f, pre + ".fu.bn"));
    f = irfft2(ctx, f, x->ne[0], x->ne[1]);

    return conv2d(m, ggml_add(ctx, x, f), pre + ".conv2", 1, 0);
}

// FFC_BN_ACT at ratio 0.75 (l 128 / g 384) inside a resnet block
static void ffc_bn_relu(Model & m, ggml_tensor *& l, ggml_tensor *& g, const std::string & pre) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * out_l = ggml_add(ctx, rconv(m, l, pre + ".ffc.convl2l", 1),
                                        rconv(m, g, pre + ".ffc.convg2l", 1));
    ggml_tensor * out_g = ggml_add(ctx, rconv(m, l, pre + ".ffc.convl2g", 1),
                                        spectral_transform(m, g, pre + ".ffc.convg2g"));
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

ggml_tensor * lama_inpaint_graph(Model & m, ggml_tensor * img, ggml_tensor * mask) {
    ggml_context * ctx = m.ctx_g;

    ggml_tensor * inv_mask = ggml_scale_bias(ctx, mask, -1.0f, 1.0f);      // 1-mask
    ggml_tensor * x = ggml_concat(ctx, ggml_mul(ctx, img, inv_mask), mask, 2);

    // stem: reflect(3) + conv k7 + BN + ReLU
    x = rconv(m, x, "model.1.ffc.convl2l", 3);
    x = ggml_relu(ctx, batch_norm(m, x, "model.1.bn_l"));

    // downsamples: k3 s2 reflect p1; the last one splits local/global
    x = rconv(m, x, "model.2.ffc.convl2l", 1, 2);
    x = ggml_relu(ctx, batch_norm(m, x, "model.2.bn_l"));
    x = rconv(m, x, "model.3.ffc.convl2l", 1, 2);
    x = ggml_relu(ctx, batch_norm(m, x, "model.3.bn_l"));
    ggml_tensor * l = rconv(m, x, "model.4.ffc.convl2l", 1, 2);
    ggml_tensor * g = rconv(m, x, "model.4.ffc.convl2g", 1, 2);
    l = ggml_relu(ctx, batch_norm(m, l, "model.4.bn_l"));
    g = ggml_relu(ctx, batch_norm(m, g, "model.4.bn_g"));

    // 18 FFC resnet blocks
    for (int i = 5; i <= 22; i++) {
        ggml_tensor * id_l = l, * id_g = g;
        ffc_bn_relu(m, l, g, "model." + std::to_string(i) + ".conv1");
        ffc_bn_relu(m, l, g, "model." + std::to_string(i) + ".conv2");
        l = ggml_add(ctx, l, id_l);
        g = ggml_add(ctx, g, id_g);
    }

    x = ggml_concat(ctx, l, g, 2);

    // upsamples: ConvTranspose k3 s2 p1 op1 + BN + ReLU (indices 24/27/30)
    for (int i = 24; i <= 30; i += 3) {
        x = up_conv(m, x, "model." + std::to_string(i));
        x = ggml_relu(ctx, batch_norm(m, x, "model." + std::to_string(i + 1)));
    }

    // head: reflect(3) + conv k7 + sigmoid
    x = rpad(m, x, 3);
    x = conv2d(m, x, "model.34", 1, 0);
    x = ggml_sigmoid(ctx, x);

    // composite: pred*mask + img*(1-mask)
    return ggml_add(ctx, ggml_mul(ctx, x, mask), ggml_mul(ctx, img, inv_mask));
}
