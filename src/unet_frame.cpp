#include "unet_frame.h"

#include <cmath>

ggml_tensor * attn_tokens(Model & m, ggml_tensor * q_src, ggml_tensor * kv_src,
                          const std::string & pre, int n_head) {
    ggml_context * ctx = m.ctx_g;
    const int64_t C  = q_src->ne[0], Tq = q_src->ne[1], B = q_src->ne[2];
    const int64_t Tk = kv_src->ne[1];
    const int64_t hd = C / n_head;

    ggml_tensor * q = linear(m, q_src, pre + ".to_q");
    ggml_tensor * k = linear(m, kv_src, pre + ".to_k");
    ggml_tensor * v = linear(m, kv_src, pre + ".to_v");
    q = ggml_scale(ctx, q, 1.0f / sqrtf((float) hd));

    q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, q, hd, n_head, Tq, B), 0, 2, 1, 3)); // (hd,Tq,H,B)
    k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, k, hd, n_head, Tk, B), 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, hd, n_head, Tk, B), 1, 2, 0, 3)); // (Tk,hd,H,B)

    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);                        // (Tk,Tq,H,B)
    kq = ggml_soft_max(ctx, kq);
    ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);                      // (hd,Tq,H,B)
    kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));          // (hd,H,Tq,B)
    kqv = ggml_reshape_3d(ctx, kqv, C, Tq, B);
    return linear(m, kqv, pre + ".to_out.0");
}

ggml_tensor * geglu_ff(Model & m, ggml_tensor * x, const std::string & pre) {
    ggml_context * ctx = m.ctx_g;
    // diffusers GEGLU is hidden * gelu(gate) with the gate in the SECOND half
    // of the projection — ggml's unswapped variant activates the first half
    x = linear(m, x, pre + ".net.0.proj");
    x = ggml_geglu_erf_swapped(ctx, x);
    return linear(m, x, pre + ".net.2");
}

ggml_tensor * basic_transformer_block(Model & m, ggml_tensor * x, ggml_tensor * ehs,
                                      const std::string & pre, int n_head) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * h = layer_norm_affine(m, x, pre + ".norm1");
    x = ggml_add(ctx, x, attn_tokens(m, h, h, pre + ".attn1", n_head));
    h = layer_norm_affine(m, x, pre + ".norm2");
    x = ggml_add(ctx, x, attn_tokens(m, h, ehs, pre + ".attn2", n_head));
    h = layer_norm_affine(m, x, pre + ".norm3");
    return ggml_add(ctx, x, geglu_ff(m, h, pre + ".ff"));
}

ggml_tensor * cross_frame_block(Model & m, ggml_tensor * x, const std::string & pre,
                                int n_head) {
    ggml_context * ctx = m.ctx_g;
    // (C, S, F) -> (C, F, S): attention runs across frames, batched over tokens
    ggml_tensor * h = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));

    ggml_tensor * res = h;
    h = layer_norm_affine(m, h, pre + ".norm_in");
    h = geglu_ff(m, h, pre + ".ff_in");
    h = ggml_add(ctx, h, res);                                   // is_res (dim ==
                                                                 // time_mix_inner_dim)
    ggml_tensor * n1 = layer_norm_affine(m, h, pre + ".norm1");
    h = ggml_add(ctx, h, attn_tokens(m, n1, n1, pre + ".attn1", n_head));

    // final FF replaces the hidden state (no residual; the caller adds the
    // whole block output to the spatial stream)
    h = geglu_ff(m, layer_norm_affine(m, h, pre + ".norm3"), pre + ".ff");

    return ggml_cont(ctx, ggml_permute(ctx, h, 0, 2, 1, 3));     // back to (C, S, F)
}

ggml_tensor * transformer3d(Model & m, ggml_tensor * x, ggml_tensor * ehs,
                            const std::string & pre, int n_head, int n_layers) {
    ggml_context * ctx = m.ctx_g;
    const int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], F = x->ne[3];
    ggml_tensor * residual = x;

    const int gg = m.gn_groups; const float ge = m.gn_eps;       // UNet resnets use
    m.gn_groups = 32; m.gn_eps = 1e-6f;                          // 32/1e-6 here
    ggml_tensor * h = group_norm_affine(m, x, pre + ".norm");
    m.gn_groups = gg; m.gn_eps = ge;

    h = ggml_reshape_3d(ctx, h, W * H, C, F);
    h = ggml_cont(ctx, ggml_permute(ctx, h, 1, 0, 2, 3));        // (C, S, F)
    h = linear(m, h, pre + ".proj_in");

    const int stride = n_layers >= 3 ? 2 : 1;
    int n_temporal = 0;
    for (int l = 0; l < n_layers; l++) {
        h = basic_transformer_block(m, h, ehs,
                pre + ".transformer_blocks." + std::to_string(l), n_head);
        if ((l + 1) % stride == 0) {
            ggml_tensor * mix = cross_frame_block(m, h,
                    pre + ".temporal_transformer_blocks." + std::to_string(n_temporal++), n_head);
            h = ggml_add(ctx, h, mix);
        }
    }

    h = linear(m, h, pre + ".proj_out");
    h = ggml_cont(ctx, ggml_permute(ctx, h, 1, 0, 2, 3));        // (S, C, F)
    h = ggml_reshape_4d(ctx, h, W, H, C, F);
    return ggml_add(ctx, h, residual);
}

ggml_tensor * time_embed_mlp(Model & m, ggml_tensor * t_emb, const std::string & pre) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * h = linear(m, t_emb, pre + ".linear_1");
    return linear(m, ggml_silu(ctx, h), pre + ".linear_2");
}

ggml_tensor * group_embedding(Model & m, ggml_tensor * x, const std::string & pre) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * params = m.get(pre + ".params");               // (C, n_cls)
    ggml_tensor * p = ggml_cast(ctx, params, GGML_TYPE_F32);
    if (ggml_n_dims(x) == 3) {                                   // (C, T, F): row f
        p = ggml_reshape_3d(ctx, p, p->ne[0], 1, p->ne[1]);
    }                                                            // broadcasts over T
    return linear(m, ggml_add(ctx, x, p), pre + ".linear");
}

// Transformer3D at `pre` if attention weights exist there: probes the number
// of transformer layers, heads = C/64
static ggml_tensor * maybe_t3d(Model & m, ggml_tensor * x, ggml_tensor * ehs,
                               const std::string & pre) {
    if (!m.has(pre + ".norm.weight")) return x;
    int n_layers = 0;
    while (m.has(pre + ".transformer_blocks." + std::to_string(n_layers) + ".norm1.weight")) {
        n_layers++;
    }
    return transformer3d(m, x, ehs, pre, (int) (x->ne[2] / 64), n_layers);
}

ggml_tensor * unet_frame_forward(Model & m, ggml_tensor * sample, ggml_tensor * emb,
                                 ggml_tensor * ehs,
                                 std::vector<ggml_tensor *> * taps) {
    ggml_context * ctx = m.ctx_g;
    m.gn_groups = 32; m.gn_eps = 1e-5f;

    sample = conv2d(m, sample, "conv_in");
    if (taps) taps->push_back(sample);

    std::vector<ggml_tensor *> res_stack = { sample };
    char pre[96];
    for (int i = 0; ; i++) {
        snprintf(pre, sizeof(pre), "down_blocks.%d", i);
        if (!m.has(std::string(pre) + ".resnets.0.norm1.weight")) break;
        for (int r = 0; ; r++) {
            snprintf(pre, sizeof(pre), "down_blocks.%d.resnets.%d", i, r);
            if (!m.has(std::string(pre) + ".norm1.weight")) break;
            sample = resnet_block(m, sample, pre, emb);
            snprintf(pre, sizeof(pre), "down_blocks.%d.attentions.%d", i, r);
            sample = maybe_t3d(m, sample, ehs, pre);
            res_stack.push_back(sample);
        }
        snprintf(pre, sizeof(pre), "down_blocks.%d.downsamplers.0.conv", i);
        if (m.has(std::string(pre) + ".weight")) {
            sample = conv2d(m, sample, pre, 2, 1);
            res_stack.push_back(sample);
        }
        if (taps) taps->push_back(sample);
    }

    sample = resnet_block(m, sample, "mid_block.resnets.0", emb);
    sample = maybe_t3d(m, sample, ehs, "mid_block.attentions.0");
    sample = resnet_block(m, sample, "mid_block.resnets.1", emb);
    if (taps) taps->push_back(sample);

    for (int i = 0; ; i++) {
        snprintf(pre, sizeof(pre), "up_blocks.%d", i);
        if (!m.has(std::string(pre) + ".resnets.0.norm1.weight")) break;
        for (int r = 0; ; r++) {
            snprintf(pre, sizeof(pre), "up_blocks.%d.resnets.%d", i, r);
            if (!m.has(std::string(pre) + ".norm1.weight")) break;
            ggml_tensor * res = res_stack.back(); res_stack.pop_back();
            sample = ggml_concat(ctx, sample, res, 2);
            sample = resnet_block(m, sample, pre, emb);
            snprintf(pre, sizeof(pre), "up_blocks.%d.attentions.%d", i, r);
            sample = maybe_t3d(m, sample, ehs, pre);
        }
        snprintf(pre, sizeof(pre), "up_blocks.%d.upsamplers.0.conv", i);
        if (m.has(std::string(pre) + ".weight")) {
            sample = ggml_upscale(ctx, sample, 2, GGML_SCALE_MODE_NEAREST);
            sample = conv2d(m, sample, pre);
        }
    }

    sample = group_norm_affine(m, sample, "conv_norm_out");
    sample = ggml_silu(ctx, sample);
    return conv2d(m, sample, "conv_out");
}

ggml_tensor * sdxl_add_embed(Model & m, ggml_tensor * text_embeds, ggml_tensor * time_ids) {
    ggml_context * ctx = m.ctx_g;
    const int64_t F = time_ids->ne[1];
    ggml_tensor * flat = ggml_reshape_1d(ctx, time_ids, 6 * F);
    ggml_tensor * te = ggml_timestep_embedding(ctx, flat, 256, 10000);   // (256, 6F)
    te = ggml_reshape_2d(ctx, te, 256 * 6, F);
    ggml_tensor * cat = ggml_concat(ctx, text_embeds, te, 0);            // (2816, F)
    return time_embed_mlp(m, cat, "add_embedding");
}
