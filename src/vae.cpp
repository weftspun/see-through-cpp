#include "vae.h"

#include <cstdio>

ggml_tensor * vae_encode(Model & m, ggml_tensor * x) {
    ggml_context * ctx = m.ctx_g;
    m.gn_groups = 32; m.gn_eps = 1e-6f; m.head_dim = 0;

    ggml_tensor * sample = conv2d(m, x, "encoder.conv_in");

    for (int i = 0; i < 4; i++) {
        char pre[64];
        for (int r = 0; r < 2; r++) {
            snprintf(pre, sizeof(pre), "encoder.down_blocks.%d.resnets.%d", i, r);
            sample = resnet_block(m, sample, pre);
        }
        snprintf(pre, sizeof(pre), "encoder.down_blocks.%d.downsamplers.0.conv", i);
        if (m.has(std::string(pre) + ".weight")) {
            // diffusers Downsample2D pads (0,1,0,1) then convs stride 2 pad 0
            sample = ggml_pad(ctx, sample, 1, 1, 0, 0);
            sample = conv2d(m, sample, pre, 2, 0);
        }
    }

    sample = resnet_block(m, sample, "encoder.mid_block.resnets.0");
    sample = attn_block(m, sample, "encoder.mid_block.attentions.0");
    sample = resnet_block(m, sample, "encoder.mid_block.resnets.1");

    sample = group_norm_affine(m, sample, "encoder.conv_norm_out");
    sample = ggml_silu(ctx, sample);
    sample = conv2d(m, sample, "encoder.conv_out");              // moments, 8ch
    sample = conv2d(m, sample, "quant_conv", 1, 0);              // 1x1
    // posterior mean = first 4 channels of the moments
    return ggml_cont(ctx, ggml_view_4d(ctx, sample, sample->ne[0], sample->ne[1], 4, 1,
                                       sample->nb[1], sample->nb[2], sample->nb[3], 0));
}

ggml_tensor * vae_decode(Model & m, ggml_tensor * z) {
    ggml_context * ctx = m.ctx_g;
    m.gn_groups = 32; m.gn_eps = 1e-6f; m.head_dim = 0;

    ggml_tensor * sample = conv2d(m, z, "post_quant_conv", 1, 0);   // 1x1
    sample = conv2d(m, sample, "decoder.conv_in");
    debug_tap(m, "vae.conv_in", sample);

    sample = resnet_block(m, sample, "decoder.mid_block.resnets.0");
    debug_tap(m, "vae.mid.resnet0", sample);
    sample = attn_block(m, sample, "decoder.mid_block.attentions.0");
    debug_tap(m, "vae.mid.attn", sample);
    sample = resnet_block(m, sample, "decoder.mid_block.resnets.1");
    debug_tap(m, "vae.mid.resnet1", sample);

    for (int i = 0; i < 4; i++) {
        char pre[64];
        for (int r = 0; r < 3; r++) {
            snprintf(pre, sizeof(pre), "decoder.up_blocks.%d.resnets.%d", i, r);
            sample = resnet_block(m, sample, pre);
        }
        char tapname[64];
        snprintf(tapname, sizeof(tapname), "vae.up%d.resnets", i);
        debug_tap(m, tapname, sample);
        snprintf(pre, sizeof(pre), "decoder.up_blocks.%d.upsamplers.0.conv", i);
        if (m.has(std::string(pre) + ".weight")) {
            sample = ggml_upscale(ctx, sample, 2, GGML_SCALE_MODE_NEAREST);
            sample = conv2d(m, sample, pre);
            snprintf(tapname, sizeof(tapname), "vae.up%d.upsample", i);
            debug_tap(m, tapname, sample);
        }
    }

    sample = group_norm_affine(m, sample, "decoder.conv_norm_out");
    sample = ggml_silu(ctx, sample);
    sample = conv2d(m, sample, "decoder.conv_out");
    debug_tap(m, "vae.conv_out", sample);
    return sample;
}

ggml_tensor * unet1024(Model & m, ggml_tensor * x, ggml_tensor * latent) {
    ggml_context * ctx = m.ctx_g;
    m.gn_groups = 4; m.gn_eps = 1e-5f; m.head_dim = 8;
    const std::string P = "decoder.model.";

    ggml_tensor * sample_latent = conv2d(m, latent, P + "latent_conv_in", 1, 0); // 1x1
    ggml_tensor * sample = conv2d(m, x, P + "conv_in");
    debug_tap(m, "u1024.conv_in", sample);

    std::vector<ggml_tensor *> res_stack = { sample };
    const bool down_attn[7] = { false, false, false, false, true, true, true };
    for (int i = 0; i < 7; i++) {
        char pre[64], tapname[64];
        if (i == 3) sample = ggml_add(ctx, sample, sample_latent);
        for (int r = 0; r < 2; r++) {
            snprintf(pre, sizeof(pre), "down_blocks.%d.resnets.%d", i, r);
            sample = resnet_block(m, sample, P + pre);
            snprintf(tapname, sizeof(tapname), "u1024.down%d.resnet%d", i, r);
            debug_tap(m, tapname, sample);
            if (down_attn[i]) {
                snprintf(pre, sizeof(pre), "down_blocks.%d.attentions.%d", i, r);
                sample = attn_block(m, sample, P + pre);
                snprintf(tapname, sizeof(tapname), "u1024.down%d.attn%d", i, r);
                debug_tap(m, tapname, sample);
            }
            res_stack.push_back(sample);
        }
        snprintf(tapname, sizeof(tapname), "u1024.down%d", i);
        debug_tap(m, tapname, sample);
        snprintf(pre, sizeof(pre), "down_blocks.%d.downsamplers.0.conv", i);
        if (m.has(P + pre + ".weight")) {
            sample = conv2d(m, sample, P + pre, 2, 1);
            res_stack.push_back(sample);
            snprintf(tapname, sizeof(tapname), "u1024.down%d.downsample", i);
            debug_tap(m, tapname, sample);
        }
    }

    sample = resnet_block(m, sample, P + "mid_block.resnets.0");
    sample = attn_block(m, sample, P + "mid_block.attentions.0");
    sample = resnet_block(m, sample, P + "mid_block.resnets.1");
    debug_tap(m, "u1024.mid", sample);

    const bool up_attn[7] = { true, true, true, false, false, false, false };
    for (int i = 0; i < 7; i++) {
        char pre[64], tapname[64];
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
        snprintf(tapname, sizeof(tapname), "u1024.up%d", i);
        debug_tap(m, tapname, sample);
        snprintf(pre, sizeof(pre), "up_blocks.%d.upsamplers.0.conv", i);
        if (m.has(P + pre + ".weight")) {
            sample = ggml_upscale(ctx, sample, 2, GGML_SCALE_MODE_NEAREST);
            sample = conv2d(m, sample, P + pre);
            snprintf(tapname, sizeof(tapname), "u1024.up%d.upsample", i);
            debug_tap(m, tapname, sample);
        }
    }

    sample = group_norm_affine(m, sample, P + "conv_norm_out");
    sample = ggml_silu(ctx, sample);
    sample = conv2d(m, sample, P + "conv_out");
    debug_tap(m, "u1024.conv_out", sample);
    return sample;
}

ggml_tensor * trans_vae_decode(Model & m, ggml_tensor * z) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * pixel = vae_decode(m, z);
    pixel = ggml_scale_bias(ctx, pixel, 0.5f, 0.5f);        // [-1,1] -> [0,1]
    ggml_tensor * y = unet1024(m, pixel, z);
    return ggml_clamp(ctx, y, 0.0f, 1.0f);                  // estimate_augmented clip
}
