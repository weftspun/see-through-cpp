#include "vae.h"

#include <algorithm>
#include <cstdio>

ggml_tensor * vae_encode(Model & m, ggml_tensor * x, std::vector<ggml_tensor *> * taps) {
    ggml_context * ctx = m.ctx_g;
    m.gn_groups = 32; m.gn_eps = 1e-6f; m.head_dim = 0;

    ggml_tensor * sample = conv2d(m, x, "encoder.conv_in");
    if (taps) { taps->push_back(sample); }

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
            debug_tap(m, std::string(pre) + "_out", sample);
        }
        if (taps) { taps->push_back(sample); }
    }

    sample = resnet_block(m, sample, "encoder.mid_block.resnets.0");
    if (taps) { taps->push_back(sample); }
    sample = attn_block(m, sample, "encoder.mid_block.attentions.0");
    if (taps) { taps->push_back(sample); }
    sample = resnet_block(m, sample, "encoder.mid_block.resnets.1");
    if (taps) { taps->push_back(sample); }

    sample = group_norm_affine(m, sample, "encoder.conv_norm_out");
    sample = ggml_silu(ctx, sample);
    sample = conv2d(m, sample, "encoder.conv_out");              // moments, 8ch
    sample = conv2d(m, sample, "quant_conv", 1, 0);              // 1x1
    // posterior mean = first 4 channels of the moments
    return ggml_cont(ctx, ggml_view_4d(ctx, sample, sample->ne[0], sample->ne[1], 4, 1,
                                       sample->nb[1], sample->nb[2], sample->nb[3], 0));
}

// blend b's leading `extent` rows (dim=1, height) or cols (dim=0, width)
// with a's trailing `extent` rows/cols, linearly ramping 0->1; matches
// diffusers' blend_v (dim=1)/blend_h (dim=0) exactly.
static ggml_tensor * blend_edge(Model & m, ggml_tensor * a, ggml_tensor * b, int64_t extent, int dim) {
    ggml_context * ctx = m.ctx_g;
    extent = std::min({ a->ne[dim], b->ne[dim], extent });
    if (extent <= 0) { return b; }

    ggml_tensor * w = ggml_scale(ctx, ggml_arange(ctx, 0.0f, (float) extent, 1.0f), 1.0f / (float) extent);
    w = dim == 0 ? ggml_reshape_4d(ctx, w, extent, 1, 1, 1)
                 : ggml_reshape_4d(ctx, w, 1, extent, 1, 1);

    auto slab = [&](ggml_tensor * t, int64_t start, int64_t len) {
        int64_t ne0 = dim == 0 ? len : t->ne[0];
        int64_t ne1 = dim == 1 ? len : t->ne[1];
        size_t off = dim == 0 ? start * t->nb[0] : start * t->nb[1];
        return ggml_cont(ctx, ggml_view_4d(ctx, t, ne0, ne1, t->ne[2], t->ne[3],
                                           t->nb[1], t->nb[2], t->nb[3], off));
    };
    ggml_tensor * a_tail = slab(a, a->ne[dim] - extent, extent);
    ggml_tensor * b_head = slab(b, 0, extent);
    ggml_tensor * blended = ggml_add(ctx,
        ggml_mul(ctx, a_tail, ggml_scale_bias(ctx, w, -1.0f, 1.0f)),  // a_tail * (1 - w)
        ggml_mul(ctx, b_head, w));                                   // + b_head * w
    if (b->ne[dim] == extent) { return blended; }
    ggml_tensor * b_rest = slab(b, extent, b->ne[dim] - extent);
    return ggml_concat(ctx, blended, b_rest, dim);
}

ggml_tensor * vae_encode_tiled(Model & m, ggml_tensor * x) {
    ggml_context * ctx = m.ctx_g;
    const int64_t W = x->ne[0], H = x->ne[1];
    const int64_t TILE_PX     = 512;                 // AutoencoderKL.tile_sample_min_size
    const int64_t OVERLAP_PX  = TILE_PX * 3 / 4;      // tile_overlap_factor = 0.25 -> 384
    const int64_t TILE_LATENT = TILE_PX / 8;          // tile_latent_min_size = 64
    const int64_t BLEND_LAT   = TILE_LATENT / 4;      // 16
    const int64_t ROW_LIMIT   = TILE_LATENT - BLEND_LAT; // 48

    if (H <= TILE_PX && W <= TILE_PX) { return vae_encode(m, x); }

    std::vector<int64_t> row0, col0;
    for (int64_t i = 0; i < H; i += OVERLAP_PX) { row0.push_back(i); }
    for (int64_t j = 0; j < W; j += OVERLAP_PX) { col0.push_back(j); }

    std::vector<std::vector<ggml_tensor *>> tiles(row0.size(), std::vector<ggml_tensor *>(col0.size()));
    for (size_t ti = 0; ti < row0.size(); ti++) {
        for (size_t tj = 0; tj < col0.size(); tj++) {
            const int64_t i = row0[ti], j = col0[tj];
            const int64_t th = std::min(TILE_PX, H - i), tw = std::min(TILE_PX, W - j);
            ggml_tensor * crop = ggml_cont(ctx, ggml_view_4d(ctx, x, tw, th, x->ne[2], x->ne[3],
                                                             x->nb[1], x->nb[2], x->nb[3],
                                                             j * x->nb[0] + i * x->nb[1]));
            tiles[ti][tj] = vae_encode(m, crop);
        }
    }

    ggml_tensor * result = nullptr;
    for (size_t ti = 0; ti < tiles.size(); ti++) {
        ggml_tensor * row_cat = nullptr;
        for (size_t tj = 0; tj < tiles[ti].size(); tj++) {
            ggml_tensor * t = tiles[ti][tj];
            if (ti > 0) { t = blend_edge(m, tiles[ti - 1][tj], t, BLEND_LAT, 1); }
            if (tj > 0) { t = blend_edge(m, tiles[ti][tj - 1], t, BLEND_LAT, 0); }
            const int64_t ch = std::min(t->ne[1], ROW_LIMIT), cw = std::min(t->ne[0], ROW_LIMIT);
            t = ggml_cont(ctx, ggml_view_4d(ctx, t, cw, ch, t->ne[2], t->ne[3],
                                            t->nb[1], t->nb[2], t->nb[3], 0));
            row_cat = row_cat ? ggml_concat(ctx, row_cat, t, 0) : t;
        }
        result = result ? ggml_concat(ctx, result, row_cat, 1) : row_cat;
    }
    return result;
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

ggml_tensor * vae_decode_tiled(Model & m, ggml_tensor * z) {
    ggml_context * ctx = m.ctx_g;
    const int64_t ZW = z->ne[0], ZH = z->ne[1];
    const int64_t TILE_LATENT = 64;                  // tile_latent_min_size
    const int64_t OVERLAP_LAT = TILE_LATENT * 3 / 4;  // tile_overlap_factor=0.25 -> 48
    const int64_t TILE_PX     = 512;                  // tile_sample_min_size
    const int64_t BLEND_PX    = TILE_PX / 4;          // 128
    const int64_t ROW_LIMIT_PX = TILE_PX - BLEND_PX;  // 384

    if (ZH <= TILE_LATENT && ZW <= TILE_LATENT) { return vae_decode(m, z); }

    std::vector<int64_t> row0, col0;
    for (int64_t i = 0; i < ZH; i += OVERLAP_LAT) { row0.push_back(i); }
    for (int64_t j = 0; j < ZW; j += OVERLAP_LAT) { col0.push_back(j); }

    std::vector<std::vector<ggml_tensor *>> tiles(row0.size(), std::vector<ggml_tensor *>(col0.size()));
    for (size_t ti = 0; ti < row0.size(); ti++) {
        for (size_t tj = 0; tj < col0.size(); tj++) {
            const int64_t i = row0[ti], j = col0[tj];
            const int64_t th = std::min(TILE_LATENT, ZH - i), tw = std::min(TILE_LATENT, ZW - j);
            ggml_tensor * crop = ggml_cont(ctx, ggml_view_4d(ctx, z, tw, th, z->ne[2], z->ne[3],
                                                             z->nb[1], z->nb[2], z->nb[3],
                                                             j * z->nb[0] + i * z->nb[1]));
            tiles[ti][tj] = vae_decode(m, crop);
        }
    }

    ggml_tensor * result = nullptr;
    for (size_t ti = 0; ti < tiles.size(); ti++) {
        ggml_tensor * row_cat = nullptr;
        for (size_t tj = 0; tj < tiles[ti].size(); tj++) {
            ggml_tensor * t = tiles[ti][tj];
            if (ti > 0) { t = blend_edge(m, tiles[ti - 1][tj], t, BLEND_PX, 1); }
            if (tj > 0) { t = blend_edge(m, tiles[ti][tj - 1], t, BLEND_PX, 0); }
            const int64_t ch = std::min(t->ne[1], ROW_LIMIT_PX), cw = std::min(t->ne[0], ROW_LIMIT_PX);
            t = ggml_cont(ctx, ggml_view_4d(ctx, t, cw, ch, t->ne[2], t->ne[3],
                                            t->nb[1], t->nb[2], t->nb[3], 0));
            row_cat = row_cat ? ggml_concat(ctx, row_cat, t, 0) : t;
        }
        result = result ? ggml_concat(ctx, result, row_cat, 1) : row_cat;
    }
    return result;
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

// unet1024, tiled: this is a LayerDiffuse-custom architecture (not stock
// diffusers, so no off-the-shelf tiled_* to copy) but its down_blocks 0-2
// each halve the spatial size once before the pixel/latent fusion at i==3
// (`sample = sample + sample_latent`) -- three halvings is exactly the
// pixel->latent 8x downscale, so a pixel tile and its co-located latent
// tile (both from vae_encode_tiled/vae_decode_tiled's own 512px/64-latent
// grid) line up exactly at that fusion point. Running this network
// untiled at full 1280px was the remaining source of visible tile seams
// (from vae_decode_tiled's own residual seam getting smeared/amplified
// through unet1024's convolutions, since nothing here was blending them
// back out) and likely of the missing fine detail (face/ears) -- same
// out-of-trained-scale reasoning as vae_encode_tiled/vae_decode_tiled.
static ggml_tensor * unet1024_tiled(Model & m, ggml_tensor * x, ggml_tensor * latent) {
    ggml_context * ctx = m.ctx_g;
    const int64_t W = x->ne[0], H = x->ne[1];
    const int64_t TILE_PX     = 512;
    const int64_t OVERLAP_PX  = TILE_PX * 3 / 4;      // 384
    const int64_t BLEND_PX    = TILE_PX / 4;          // 128
    const int64_t ROW_LIMIT_PX = TILE_PX - BLEND_PX;  // 384

    if (H <= TILE_PX && W <= TILE_PX) { return unet1024(m, x, latent); }

    std::vector<int64_t> row0, col0;
    for (int64_t i = 0; i < H; i += OVERLAP_PX) { row0.push_back(i); }
    for (int64_t j = 0; j < W; j += OVERLAP_PX) { col0.push_back(j); }

    std::vector<std::vector<ggml_tensor *>> tiles(row0.size(), std::vector<ggml_tensor *>(col0.size()));
    for (size_t ti = 0; ti < row0.size(); ti++) {
        for (size_t tj = 0; tj < col0.size(); tj++) {
            const int64_t i = row0[ti], j = col0[tj];
            const int64_t th = std::min(TILE_PX, H - i), tw = std::min(TILE_PX, W - j);
            ggml_tensor * x_tile = ggml_cont(ctx, ggml_view_4d(ctx, x, tw, th, x->ne[2], x->ne[3],
                                                               x->nb[1], x->nb[2], x->nb[3],
                                                               j * x->nb[0] + i * x->nb[1]));
            const int64_t li = i / 8, lj = j / 8, lh = th / 8, lw = tw / 8;
            ggml_tensor * lat_tile = ggml_cont(ctx, ggml_view_4d(ctx, latent, lw, lh, latent->ne[2], latent->ne[3],
                                                                 latent->nb[1], latent->nb[2], latent->nb[3],
                                                                 lj * latent->nb[0] + li * latent->nb[1]));
            tiles[ti][tj] = unet1024(m, x_tile, lat_tile);
        }
    }

    ggml_tensor * result = nullptr;
    for (size_t ti = 0; ti < tiles.size(); ti++) {
        ggml_tensor * row_cat = nullptr;
        for (size_t tj = 0; tj < tiles[ti].size(); tj++) {
            ggml_tensor * t = tiles[ti][tj];
            if (ti > 0) { t = blend_edge(m, tiles[ti - 1][tj], t, BLEND_PX, 1); }
            if (tj > 0) { t = blend_edge(m, tiles[ti][tj - 1], t, BLEND_PX, 0); }
            const int64_t ch = std::min(t->ne[1], ROW_LIMIT_PX), cw = std::min(t->ne[0], ROW_LIMIT_PX);
            t = ggml_cont(ctx, ggml_view_4d(ctx, t, cw, ch, t->ne[2], t->ne[3],
                                            t->nb[1], t->nb[2], t->nb[3], 0));
            row_cat = row_cat ? ggml_concat(ctx, row_cat, t, 0) : t;
        }
        result = result ? ggml_concat(ctx, result, row_cat, 1) : row_cat;
    }
    return result;
}

ggml_tensor * trans_vae_decode(Model & m, ggml_tensor * z) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * pixel = vae_decode_tiled(m, z);
    pixel = ggml_scale_bias(ctx, pixel, 0.5f, 0.5f);        // [-1,1] -> [0,1]
    ggml_tensor * y = unet1024_tiled(m, pixel, z);
    return ggml_clamp(ctx, y, 0.0f, 1.0f);                  // estimate_augmented clip
}
