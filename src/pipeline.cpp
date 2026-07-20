#include "pipeline.h"

#include "clip.h"
#include "image_utils.h"
#include "lama.h"
#include "scheduler.h"
#include "unet_frame.h"
#include "vae.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>

const std::vector<std::string> BODY_TAGS_V3 = {
    "front hair", "back hair", "head", "neck", "neckwear", "topwear", "handwear",
    "bottomwear", "legwear", "footwear", "tail", "wings", "objects"
};
const std::vector<std::string> HEAD_TAGS_V3 = {
    "headwear", "face", "irides", "eyebrow", "eyewhite", "eyelash", "eyewear",
    "ears", "earwear", "nose", "mouth"
};
const std::vector<std::string> VALID_BODY_PARTS_V2 = {
    "hair", "headwear", "face", "eyes", "eyewear", "ears", "earwear", "nose", "mouth",
    "neck", "neckwear", "topwear", "handwear", "bottomwear", "legwear", "footwear",
    "tail", "wings", "objects"
};

static std::string kv_by_suffix(const Model & m, const char * suf) {
    for (const auto & kv : m.config_json) {
        size_t n = strlen(suf);
        if (kv.first.size() > n && kv.first.compare(kv.first.size() - n, n, suf) == 0) {
            return kv.second;
        }
    }
    return "";
}

static ggml_backend_dev_t pipe_gpu(const PipelineConfig & cfg) {
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
}

static ggml_backend_t pipe_backend(const PipelineConfig & cfg) {
    if (cfg.device == "cpu") {
        // CPU route is blocklisted -- GPU (Vulkan) only. This isn't just
        // "no silent fallback" anymore: --device cpu is rejected outright,
        // same as any other unsupported --device value.
        fprintf(stderr, "error: --device cpu is not supported -- this build is GPU"
                        "-only (Vulkan). Run without --device (auto-selects the "
                        "first GPU) or pass --device vulkan.\n");
        exit(1);
    }
    ggml_backend_dev_t d = pipe_gpu(cfg);
    if (d) return ggml_backend_dev_init(d, nullptr);
    fprintf(stderr, "error: no GPU device found (Vulkan) -- this build is GPU-only, "
                    "there is no CPU fallback.\n");
    exit(1);
}

static bool pipe_load(const PipelineConfig & cfg, Model & m, const std::string & path,
                      bool unet = false) {
    ggml_backend_dev_t d = pipe_gpu(cfg);
    if (d) {
        if (unet) {
            m.flash_attn = !getenv("SEETHROUGH_NO_FLASH_UNET");  // naive 80x80 spatial attention is ~21GB at 1280px
                                    // VRAM; NOT for VAEs: ggml_conv_2d_direct is
                                    // wrong for the encoder s2/p0 downsample path
            // row-chunked im2col ahead of direct_conv for the stride-1 k3
            // resnet convs: suspected real-weight-specific
            // ggml_conv_2d_direct defect at exactly the 160x160/res=1280
            // latent level (docs/ggml-upstream-issues.md #4) -- row-chunk
            // is Lean-witness-validated exact and, unlike plain im2col,
            // doesn't reintroduce the ~5.75GB per-level transient that
            // direct_conv was avoiding (tiled 8-way, batch(frames)-
            // generic). min_hw covers all 3 UNet latent levels (as low as
            // 40x40). direct_conv stays on as the fallback for the
            // stride-2 downsampler convs, which conv_row_chunk doesn't
            // handle.
            m.direct_conv = true;
            m.conv_row_chunk = !getenv("SEETHROUGH_NO_ROWCHUNK_UNET");
            m.conv_row_chunk_min_hw = 40 * 40;
        }
        // Query-tiled naive attention, VRAM-bounded so it can actually run
        // at production token counts (unlike a plain flash_attn=false
        // toggle, which OOMs). Applies to both attn_tokens (diffusion UNet,
        // ahead of flash_attn) and attn_block (VAE mid_block/unet1024
        // spatial self-attention, always-naive otherwise, unguarded at any
        // T) -- not gated on `unet` since attn_block matters for the VAE
        // model. Part of the confirmed 1280px-collapse fix (docs/ggml-
        // upstream-issues.md #4): unet1024's first spatial self-attention
        // (head_dim=8, 32 heads, T=6400) overflows Vulkan's ~4.295GB
        // maxStorageBufferRange in the plain naive path, corrupting output
        // silently -- confirmed exact against independent CPU ground truth
        // when tiled. On by default now; SEETHROUGH_NO_TILED_ATTN reverts
        // to the plain (defective, at this shape) path for A/B testing.
        m.tiled_naive_attn = !getenv("SEETHROUGH_NO_TILED_ATTN");
        return m.load_backend(path.c_str(), ggml_backend_dev_buffer_type(d));
    }
    return m.load(path.c_str());
}

// one-shot graph: build outputs, set inputs, compute, read all outputs, free
template <typename Build, typename SetInputs>
static bool run_graph_dev(const PipelineConfig & cfg, Model & m, size_t max_nodes,
                          Build build, SetInputs set_inputs,
                          std::vector<std::vector<float>> & outs) {
    size_t meta = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    ggml_init_params ip = { meta, nullptr, true };
    m.ctx_g = ggml_init(ip);
    std::vector<ggml_tensor *> out_t = build();
    for (ggml_tensor * t : out_t) ggml_set_output(t);
    ggml_backend_t backend = pipe_backend(cfg);
    ggml_cgraph * gf = ggml_new_graph_custom(m.ctx_g, max_nodes, false);
    for (ggml_tensor * t : out_t) ggml_build_forward_expand(gf, t);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool ok = ggml_gallocr_alloc_graph(alloc, gf);
    if (ok) {
        set_inputs();
        ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        outs.resize(out_t.size());
        for (size_t i = 0; i < out_t.size(); i++) {
            outs[i].resize(ggml_nelements(out_t[i]));
            ggml_backend_tensor_get(out_t[i], outs[i].data(), 0, outs[i].size() * 4);
        }
    }
    ggml_gallocr_free(alloc);
    ggml_backend_free(backend);
    ggml_free(m.ctx_g);
    m.ctx_g = nullptr;
    return ok;
}

template <typename Build, typename SetInputs>
static bool run_graph(Model & m, size_t max_nodes, int threads, Build build,
                      SetInputs set_inputs, std::vector<std::vector<float>> & outs) {
    size_t meta = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    ggml_init_params ip = { meta, nullptr, true };
    m.ctx_g = ggml_init(ip);
    std::vector<ggml_tensor *> out_t = build();
    for (ggml_tensor * t : out_t) ggml_set_output(t);
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, threads);
    ggml_cgraph * gf = ggml_new_graph_custom(m.ctx_g, max_nodes, false);
    for (ggml_tensor * t : out_t) ggml_build_forward_expand(gf, t);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool ok = ggml_gallocr_alloc_graph(alloc, gf);
    if (ok) {
        set_inputs();
        ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        outs.resize(out_t.size());
        for (size_t i = 0; i < out_t.size(); i++) {
            outs[i].resize(ggml_nelements(out_t[i]));
            ggml_backend_tensor_get(out_t[i], outs[i].data(), 0, outs[i].size() * 4);
        }
    }
    ggml_gallocr_free(alloc);
    ggml_backend_free(backend);
    ggml_free(m.ctx_g);
    m.ctx_g = nullptr;
    return ok;
}

// ---------------------------------------------------------------------------
// stage 1: text encoding
// ---------------------------------------------------------------------------

bool encode_tags(const PipelineConfig & cfg, const std::vector<std::string> & tags,
                 std::vector<float> & ehs, std::vector<float> & pooled) {
    const int F = (int) tags.size();
    ehs.assign((size_t) F * 77 * 2048, 0.0f);
    pooled.assign((size_t) F * 1280, 0.0f);

    for (int enc = 0; enc < 2; enc++) {
        Model m;
        std::string path = cfg.model_dir + (enc == 0 ? "/layerdiff-te1.gguf" : "/layerdiff-te2.gguf");
        if (!pipe_load(cfg, m, path)) { fprintf(stderr, "failed to load %s\n", path.c_str()); return false; }
        ClipTokenizer tok;
        if (!tok.load(kv_by_suffix(m, ".vocab_json"), kv_by_suffix(m, ".merges_txt"))) return false;
        ClipParams p = clip_params_from_config(kv_by_suffix(m, ".config_json"));
        const int32_t pad_id = enc == 0 ? tok.eos_id : 0;
        const int d = p.d_model;
        const int off = enc == 0 ? 0 : 768;    // concat slot in the 2048 dim

        for (int f = 0; f < F; f++) {
            int eos_pos = 0;
            std::vector<int32_t> ids = tok.encode_padded(tags[f], 77, pad_id, &eos_pos);
            ggml_tensor * ids_t = nullptr;
            std::vector<std::vector<float>> outs;
            bool ok = run_graph_dev(cfg, m, 8192,
                [&]() {
                    ids_t = ggml_new_tensor_1d(m.ctx_g, GGML_TYPE_I32, 77);
                    ggml_set_input(ids_t);
                    ggml_tensor * penult = nullptr, * final_out = nullptr;
                    clip_text_graph(m, ids_t, p, &penult, &final_out);
                    std::vector<ggml_tensor *> res = { penult };
                    if (enc == 1) res.push_back(clip_pooled_projection(m, final_out, eos_pos));
                    return res;
                },
                [&]() { ggml_backend_tensor_set(ids_t, ids.data(), 0, 77 * 4); },
                outs);
            if (!ok) return false;
            for (int t = 0; t < 77; t++) {
                std::copy(outs[0].begin() + (size_t) t * d, outs[0].begin() + (size_t) (t + 1) * d,
                          ehs.begin() + ((size_t) f * 77 + t) * 2048 + off);
            }
            if (enc == 1) {
                std::copy(outs[1].begin(), outs[1].end(), pooled.begin() + (size_t) f * 1280);
            }
        }
        if (cfg.verbose) printf("[clip] encoder %d done (%d tags)\n", enc + 1, F);
    }
    return true;
}

// ---------------------------------------------------------------------------
// stage 2: layerdiff diffusion pass
// ---------------------------------------------------------------------------

bool layerdiff_pass(const PipelineConfig & cfg, const Image & page_rgb,
                    const std::vector<float> & ehs, const std::vector<float> & pooled,
                    int group_index, std::vector<Image> & layers_out) {
    const int RES = page_rgb.w, ZR = RES / 8;
    const int F = (int) pooled.size() / 1280;
    const float SCALE = 0.13025f;
    const size_t DZ = (size_t) ZR * ZR * 4, D = DZ * F;

    // page latent (SDXL VAE, f32 weights)
    std::vector<float> c_concat;
    {
        Model mv;
        std::string p = cfg.model_dir + "/layerdiff-vae.gguf";
        if (!pipe_load(cfg, mv, p)) { fprintf(stderr, "failed to load %s\n", p.c_str()); return false; }
        std::vector<float> feed((size_t) RES * RES * 3);
        for (size_t i = 0; i < (size_t) RES * RES; i++) {
            for (int c = 0; c < 3; c++) {
                feed[(size_t) c * RES * RES + i] = page_rgb.data[i * page_rgb.c + c] * 2.0f - 1.0f;
            }
        }
        ggml_tensor * x = nullptr;
        std::vector<std::vector<float>> outs;
        // vae_encode_tiled: this VAE's own trained config is sample_size=512;
        // encoding untiled far beyond that (e.g. 1280) pushes some resnets
        // into extreme near-cancelling activations that amplify ordinary
        // GPU float rounding into a visible defect (docs/ggml-upstream-
        // issues.md #4). Tiling matches diffusers' own enable_tiling()
        // behavior and keeps every tile within the trained scale; passes
        // through to plain vae_encode when RES already fits one tile.
        // Needs a much larger node budget: 16 tiles at res=1280, each
        // running the full encoder graph.
        bool ok = run_graph_dev(cfg, mv, 98304,
            [&]() {
                x = ggml_new_tensor_4d(mv.ctx_g, GGML_TYPE_F32, RES, RES, 3, 1);
                ggml_set_input(x);
                return std::vector<ggml_tensor *>{ ggml_scale(mv.ctx_g, vae_encode_tiled(mv, x), SCALE) };
            },
            [&]() { ggml_backend_tensor_set(x, feed.data(), 0, feed.size() * 4); },
            outs);
        if (!ok) return false;
        c_concat = std::move(outs[0]);
        if (!cfg.debug_dir.empty()) {
            double s = 0, s2 = 0;
            for (float v : c_concat) { s += v; s2 += (double) v * v; }
            double mu = s / c_concat.size();
            printf("[debug] c_concat mean=%.4f std=%.4f\n", mu,
                   sqrt(std::max(0.0, s2 / c_concat.size() - mu * mu)));
        }
        if (cfg.verbose) printf("[layerdiff] page latent done\n");
    }

    // noise: same init across frames + per-step SDE noise (own RNG,
    // statistically equivalent to upstream's)
    std::mt19937_64 rng(cfg.seed + group_index);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    std::vector<float> lat(D);
    {
        std::vector<float> init(DZ);
        for (float & v : init) v = nrm(rng);
        for (int f = 0; f < F; f++) std::copy(init.begin(), init.end(), lat.begin() + f * DZ);
    }

    DpmSolverSDE sch;
    sch.set_timesteps(cfg.steps);

    {
        Model m;
        std::string p = cfg.model_dir + "/layerdiff-unet.gguf";
        if (!pipe_load(cfg, m, p, true)) { fprintf(stderr, "failed to load %s\n", p.c_str()); return false; }

        size_t max_nodes = 98304;
        size_t meta = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
        ggml_init_params ip = { meta, nullptr, true };
        m.ctx_g = ggml_init(ip);
        ggml_context * ctx = m.ctx_g;

        ggml_tensor * sample = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ZR, ZR, 8, F);
        ggml_tensor * ehs_t  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 77, F);
        ggml_tensor * text   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1280, F);
        ggml_tensor * tids   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6, F);
        ggml_tensor * ts     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, F);
        for (ggml_tensor * t : { sample, ehs_t, text, tids, ts }) ggml_set_input(t);

        const std::string gi = std::to_string(group_index);
        ggml_tensor * ehs2 = ggml_add(ctx, ehs_t, group_embedding(m, ehs_t, "group_embeds2." + gi));
        ggml_tensor * emb = time_embed_mlp(m, ggml_timestep_embedding(ctx, ts, 320, 10000),
                                           "time_embedding");
        ggml_tensor * aug = ggml_add(ctx, text, group_embedding(m, text, "group_embeds." + gi));
        emb = ggml_add(ctx, emb, sdxl_add_embed(m, aug, tids));
        ggml_tensor * out = unet_frame_forward(m, sample, emb, ehs2);
        ggml_set_output(out);

        ggml_backend_t backend = pipe_backend(cfg);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, max_nodes, false);
        ggml_build_forward_expand(gf, out);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(alloc, gf)) { fprintf(stderr, "alloc failed\n"); return false; }

        std::vector<float> tid_v(6 * F), eps(D), input((size_t) ZR * ZR * 8 * F), noise(D);
        for (int f = 0; f < F; f++) {
            const float ids[6] = { (float) RES, (float) RES, 0, 0, (float) RES, (float) RES };
            for (int i = 0; i < 6; i++) tid_v[f * 6 + i] = ids[i];
        }

        for (int s = 0; s < cfg.steps; s++) {
            for (int f = 0; f < F; f++) {
                std::copy(lat.begin() + f * DZ, lat.begin() + (f + 1) * DZ,
                          input.begin() + f * 2 * DZ);
                std::copy(c_concat.begin(), c_concat.end(), input.begin() + f * 2 * DZ + DZ);
            }
            // gallocr recycles input buffers: re-set ALL inputs every compute
            ggml_backend_tensor_set(ehs_t, ehs.data(), 0, ehs.size() * 4);
            ggml_backend_tensor_set(text, pooled.data(), 0, pooled.size() * 4);
            ggml_backend_tensor_set(tids, tid_v.data(), 0, tid_v.size() * 4);
            ggml_backend_tensor_set(sample, input.data(), 0, input.size() * 4);
            std::vector<float> tstep(F, (float) sch.timesteps[s]);
            ggml_backend_tensor_set(ts, tstep.data(), 0, F * 4);

            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) return false;
            ggml_backend_tensor_get(out, eps.data(), 0, D * 4);
            for (float & v : noise) v = nrm(rng);
            sch.step(lat, eps, noise);
            if (!cfg.debug_dir.empty()) {
                double acc = 0, acc2 = 0;
                for (float v : lat) { acc += v; acc2 += (double) v * v; }
                double mu = acc / lat.size();
                printf("[debug] step %d eps0=%.4f lat mean=%.4f std=%.4f\n", s,
                       (double) eps[0], mu, sqrt(std::max(0.0, acc2 / lat.size() - mu * mu)));
            }
            if (cfg.verbose) { printf("[layerdiff] step %d/%d (t=%d)\r", s + 1, cfg.steps, sch.timesteps[s]); fflush(stdout); }
        }
        if (cfg.verbose) printf("\n");
        ggml_gallocr_free(alloc);
        ggml_backend_free(backend);
        ggml_free(ctx);
        m.ctx_g = nullptr;
    }

    // decode each frame through the TransparentVAE chain
    {
        Model mv;
        std::string p1 = cfg.model_dir + "/layerdiff-vae.gguf";
        std::string p2 = cfg.model_dir + "/trans-vae.gguf";
        if (!pipe_load(cfg, mv, p1) || !pipe_load(cfg, mv, p2)) return false;
        if (pipe_gpu(cfg)) mv.conv_row_chunk = true;   // exact im2col, row-tiled

        layers_out.assign(F, Image{});
        for (int f = 0; f < F; f++) {
            std::vector<float> z(DZ);
            for (size_t i = 0; i < DZ; i++) z[i] = lat[f * DZ + i] / SCALE;
            ggml_tensor * zt = nullptr;
            std::vector<std::vector<float>> outs;
            // vae_decode_tiled multiplies node count ~16x at res=1280 (16
            // latent tiles, each running the full decoder) -- generous budget.
            bool ok = run_graph_dev(cfg, mv, 131072,
                [&]() {
                    zt = ggml_new_tensor_4d(mv.ctx_g, GGML_TYPE_F32, ZR, ZR, 4, 1);
                    ggml_set_input(zt);
                    ggml_tensor * out = trans_vae_decode(mv, zt);
                    return std::vector<ggml_tensor *>{ out };
                },
                [&]() { ggml_backend_tensor_set(zt, z.data(), 0, z.size() * 4); },
                outs);
            if (!ok) return false;
            // planar RGBA -> interleaved RGBA
            Image & im = layers_out[f];
            im.w = im.h = RES; im.c = 4;
            im.data.resize((size_t) RES * RES * 4);
            const size_t P = (size_t) RES * RES;
            // TransparentVAE output is ARGB planar: ch0 = alpha, ch1..3 = RGB
            for (size_t i = 0; i < P; i++) {
                for (int c = 0; c < 3; c++) im.data[i * 4 + c] = outs[0][(size_t) (c + 1) * P + i];
                im.data[i * 4 + 3] = outs[0][i];
            }
            if (!cfg.debug_dir.empty()) {
                double amax = 0, rmax = 0;
                for (size_t i = 0; i < P; i++) {
                    rmax = std::max(rmax, (double) im.data[i * 4]);
                    amax = std::max(amax, (double) im.data[i * 4 + 3]);
                }
                printf("[debug] frame %d decoded rgb_max=%.4f alpha_max=%.4f\n", f, rmax, amax);
                save_image(cfg.debug_dir + "/frame_" + std::to_string(f) + ".png", im);
            }
            if (cfg.verbose) { printf("[layerdiff] decode %d/%d\r", f + 1, F); fflush(stdout); }
        }
        if (cfg.verbose) printf("\n");
    }
    return true;
}

// ---------------------------------------------------------------------------
// stage 3: Marigold depth
// ---------------------------------------------------------------------------

bool marigold_depth(const PipelineConfig & cfg, const std::vector<Image> & layers_argb,
                    std::vector<Image> & depths_out) {
    const int F = (int) layers_argb.size();
    const int RES = cfg.depth_res, ZR = RES / 8;
    const float SCALE = 0.18215f;
    const size_t DZ = (size_t) ZR * ZR * 4, D = DZ * F;

    // cond latents: page (last) + per-layer, pad_rgb bleed then VAE encode
    std::vector<float> cond((size_t) F * 2 * DZ);
    {
        Model mv;
        std::string p = cfg.model_dir + "/marigold-vae.gguf";
        if (!pipe_load(cfg, mv, p)) { fprintf(stderr, "failed to load %s\n", p.c_str()); return false; }

        auto encode_one = [&](const Image & argb, std::vector<float> & out_lat) -> bool {
            Image im = argb;
            if (im.w != RES || im.h != RES) im = smart_resize(im, RES, RES);
            Image rgb = pad_rgb(im);
            std::vector<float> feed((size_t) RES * RES * 3);
            for (size_t i = 0; i < (size_t) RES * RES; i++) {
                for (int c = 0; c < 3; c++) {
                    feed[(size_t) c * RES * RES + i] = rgb.data[i * 3 + c] * 2.0f - 1.0f;
                }
            }
            ggml_tensor * x = nullptr;
            std::vector<std::vector<float>> outs;
            // vae_encode_tiled: see the layerdiff page-latent comment above;
            // passes through to plain vae_encode when RES fits in one tile
            // (depth_res is usually <=512, but stay consistent/safe if not).
            bool ok = run_graph_dev(cfg, mv, 98304,
                [&]() {
                    x = ggml_new_tensor_4d(mv.ctx_g, GGML_TYPE_F32, RES, RES, 3, 1);
                    ggml_set_input(x);
                    return std::vector<ggml_tensor *>{ ggml_scale(mv.ctx_g, vae_encode_tiled(mv, x), SCALE) };
                },
                [&]() { ggml_backend_tensor_set(x, feed.data(), 0, feed.size() * 4); },
                outs);
            if (ok) out_lat = std::move(outs[0]);
            return ok;
        };

        std::vector<float> page_lat;
        if (!encode_one(layers_argb.back(), page_lat)) return false;
        for (int f = 0; f < F; f++) {
            std::vector<float> ll;
            if (!encode_one(layers_argb[f], ll)) return false;
            std::copy(page_lat.begin(), page_lat.end(), cond.begin() + (size_t) f * 2 * DZ);
            std::copy(ll.begin(), ll.end(), cond.begin() + (size_t) f * 2 * DZ + DZ);
            if (cfg.verbose) { printf("[marigold] cond %d/%d\r", f + 1, F); fflush(stdout); }
        }
        if (cfg.verbose) printf("\n");
    }

    // empty-prompt embedding from marigold-te
    std::vector<float> ehs;
    {
        Model m;
        std::string p = cfg.model_dir + "/marigold-te.gguf";
        if (!pipe_load(cfg, m, p)) { fprintf(stderr, "failed to load %s\n", p.c_str()); return false; }
        ClipTokenizer tok;
        if (!tok.load(kv_by_suffix(m, ".vocab_json"), kv_by_suffix(m, ".merges_txt"))) return false;
        ClipParams p2 = clip_params_from_config(kv_by_suffix(m, ".config_json"));
        std::vector<int32_t> ids = { tok.bos_id, tok.eos_id };
        ggml_tensor * ids_t = nullptr;
        std::vector<std::vector<float>> outs;
        bool ok = run_graph_dev(cfg, m, 8192,
            [&]() {
                ids_t = ggml_new_tensor_1d(m.ctx_g, GGML_TYPE_I32, 2);
                ggml_set_input(ids_t);
                ggml_tensor * final_out = nullptr;
                clip_text_graph(m, ids_t, p2, nullptr, &final_out);
                return std::vector<ggml_tensor *>{ final_out };
            },
            [&]() { ggml_backend_tensor_set(ids_t, ids.data(), 0, 2 * 4); },
            outs);
        if (!ok) return false;
        ehs = std::move(outs[0]);                 // (1024, 2)
    }

    // DDIM loop
    std::mt19937_64 rng(cfg.seed * 31 + 7);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    std::vector<float> lat(D);
    for (float & v : lat) v = nrm(rng);

    DdimTrailing sch;
    sch.set_timesteps(cfg.depth_steps);

    {
        Model m;
        std::string p = cfg.model_dir + "/marigold-unet.gguf";
        if (!pipe_load(cfg, m, p, true)) { fprintf(stderr, "failed to load %s\n", p.c_str()); return false; }
        size_t max_nodes = 98304;
        size_t meta = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
        ggml_init_params ip = { meta, nullptr, true };
        m.ctx_g = ggml_init(ip);
        ggml_context * ctx = m.ctx_g;

        ggml_tensor * sample = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ZR, ZR, 12, F);
        ggml_tensor * ehs_t  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1024, 2, F);
        ggml_tensor * ts     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, F);
        for (ggml_tensor * t : { sample, ehs_t, ts }) ggml_set_input(t);
        ggml_tensor * emb = time_embed_mlp(m, ggml_timestep_embedding(ctx, ts, 320, 10000),
                                           "time_embedding");
        ggml_tensor * out = unet_frame_forward(m, sample, emb, ehs_t);
        ggml_set_output(out);

        ggml_backend_t backend = pipe_backend(cfg);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, max_nodes, false);
        ggml_build_forward_expand(gf, out);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(alloc, gf)) return false;

        std::vector<float> ehs_f((size_t) 1024 * 2 * F), eps(D), input((size_t) ZR * ZR * 12 * F);
        for (int f = 0; f < F; f++) {
            std::copy(ehs.begin(), ehs.end(), ehs_f.begin() + (size_t) f * 2048);
        }
        for (int s = 0; s < cfg.depth_steps; s++) {
            for (int f = 0; f < F; f++) {
                std::copy(cond.begin() + (size_t) f * 2 * DZ, cond.begin() + (size_t) (f + 1) * 2 * DZ,
                          input.begin() + (size_t) f * 3 * DZ);
                std::copy(lat.begin() + f * DZ, lat.begin() + (f + 1) * DZ,
                          input.begin() + (size_t) f * 3 * DZ + 2 * DZ);
            }
            ggml_backend_tensor_set(ehs_t, ehs_f.data(), 0, ehs_f.size() * 4);
            ggml_backend_tensor_set(sample, input.data(), 0, input.size() * 4);
            std::vector<float> tstep(F, (float) sch.timesteps[s]);
            ggml_backend_tensor_set(ts, tstep.data(), 0, F * 4);
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) return false;
            ggml_backend_tensor_get(out, eps.data(), 0, D * 4);
            sch.step(lat, eps, s);
            if (cfg.verbose) { printf("[marigold] step %d/%d\r", s + 1, cfg.depth_steps); fflush(stdout); }
        }
        if (cfg.verbose) printf("\n");
        ggml_gallocr_free(alloc);
        ggml_backend_free(backend);
        ggml_free(ctx);
        m.ctx_g = nullptr;
    }

    // decode -> RGB mean -> [0,1]
    {
        Model mv;
        std::string p = cfg.model_dir + "/marigold-vae.gguf";
        if (!pipe_load(cfg, mv, p)) return false;
        if (pipe_gpu(cfg)) mv.conv_row_chunk = true;   // decode stage only
        depths_out.assign(F, Image{});
        for (int f = 0; f < F; f++) {
            std::vector<float> z(DZ);
            for (size_t i = 0; i < DZ; i++) z[i] = lat[f * DZ + i] / SCALE;
            ggml_tensor * zt = nullptr;
            std::vector<std::vector<float>> outs;
            bool ok = run_graph_dev(cfg, mv, 8192,
                [&]() {
                    zt = ggml_new_tensor_4d(mv.ctx_g, GGML_TYPE_F32, ZR, ZR, 4, 1);
                    ggml_set_input(zt);
                    return std::vector<ggml_tensor *>{ vae_decode_tiled(mv, zt) };
                },
                [&]() { ggml_backend_tensor_set(zt, z.data(), 0, z.size() * 4); },
                outs);
            if (!ok) return false;
            Image & d = depths_out[f];
            d.w = d.h = RES; d.c = 1;
            d.data.resize((size_t) RES * RES);
            const size_t P = (size_t) RES * RES;
            for (size_t i = 0; i < P; i++) {
                float v = (outs[0][i] + outs[0][P + i] + outs[0][2 * P + i]) / 3.0f;
                v = v < -1.0f ? -1.0f : v > 1.0f ? 1.0f : v;
                d.data[i] = (v + 1.0f) / 2.0f;
            }
            if (cfg.verbose) { printf("[marigold] decode %d/%d\r", f + 1, F); fflush(stdout); }
        }
        if (cfg.verbose) printf("\n");
    }
    return true;
}

// ---------------------------------------------------------------------------
// LaMa callback
// ---------------------------------------------------------------------------

InpaintFn make_lama_inpaint(const PipelineConfig & cfg) {
    auto model = std::make_shared<Model>();
    std::string path = cfg.model_dir + "/lama.gguf";
    int threads = cfg.threads;
    return [model, path, threads](const Image & rgb, const std::vector<uint8_t> & mask) -> Image {
        if (model->weights.empty() && !model->load(path.c_str())) {
            fprintf(stderr, "failed to load %s - skipping inpaint\n", path.c_str());
            return rgb;
        }
        // upstream inpaint_preprocess: resize <=1024 stride 64, square pad
        int W = rgb.w, H = rgb.h;
        int side = std::max(W, H);
        int target = side > 1024 ? 1024 : (side + 63) / 64 * 64;
        Image img = rgb;
        Image m4;
        m4.w = W; m4.h = H; m4.c = 1;
        m4.data.resize(mask.size());
        for (size_t i = 0; i < mask.size(); i++) m4.data[i] = mask[i] / 255.0f;
        img = smart_resize(img, target, target);
        m4 = smart_resize(m4, target, target);
        for (float & v : m4.data) v = v < 0.5f ? 0.0f : 1.0f;

        ggml_tensor * xi = nullptr, * xm = nullptr;
        std::vector<std::vector<float>> outs;
        std::vector<float> feed((size_t) target * target * 3), mfeed((size_t) target * target);
        const size_t P = (size_t) target * target;
        for (size_t i = 0; i < P; i++) {
            for (int c = 0; c < 3; c++) feed[(size_t) c * P + i] = img.data[i * 3 + c];
            mfeed[i] = m4.data[i];
        }
        bool ok = run_graph(*model, 8192, threads,
            [&]() {
                xi = ggml_new_tensor_4d(model->ctx_g, GGML_TYPE_F32, target, target, 3, 1);
                xm = ggml_new_tensor_4d(model->ctx_g, GGML_TYPE_F32, target, target, 1, 1);
                ggml_set_input(xi);
                ggml_set_input(xm);
                return std::vector<ggml_tensor *>{ lama_inpaint_graph(*model, xi, xm) };
            },
            [&]() {
                ggml_backend_tensor_set(xi, feed.data(), 0, feed.size() * 4);
                ggml_backend_tensor_set(xm, mfeed.data(), 0, mfeed.size() * 4);
            },
            outs);
        if (!ok) return rgb;
        Image res;
        res.w = res.h = target; res.c = 3;
        res.data.resize(P * 3);
        for (size_t i = 0; i < P; i++) {
            for (int c = 0; c < 3; c++) {
                float v = std::round(outs[0][(size_t) c * P + i] * 255.0f) / 255.0f;
                res.data[i * 3 + c] = std::min(1.0f, std::max(0.0f, v));
            }
        }
        if (target != W || target != H) res = smart_resize(res, W, H);
        // composite only inside the mask
        Image final_img = rgb;
        for (size_t i = 0; i < mask.size(); i++) {
            if (mask[i] >= 127) {
                for (int c = 0; c < 3; c++) final_img.data[i * 3 + c] = res.data[i * 3 + c];
            }
        }
        return final_img;
    };
}

// ---------------------------------------------------------------------------
// full see-through orchestration (extracted from the CLI's main() so it can
// be called from a C ABI / server too, not just argv-driven binaries):
// input image -> per-tag depth-ordered layers -> SVG (+ optional PNG layers)
// ---------------------------------------------------------------------------

// upstream img_alpha_blending (premultiplied=False): sequential "over"
static void blend_over(Image & dst, const Image & src) {
    for (size_t i = 0; i < (size_t) dst.w * dst.h; i++) {
        float a = src.data[i * 4 + 3];
        for (int c = 0; c < 3; c++) {
            dst.data[i * 4 + c] = src.data[i * 4 + c] * a + dst.data[i * 4 + c] * (1 - a);
        }
        dst.data[i * 4 + 3] = a + dst.data[i * 4 + 3] * (1 - a);
    }
}

static void alpha_floor(Image & img) {          // upstream: alpha < 15/255 -> 0
    for (size_t i = 3; i < img.data.size(); i += 4) {
        if (img.data[i] < 15.0f / 255.0f) { img.data[i] = 0.0f; }
    }
}

static bool bbox_alpha(const Image & img, float thr, int * x, int * y, int * w, int * h) {
    int x0 = img.w, y0 = img.h, x1 = -1, y1 = -1;
    for (int yy = 0; yy < img.h; yy++) {
        for (int xx = 0; xx < img.w; xx++) {
            if (img.px(xx, yy)[3] > thr) {
                x0 = std::min(x0, xx); y0 = std::min(y0, yy);
                x1 = std::max(x1, xx); y1 = std::max(y1, yy);
            }
        }
    }
    if (x1 < 0) { return false; }
    *x = x0; *y = y0; *w = x1 - x0 + 1; *h = y1 - y0 + 1;
    return true;
}

// base64 (no line wrapping, standard alphabet)
static std::string b64_encode(const std::vector<uint8_t> & d) {
    static const char * T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o;
    o.reserve((d.size() + 2) / 3 * 4);
    for (size_t i = 0; i < d.size(); i += 3) {
        uint32_t v = d[i] << 16 | (i + 1 < d.size() ? d[i + 1] << 8 : 0)
                   | (i + 2 < d.size() ? d[i + 2] : 0);
        o += T[v >> 18]; o += T[(v >> 12) & 63];
        o += i + 1 < d.size() ? T[(v >> 6) & 63] : '=';
        o += i + 2 < d.size() ? T[v & 63] : '=';
    }
    return o;
}

// semantic identifier shared by --png-dir filenames and the SVG's per-layer
// <image id="..."> attributes (e.g. ThorVG lookup by id)
static std::string safe_id(const std::string & tag) {
    std::string s = tag;
    std::replace(s.begin(), s.end(), ' ', '_');
    return s;
}

bool run_see_through(const PipelineConfig & cfg, const Image & input, SeeThroughResult & result) {
    if (cfg.device == "cpu") {
        // Fail before any (multi-GB) model loading happens, not partway
        // through the first graph compute -- see pipe_backend().
        fprintf(stderr, "error: --device cpu is not supported -- this build is GPU"
                        "-only (Vulkan). Run without --device (auto-selects the "
                        "first GPU) or pass --device vulkan.\n");
        return false;
    }
    const int RES = cfg.res;
    int pad_w, pad_h, pad_x, pad_y;
    Image fullpage = center_square_pad_resize(input, RES, 0.0f, &pad_w, &pad_h, &pad_x, &pad_y);
    const double scale = (double) pad_w / RES;

    // page RGB (alpha treated as 255) + page alpha for masking decoded layers
    Image page_rgb;
    page_rgb.w = page_rgb.h = RES; page_rgb.c = 3;
    page_rgb.data.resize((size_t) RES * RES * 3);
    std::vector<float> page_alpha((size_t) RES * RES);
    for (size_t i = 0; i < (size_t) RES * RES; i++) {
        for (int c = 0; c < 3; c++) { page_rgb.data[i * 3 + c] = fullpage.data[i * 4 + c]; }
        page_alpha[i] = fullpage.data[i * 4 + 3];
    }

    if (!cfg.debug_dir.empty()) {
        Image dbg; dbg.w = dbg.h = RES; dbg.c = 1;
        dbg.data = page_alpha;
        std::ofstream f(cfg.debug_dir + "/page_alpha.png", std::ios::binary);
        std::vector<uint8_t> png = encode_png(dbg);
        f.write(reinterpret_cast<const char *>(png.data()), (std::streamsize) png.size());
        int rows_nonzero = 0;
        for (int y = 0; y < RES; y++) {
            bool any = false;
            for (int x = 0; x < RES; x++) {
                if (page_alpha[(size_t) y * RES + x] > 0.5f) { any = true; break; }
            }
            if (any) { rows_nonzero++; }
        }
        printf("[debug] page_alpha: %d/%d rows have any nonzero alpha (pad_w=%d pad_h=%d pad_x=%d pad_y=%d)\n",
               rows_nonzero, RES, pad_w, pad_h, pad_x, pad_y);
    }

    // ---- layerdiff pass 1: body ----
    std::vector<float> ehs, pooled;
    if (!encode_tags(cfg, BODY_TAGS_V3, ehs, pooled)) { return false; }
    std::vector<Image> body_layers;
    if (!layerdiff_pass(cfg, page_rgb, ehs, pooled, 0, body_layers)) { return false; }

    std::vector<float> pre_mul_alpha5;   // snapshot before the page_alpha multiply
    if (!cfg.debug_dir.empty() && body_layers.size() > 5) {
        Image & l = body_layers[5];
        pre_mul_alpha5.resize((size_t) l.w * l.h);
        for (size_t i = 0; i < pre_mul_alpha5.size(); i++) { pre_mul_alpha5[i] = l.data[i * 4 + 3]; }
    }

    for (Image & l : body_layers) {
        for (size_t i = 0; i < page_alpha.size(); i++) { l.data[i * 4 + 3] *= page_alpha[i]; }
    }

    if (!cfg.debug_dir.empty() && body_layers.size() > 5) {
        Image & l = body_layers[5];   // topwear, pre-alpha_floor
        printf("[debug] body_layers[5] (topwear), w=%d h=%d, page_alpha.size=%zu\n",
               l.w, l.h, page_alpha.size());
        for (int y = 0; y < l.h; y += (l.h / 20 > 0 ? l.h / 20 : 1)) {
            float pre_max = 0, pa_max = 0, post_max = 0;
            for (int x = 0; x < l.w; x++) {
                size_t i = (size_t) y * l.w + x;
                pre_max = std::max(pre_max, pre_mul_alpha5[i]);
                pa_max = std::max(pa_max, page_alpha[i]);
                post_max = std::max(post_max, l.px(x, y)[3]);
            }
            printf("[debug]   row %4d: pre_alpha=%.4f page_alpha=%.4f post_alpha=%.4f\n",
                   y, pre_max, pa_max, post_max);
        }
    }

    // ---- head crop + pass 2 ----
    std::map<std::string, Image> v3_layers;
    for (size_t i = 0; i < BODY_TAGS_V3.size(); i++) { v3_layers[BODY_TAGS_V3[i]] = body_layers[i]; }

    {
        const Image & head = body_layers[2];
        int hx0, hy0, hw, hh;
        if (!bbox_alpha(head, 15.0f / 255.0f, &hx0, &hy0, &hw, &hh)) {
            fprintf(stderr, "no head region found — skipping head pass\n");
        } else {
            // map to original-image coords (upstream apply_layerdiff v3)
            int hx = (int) (hx0 * scale) - pad_x;
            int hy = (int) (hy0 * scale) - pad_y;
            hw = (int) (hw * scale);
            hh = (int) (hh * scale);
            int iw = input.w, ih = input.h;
            int x1 = hx, y1 = hy, x2 = hx + hw, y2 = hy + hh;
            if (hw < iw / 2) {
                int px = std::min(std::min(iw - hx - hw, hx), hw / 5);
                x1 = std::min(std::max(hx - px, 0), iw);
                x2 = std::min(std::max(hx + hw + px, 0), iw);
            }
            if (hh < ih / 2) {
                int py = std::min(std::min(ih - hy - hh, hy), hh / 5);
                y1 = std::min(std::max(hy - py, 0), ih);
                y2 = std::min(std::max(hy + hh + py, 0), ih);
            }
            int hx1 = (int) (x1 / scale + pad_x / scale);
            int hy1 = (int) (y1 / scale + pad_y / scale);

            Image head_crop;
            head_crop.w = x2 - x1; head_crop.h = y2 - y1; head_crop.c = 4;
            head_crop.data.resize((size_t) head_crop.w * head_crop.h * 4);
            for (int yy = y1; yy < y2; yy++) {
                std::copy(input.px(x1, yy), input.px(x1, yy) + (size_t) head_crop.w * 4,
                          head_crop.data.begin() + (size_t) (yy - y1) * head_crop.w * 4);
            }
            int hp_w, hp_h, hp_x, hp_y;
            Image head_page = center_square_pad_resize(head_crop, RES, 0.0f, &hp_w, &hp_h, &hp_x, &hp_y);

            Image head_rgb;
            head_rgb.w = head_rgb.h = RES; head_rgb.c = 3;
            head_rgb.data.resize((size_t) RES * RES * 3);
            std::vector<float> head_alpha((size_t) RES * RES);
            for (size_t i = 0; i < (size_t) RES * RES; i++) {
                for (int c = 0; c < 3; c++) { head_rgb.data[i * 3 + c] = head_page.data[i * 4 + c]; }
                head_alpha[i] = head_page.data[i * 4 + 3];
            }

            std::vector<float> ehs2, pooled2;
            if (!encode_tags(cfg, HEAD_TAGS_V3, ehs2, pooled2)) { return false; }
            std::vector<Image> head_layers;
            if (!layerdiff_pass(cfg, head_rgb, ehs2, pooled2, 1, head_layers)) { return false; }

            // reproject each head layer to the fullpage canvas
            const int ch = head_crop.h, cw = head_crop.w;
            int py1 = (int) (hp_y / scale), py2 = (int) ((hp_y + ch) / scale);
            int px1 = (int) (hp_x / scale), px2 = (int) ((hp_x + cw) / scale);
            int sw = (int) (hp_w / scale), sh = (int) (hp_h / scale);
            for (size_t t = 0; t < HEAD_TAGS_V3.size(); t++) {
                Image & hl = head_layers[t];
                for (size_t i = 0; i < head_alpha.size(); i++) { hl.data[i * 4 + 3] *= head_alpha[i]; }
                Image big = smart_resize(hl, sw, sh);
                Image canvas;
                canvas.w = canvas.h = RES; canvas.c = 4;
                canvas.data.assign((size_t) RES * RES * 4, 0.0f);
                for (int yy = py1; yy < py2 && yy < big.h; yy++) {
                    int cy = hy1 + (yy - py1);
                    if (cy < 0 || cy >= RES) { continue; }
                    for (int xx = px1; xx < px2 && xx < big.w; xx++) {
                        int cx = hx1 + (xx - px1);
                        if (cx < 0 || cx >= RES) { continue; }
                        std::copy(big.px(xx, yy), big.px(xx, yy) + 4, canvas.px(cx, cy));
                    }
                }
                v3_layers[HEAD_TAGS_V3[t]] = std::move(canvas);
            }
        }
    }
    for (auto & kv : v3_layers) { alpha_floor(kv.second); }

    if (!cfg.debug_dir.empty()) {
        // temporary: dump one full, UNCROPPED layer to see the raw alpha
        // spatial distribution before crop_part's bbox tightens it
        auto it = v3_layers.find("topwear");
        if (it != v3_layers.end()) {
            std::ofstream f(cfg.debug_dir + "/raw_topwear_full.png", std::ios::binary);
            std::vector<uint8_t> png = encode_png(it->second);
            f.write(reinterpret_cast<const char *>(png.data()), (std::streamsize) png.size());
            printf("[debug] dumped raw uncropped topwear layer (%dx%d)\n",
                   it->second.w, it->second.h);
        }
    }

    // ---- marigold: assemble the V2 list (compose eyes/hair), page last ----
    const std::map<std::string, std::vector<std::string>> COMPOSE = {
        { "eyes", { "eyewhite", "irides", "eyelash", "eyebrow" } },
        { "hair", { "back hair", "front hair" } },
    };
    std::vector<Image> v2_list;
    std::vector<float> blended_alpha((size_t) RES * RES, 0.0f);
    Image empty;
    empty.w = empty.h = RES; empty.c = 4;
    empty.data.assign((size_t) RES * RES * 4, 0.0f);

    for (const std::string & tag : VALID_BODY_PARTS_V2) {
        Image img = empty;
        auto cit = COMPOSE.find(tag);
        if (cit != COMPOSE.end()) {
            for (const std::string & sub : cit->second) {
                auto it = v3_layers.find(sub);
                if (it != v3_layers.end()) { blend_over(img, it->second); }
            }
        } else {
            auto it = v3_layers.find(tag);
            if (it != v3_layers.end()) { img = it->second; }
        }
        for (size_t i = 0; i < blended_alpha.size(); i++) { blended_alpha[i] += img.data[i * 4 + 3]; }
        v2_list.push_back(std::move(img));
    }
    Image page_entry = fullpage;
    for (size_t i = 0; i < blended_alpha.size(); i++) {
        page_entry.data[i * 4 + 3] = std::min(1.0f, blended_alpha[i]);
    }
    v2_list.push_back(page_entry);

    std::vector<Image> depths;
    if (!marigold_depth(cfg, v2_list, depths)) { return false; }
    for (Image & d : depths) {
        if (d.w != RES) { d = smart_resize(d, RES, RES); }
    }

    // ---- assign depths to v3 parts (compose depth_local reconstruction) ----
    std::map<std::string, Part> parts;
    auto make_part = [&](const std::string & tag, const Image & img, const Image & depth) {
        Part p;
        p.tag = tag;
        p.img = img;
        p.depth = depth;
        p.xyxy[0] = p.xyxy[1] = 0; p.xyxy[2] = RES; p.xyxy[3] = RES;
        crop_part(p);
        if (p.img.w > 0) { parts[tag] = std::move(p); }
    };
    for (size_t v = 0; v < VALID_BODY_PARTS_V2.size(); v++) {
        const std::string & tag = VALID_BODY_PARTS_V2[v];
        const Image & depth = depths[v];
        auto cit = COMPOSE.find(tag);
        if (cit == COMPOSE.end()) {
            auto it = v3_layers.find(tag);
            if (it != v3_layers.end()) { make_part(tag, it->second, depth); }
            continue;
        }
        // per-sub-tag depth_local: reverse order, occluded regions get the
        // visible-region median (upstream apply_marigold)
        std::vector<uint8_t> occl((size_t) RES * RES, 0);
        for (size_t i = 0; i < occl.size(); i++) { occl[i] = blended_alpha[i] > 256.0f / 255.0f ? 1 : 0; }
        std::vector<std::string> subs = cit->second;
        for (auto sit = subs.rbegin(); sit != subs.rend(); ++sit) {
            auto it = v3_layers.find(*sit);
            if (it == v3_layers.end()) { continue; }
            const Image & sub = it->second;
            Image dl;
            dl.w = dl.h = RES; dl.c = 1;
            dl.data.assign((size_t) RES * RES, 1.0f);
            std::vector<float> visible;
            for (size_t i = 0; i < dl.data.size(); i++) {
                bool local = sub.data[i * 4 + 3] > 15.0f / 255.0f;
                if (local) {
                    dl.data[i] = std::min(1.0f, std::max(0.0f, depth.data[i]));
                    if (!occl[i]) { visible.push_back(dl.data[i]); }
                }
            }
            std::sort(visible.begin(), visible.end());
            float med = visible.empty() ? 1.0f : visible[visible.size() / 2];
            for (size_t i = 0; i < dl.data.size(); i++) {
                bool local = sub.data[i * 4 + 3] > 15.0f / 255.0f;
                if (local && occl[i]) { dl.data[i] = med; }
                if (local) { occl[i] = 1; }
            }
            make_part(*sit, sub, dl);
        }
    }

    // ---- heuristics + SVG/PNG assembly ----
    InpaintFn inpaint = make_lama_inpaint(cfg);
    further_extr_parts(parts, fullpage, inpaint, cfg.partseg_flags,
                       cfg.depth_split_tags, cfg.lr_split_tags);

    std::vector<const Part *> ordered;
    for (const auto & kv : parts) { ordered.push_back(&kv.second); }
    std::sort(ordered.begin(), ordered.end(),
              [](const Part * a, const Part * b) { return a->depth_median > b->depth_median; });

    result.png_layers.clear();
    for (const Part * p : ordered) {
        result.png_layers.emplace_back(safe_id(p->tag), encode_png(p->img));
    }

    // layered SVG: document order = z order (back to front); depth maps in
    // a hidden group; tag/depth_median/xyxy carried as data- attributes
    std::string svg;
    svg += "<svg xmlns=\"http://www.w3.org/2000/svg\" ";
    svg += "xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"" + std::to_string(RES)
         + "\" height=\"" + std::to_string(RES) + "\">\n";
    int zi = 0;
    for (const Part * p : ordered) {
        std::string png = b64_encode(encode_png(p->img));
        svg += "  <image id=\"" + safe_id(p->tag) + "\" x=\"" + std::to_string(p->xyxy[0])
             + "\" y=\"" + std::to_string(p->xyxy[1])
             + "\" width=\"" + std::to_string(p->img.w) + "\" height=\"" + std::to_string(p->img.h)
             + "\" data-tag=\"" + p->tag + "\" data-z=\"" + std::to_string(zi)
             + "\" data-depth-median=\"" + std::to_string(p->depth_median)
             + "\" xlink:href=\"data:image/png;base64," + png + "\"/>\n";
        zi++;
    }
    svg += "  <g id=\"depth\" display=\"none\">\n";
    zi = 0;
    for (const Part * p : ordered) {
        Image d1;
        d1.w = p->depth.w; d1.h = p->depth.h; d1.c = 1;
        d1.data = p->depth.data;
        svg += "    <image id=\"depth-" + safe_id(p->tag) + "\" x=\"" + std::to_string(p->xyxy[0])
             + "\" y=\"" + std::to_string(p->xyxy[1])
             + "\" width=\"" + std::to_string(d1.w) + "\" height=\"" + std::to_string(d1.h)
             + "\" data-tag=\"" + p->tag + "\" data-z=\"" + std::to_string(zi)
             + "\" xlink:href=\"data:image/png;base64," + b64_encode(encode_png(d1))
             + "\"/>\n";
        zi++;
    }
    svg += "  </g>\n</svg>\n";
    result.svg = std::move(svg);
    return true;
}
