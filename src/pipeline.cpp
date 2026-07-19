#include "pipeline.h"

#include "clip.h"
#include "lama.h"
#include "scheduler.h"
#include "unet_frame.h"
#include "vae.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
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
    if (cfg.device == "cpu") return nullptr;
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
}

static ggml_backend_t pipe_backend(const PipelineConfig & cfg) {
    ggml_backend_dev_t d = pipe_gpu(cfg);
    if (d) return ggml_backend_dev_init(d, nullptr);
    if (cfg.device != "cpu") {
        // GPU is the primary target; CPU is an explicit opt-in only
        // (--device cpu), never a silent fallback when Vulkan isn't found
        fprintf(stderr, "error: no GPU device found (Vulkan) and --device cpu was not "
                        "requested -- refusing to silently fall back to CPU. Pass "
                        "--device cpu if you really want a CPU run.\n");
        exit(1);
    }
    ggml_backend_t b = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(b, cfg.threads);
    return b;
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
            // A/B diagnostic for the 1280px collapse (docs/ggml-upstream-
            // issues.md #4): query-tiled naive attention instead of
            // flash_attn, VRAM-bounded so it can actually run at production
            // token counts (unlike a plain flash_attn=false toggle, which
            // OOMs). attn_tokens() checks this ahead of flash_attn.
            m.tiled_naive_attn = getenv("SEETHROUGH_TILED_ATTN") != nullptr;
        }
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
        bool ok = run_graph_dev(cfg, mv, 4096,
            [&]() {
                x = ggml_new_tensor_4d(mv.ctx_g, GGML_TYPE_F32, RES, RES, 3, 1);
                ggml_set_input(x);
                return std::vector<ggml_tensor *>{ ggml_scale(mv.ctx_g, vae_encode(mv, x), SCALE) };
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
            bool ok = run_graph_dev(cfg, mv, 32768,
                [&]() {
                    zt = ggml_new_tensor_4d(mv.ctx_g, GGML_TYPE_F32, ZR, ZR, 4, 1);
                    ggml_set_input(zt);
                    return std::vector<ggml_tensor *>{ trans_vae_decode(mv, zt) };
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
            bool ok = run_graph_dev(cfg, mv, 4096,
                [&]() {
                    x = ggml_new_tensor_4d(mv.ctx_g, GGML_TYPE_F32, RES, RES, 3, 1);
                    ggml_set_input(x);
                    return std::vector<ggml_tensor *>{ ggml_scale(mv.ctx_g, vae_encode(mv, x), SCALE) };
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
                    return std::vector<ggml_tensor *>{ vae_decode(mv, zt) };
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
