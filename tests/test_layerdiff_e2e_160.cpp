// 1280px collapse root-cause check (docs/ggml-upstream-issues.md #4): same
// M7 denoising loop as test_layerdiff_e2e.cpp, but at res=1280 (ZR=160,
// the actual problem shape) / 8 steps (the pipeline collapses at
// --steps >= 8), against gen_reference_layerdiff_160.py's real upstream
// trajectory. Reports per-step eps/latent divergence so the first step
// where our output departs from upstream is visible directly, unlike any
// single-step check.
//
//   test_layerdiff_e2e_160 <layerdiff-unet.gguf> <layerdiff-vae.gguf> <reference_layerdiff_160.bin>

#include "test_common.h"
#include "scheduler.h"
#include "unet_frame.h"
#include "vae.h"

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s layerdiff-unet.gguf layerdiff-vae.gguf reference.bin\n", argv[0]);
        return 1;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);

    // records: vae_feed, c_concat, embeds, pooled, init_noise, sde_noise x8,
    // latents, eps x8, per-step latents x8
    const int STEPS = 8;
    const int NREC  = 6 + 3 * STEPS;   // 4 header + init + STEPS sde + final + STEPS eps + STEPS lat
    const int IDX_INIT  = 4;
    const int IDX_SDE0  = 5;
    const int IDX_FINAL = IDX_SDE0 + STEPS;
    const int IDX_EPS0  = IDX_FINAL + 1;
    const int IDX_LAT0  = IDX_EPS0 + STEPS;
    std::vector<NpyArray> ref;
    if (!read_ref(argv[3], ref, NREC)) { fprintf(stderr, "failed to read %s\n", argv[3]); return 1; }
    const NpyArray & r_feed = ref[0], & r_cc = ref[1], & r_ehs = ref[2], & r_pool = ref[3];
    const int64_t F = r_ehs.shape[0];
    const int64_t RES = r_feed.shape[2], ZR = RES / 8;
    const float SCALE = 0.13025f;

    // ---- stage 1: page latent via the SDXL VAE encoder ----
    std::vector<float> c_concat;
    {
        Model mv;
        if (!st_load(mv, argv[2])) { fprintf(stderr, "failed to load %s\n", argv[2]); return 1; }
        // st_load applies SEETHROUGH_CHUNK/ROWCHUNK unconditionally, but
        // production (pipe_load, pipeline.cpp) only enables direct_conv/
        // conv_row_chunk for the UNet load (unet=true) -- ggml_conv_2d_direct
        // is documented-wrong for the VAE encoder's stride-2/pad-0 downsample
        // pattern (docs/ggml-upstream-issues.md #3). Match production here.
        mv.direct_conv = false;
        mv.conv_row_chunk = false;
        // st_load doesn't wire up SEETHROUGH_TILED_ATTN (only pipe_load,
        // pipeline.cpp, does) -- set directly to test the VAE mid_block
        // attn_block Vulkan buffer-overflow fix (docs/ggml-upstream-
        // issues.md #4's unet1024 finding) at this encoder's own bottleneck
        // T=ZR*ZR.
        if (getenv("SEETHROUGH_TILED_ATTN")) { mv.tiled_naive_attn = true; }
        init_graph_ctx(mv, 4096);
        ggml_tensor * x = ggml_new_tensor_4d(mv.ctx_g, GGML_TYPE_F32, RES, RES, 3, 1);
        ggml_set_input(x);
        ggml_tensor * z = ggml_scale(mv.ctx_g, vae_encode(mv, x), SCALE);
        ggml_set_output(z);
        if (!compute_cpu(mv, z, 4096, [&] {
                ggml_backend_tensor_set(x, r_feed.data.data(), 0, r_feed.data.size() * 4);
            })) { return 1; }
        printf("-- c_concat --\n");
        if (compare_ref(z, r_cc)) { return 1; }
        c_concat.resize(ggml_nelements(z));
        ggml_backend_tensor_get(z, c_concat.data(), 0, c_concat.size() * 4);
        ggml_free(mv.ctx_g);
    }

    // ---- stage 2: denoising loop ----
    Model m;
    if (!st_load(m, argv[1])) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
    printf("weights: %zu tensors\n", m.weights.size());

    init_graph_ctx(m, 16384);
    ggml_context * ctx = m.ctx_g;

    ggml_tensor * sample = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ZR, ZR, 8, F);
    ggml_tensor * ehs    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 77, F);
    ggml_tensor * text   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1280, F);
    ggml_tensor * tids   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6, F);
    ggml_tensor * ts     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, F);
    for (ggml_tensor * t : { sample, ehs, text, tids, ts }) ggml_set_input(t);

    ggml_tensor * ehs2 = ggml_add(ctx, ehs, group_embedding(m, ehs, "group_embeds2.0"));
    ggml_tensor * emb = time_embed_mlp(m, ggml_timestep_embedding(ctx, ts, 320, 10000),
                                       "time_embedding");
    ggml_tensor * aug_text = ggml_add(ctx, text, group_embedding(m, text, "group_embeds.0"));
    emb = ggml_add(ctx, emb, sdxl_add_embed(m, aug_text, tids));
    ggml_tensor * out = unet_frame_forward(m, sample, emb, ehs2);
    ggml_set_output(out);

    // static graph reused every step: allocate once, swap inputs
    ggml_backend_t backend = st_backend_init();
    if (ggml_backend_is_cpu(backend)) { ggml_backend_cpu_set_n_threads(backend, 8); }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(gf, out);
    printf("graph: %d nodes\n", ggml_graph_n_nodes(gf));
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) { fprintf(stderr, "alloc failed\n"); return 1; }

    std::vector<float> v(6 * F);
    for (int f = 0; f < F; f++) {
        const float ids[6] = { (float) RES, (float) RES, 0, 0, (float) RES, (float) RES };
        for (int i = 0; i < 6; i++) { v[f * 6 + i] = ids[i]; }
    }

    DpmSolverSDE sch;
    sch.set_timesteps(STEPS);

    const size_t DZ = (size_t) (ZR * ZR * 4);         // per-frame latent
    const size_t D  = DZ * F;
    std::vector<float> lat(D), eps(D), input(ZR * ZR * 8 * F);
    // init noise (1,1,4,64,64) broadcast over frames
    for (int f = 0; f < F; f++) {
        std::copy(ref[IDX_INIT].data.begin(), ref[IDX_INIT].data.end(), lat.begin() + f * DZ);
    }

    for (int s = 0; s < STEPS; s++) {
        // channel-concat latent + c_concat per frame
        for (int f = 0; f < F; f++) {
            std::copy(lat.begin() + f * DZ, lat.begin() + (f + 1) * DZ,
                      input.begin() + f * 2 * DZ);
            std::copy(c_concat.begin(), c_concat.end(), input.begin() + f * 2 * DZ + DZ);
        }
        // gallocr recycles input buffers after their last read within one
        // compute - EVERY input must be re-set before EVERY compute
        ggml_backend_tensor_set(ehs, r_ehs.data.data(), 0, r_ehs.data.size() * 4);
        ggml_backend_tensor_set(text, r_pool.data.data(), 0, r_pool.data.size() * 4);
        ggml_backend_tensor_set(tids, v.data(), 0, v.size() * 4);
        ggml_backend_tensor_set(sample, input.data(), 0, input.size() * 4);
        std::vector<float> tstep(F, (float) sch.timesteps[s]);
        ggml_backend_tensor_set(ts, tstep.data(), 0, F * 4);

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "compute failed at step %d\n", s); return 1;
        }
        ggml_backend_tensor_get(out, eps.data(), 0, D * 4);
        sch.step(lat, eps, ref[IDX_SDE0 + s].data);
        printf("step %d (t=%d) done\n", s, sch.timesteps[s]);
        auto stage = [&](const char * what, const std::vector<float> & mine, const NpyArray & r) {
            double mx = 0, sm = 0;
            for (size_t i = 0; i < D; i++) {
                double d = fabs((double) mine[i] - (double) r.data[i]);
                if (d > mx) { mx = d; }
                sm += d;
            }
            printf("  %s: max_abs=%.6f mean_abs=%.6f\n", what, mx, sm / D);
        };
        stage("eps", eps, ref[IDX_EPS0 + s]);
        stage("lat", lat, ref[IDX_LAT0 + s]);
        if (argc > 4 && std::string(argv[4]) == "--teacher-force") {
            lat = ref[IDX_LAT0 + s].data;      // continue from upstream's own latents
        }
    }

    NpyArray out_arr;
    out_arr.shape = ref[IDX_FINAL].shape;
    out_arr.data = ref[IDX_FINAL].data;
    // compare against final latents
    double max_abs = 0, sum = 0;
    for (size_t i = 0; i < D; i++) {
        double d = fabs((double) lat[i] - (double) ref[IDX_FINAL].data[i]);
        if (d > max_abs) { max_abs = d; }
        sum += d;
    }
    printf("final latents: max_abs_diff=%.6f mean_abs_diff=%.6f\n", max_abs, sum / D);
    printf("%s\n", max_abs < 5e-2 ? "VALIDATION PASS" : "VALIDATION FAIL");
    return max_abs < 5e-2 ? 0 : 1;
}
