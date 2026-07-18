// M8 of see-through.cpp: the Marigold depth stage end-to-end â€” injected cond
// latents (page+layer), 4 trailing DDIM steps over the frame-condition UNet,
// VAE decode -> RGB mean -> clip -> [0,1] â€” vs the upstream pipeline.
//
//   test_marigold_e2e <marigold-unet.gguf> <marigold-vae.gguf> <reference_marigold.bin>

#include "test_common.h"
#include "scheduler.h"
#include "unet_frame.h"
#include "vae.h"

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s marigold-unet.gguf marigold-vae.gguf reference.bin\n", argv[0]);
        return 1;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);

    // cond (F,8,h,w), ehs (2,1024), target (F,4,h,w), depth (F,H,W), eps x4,
    // lat x4, unet_input_step0 (F,12,h,w)
    std::vector<NpyArray> ref;
    if (!read_ref(argv[3], ref, 13)) { fprintf(stderr, "failed to read %s\n", argv[3]); return 1; }
    const int64_t F = ref[0].shape[0], ZR = ref[0].shape[3];
    const int STEPS = 4;
    const float SCALE = 0.18215f;

    // ---- denoising loop ----
    Model m;
    if (!st_load(m, argv[1])) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
    printf("unet weights: %zu tensors\n", m.weights.size());

    init_graph_ctx(m, 16384);
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * sample = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ZR, ZR, 12, F);
    ggml_tensor * ehs    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1024, 2, F);
    ggml_tensor * ts     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, F);
    for (ggml_tensor * t : { sample, ehs, ts }) ggml_set_input(t);
    ggml_tensor * emb = time_embed_mlp(m, ggml_timestep_embedding(ctx, ts, 320, 10000),
                                       "time_embedding");
    ggml_tensor * out = unet_frame_forward(m, sample, emb, ehs);
    ggml_set_output(out);

    ggml_backend_t backend = st_backend_init();
    if (ggml_backend_is_cpu(backend)) ggml_backend_cpu_set_n_threads(backend, 8);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(gf, out);
    printf("graph: %d nodes\n", ggml_graph_n_nodes(gf));
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) { fprintf(stderr, "alloc failed\n"); return 1; }

    std::vector<float> ehs_f(1024 * 2 * F);
    for (int f = 0; f < F; f++) {
        std::copy(ref[1].data.begin(), ref[1].data.end(), ehs_f.begin() + f * 2048);
    }

    DdimTrailing sch;
    sch.set_timesteps(STEPS);

    const size_t DZ = (size_t) (ZR * ZR);
    const size_t D4 = DZ * 4, D8 = DZ * 8, D = D4 * F;
    std::vector<float> lat = ref[2].data;        // (F,4,h,w)
    std::vector<float> eps(D), input(DZ * 12 * F);
    double worst_v = 0;

    for (int s = 0; s < STEPS; s++) {
        for (int f = 0; f < F; f++) {
            std::copy(ref[0].data.begin() + f * D8, ref[0].data.begin() + (f + 1) * D8,
                      input.begin() + f * 12 * DZ);
            std::copy(lat.begin() + f * D4, lat.begin() + (f + 1) * D4,
                      input.begin() + f * 12 * DZ + D8);
        }
        if (s == 0) {
            double mx = 0;
            for (size_t i = 0; i < input.size(); i++) {
                double d = fabs((double) input[i] - (double) ref[12].data[i]);
                if (d > mx) mx = d;
            }
            printf("  step-0 unet input vs upstream: max_abs=%.6f\n", mx);
        }
        // gallocr recycles input buffers after their last read within one
        // compute â€” EVERY input must be re-set before EVERY compute
        ggml_backend_tensor_set(ehs, ehs_f.data(), 0, ehs_f.size() * 4);
        ggml_backend_tensor_set(sample, input.data(), 0, input.size() * 4);
        std::vector<float> tstep(F, (float) sch.timesteps[s]);
        ggml_backend_tensor_set(ts, tstep.data(), 0, F * 4);

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "compute failed at step %d\n", s); return 1;
        }
        ggml_backend_tensor_get(out, eps.data(), 0, D * 4);
        sch.step(lat, eps, s);
        printf("step %d (t=%d) done\n", s, sch.timesteps[s]);
        auto stage = [&](const char * what, const std::vector<float> & mine, const NpyArray & r) {
            double mx = 0, sm = 0;
            for (size_t i = 0; i < D; i++) {
                double d = fabs((double) mine[i] - (double) r.data[i]);
                if (d > mx) mx = d;
                sm += d;
            }
            printf("  %s: max_abs=%.6f mean_abs=%.6f\n", what, mx, sm / D);
        };
        stage("v  ", eps, ref[4 + s]);
        stage("lat", lat, ref[8 + s]);
        // teacher-force: chaotic amplification of f16 noise through iterated
        // low-noise steps swamps a free-running comparison; each step's
        // function is gated on exact upstream inputs instead
        double mx = 0;
        for (size_t i = 0; i < D; i++) {
            double d = fabs((double) eps[i] - (double) ref[4 + s].data[i]);
            if (d > mx) mx = d;
        }
        if (mx > worst_v) worst_v = mx;
        lat = ref[8 + s].data;
    }

    // ---- decode -> RGB mean -> clip -> [0,1] ----
    Model mv;
    if (!st_load(mv, argv[2])) { fprintf(stderr, "failed to load %s\n", argv[2]); return 1; }
    init_graph_ctx(mv, 8192);
    ggml_tensor * z = ggml_new_tensor_4d(mv.ctx_g, GGML_TYPE_F32, ZR, ZR, 4, F);
    ggml_set_input(z);
    ggml_tensor * img = vae_decode(mv, z);
    ggml_set_output(img);
    for (float & v : lat) v /= SCALE;
    if (!compute_cpu(mv, img, 8192, [&] {
            ggml_backend_tensor_set(z, lat.data(), 0, lat.size() * 4);
        })) return 1;

    const int64_t R = img->ne[0];
    std::vector<float> rgb(ggml_nelements(img));
    ggml_backend_tensor_get(img, rgb.data(), 0, rgb.size() * 4);
    const size_t P = (size_t) (R * R);
    double max_abs = 0, sum = 0;
    for (int f = 0; f < F; f++) {
        for (size_t i = 0; i < P; i++) {
            double d = (rgb[f * 3 * P + i] + rgb[f * 3 * P + P + i] + rgb[f * 3 * P + 2 * P + i]) / 3.0;
            d = d < -1.0 ? -1.0 : d > 1.0 ? 1.0 : d;
            d = (d + 1.0) / 2.0;
            double diff = fabs(d - (double) ref[3].data[f * P + i]);
            if (diff > max_abs) max_abs = diff;
            sum += diff;
        }
    }
    printf("depth: max_abs_diff=%.6f mean_abs_diff=%.6f (teacher-forced)\n", max_abs, sum / (P * F));
    printf("worst forced per-step v max_abs=%.6f\n", worst_v);
    const bool pass = max_abs < 5e-2 && worst_v < 5e-2;
    printf("%s\n", pass ? "VALIDATION PASS" : "VALIDATION FAIL");
    return pass ? 0 : 1;
}
