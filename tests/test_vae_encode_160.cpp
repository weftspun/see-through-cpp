// 1280px collapse root-cause check (docs/ggml-upstream-issues.md #4): the
// SDXL VAE encoder at the actual production shape (res=1280), with
// per-stage taps (conv_in, each down_block, mid_block) against
// gen_reference_vae_encode_160_taps.py's real upstream hooks -- this path
// was never exercised at this resolution before (the layerdiff e2e 1280px/
// 8-step reference showed c_concat, the encoded page latent, diverging
// wildly from upstream here).
//
//   test_vae_encode_160 <layerdiff-vae.gguf> <reference_vae_encode_160_taps.bin>

#include "test_common.h"
#include "vae.h"

#include <algorithm>

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s vae.gguf reference.bin\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);

    Model m;
    if (!st_load(m, argv[1])) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
    // production (pipe_load, pipeline.cpp) does not enable direct_conv/
    // conv_row_chunk for the VAE load -- ggml_conv_2d_direct is
    // documented-wrong for the encoder's stride-2/pad-0 downsample pattern
    // (docs/ggml-upstream-issues.md #3). st_load applies the env-var knobs
    // unconditionally, so reset here to match production.
    m.direct_conv = false;
    m.conv_row_chunk = false;
    if (getenv("SEETHROUGH_TILED_ATTN")) { m.tiled_naive_attn = true; }
    m.debug_capture = getenv("SEETHROUGH_DEBUG_CONV_STAGES") != nullptr;
    printf("weights: %zu tensors\n", m.weights.size());

    // x, z, then taps: conv_in, down0-3, mid.resnet0, mid.attn, mid.resnet1
    std::vector<NpyArray> ref;
    if (!read_ref(argv[2], ref, 10)) { fprintf(stderr, "failed to read %s\n", argv[2]); return 1; }
    const NpyArray & rx = ref[0], & rz = ref[1];
    const int64_t RES = rx.shape[3];

    init_graph_ctx(m, 8192);
    ggml_tensor * x = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, RES, RES, 3, 1);
    ggml_set_input(x);

    std::vector<ggml_tensor *> taps;
    ggml_tensor * z = vae_encode(m, x, &taps);
    ggml_set_output(z);
    for (ggml_tensor * t : taps) ggml_set_output(t);

    std::vector<ggml_tensor *> outs = { z };
    outs.insert(outs.end(), taps.begin(), taps.end());
    if (!compute_cpu_multi(m, outs, 8192, [&] {
            ggml_backend_tensor_set(x, rx.data.data(), 0, rx.data.size() * 4);
        })) { return 1; }

    static const char * tap_names[] = { "conv_in", "down_block 0", "down_block 1",
                                        "down_block 2", "down_block 3",
                                        "mid.resnet0", "mid.attn", "mid.resnet1" };
    for (size_t i = 0; i < taps.size() && i + 2 < ref.size(); i++) {
        printf("-- %s --\n", tap_names[i]);
        compare_ref(taps[i], ref[2 + i]);
    }
    printf("-- final (z) --\n");
    int rc = compare_ref(z, rz);

    const char * dump_dir = getenv("SEETHROUGH_DUMP_CONV_STAGES");
    for (auto & dt : m.debug_taps) {
        int64_t n = 1;
        for (int i = 0; i < 4; i++) n *= dt.ne[i];
        std::vector<float> v((size_t) n);
        ggml_backend_tensor_get(dt.t, v.data(), 0, v.size() * 4);
        double sum = 0, sum2 = 0, amax = 0;
        for (float f : v) { sum += f; sum2 += (double) f * f; amax = std::max(amax, (double) fabs(f)); }
        double mu = sum / n, sd = sqrt(std::max(0.0, sum2 / n - mu * mu));
        printf("[conv-stage] %-40s ne=(%lld,%lld,%lld,%lld) mean=%.4f std=%.4f max|v|=%.4f\n",
               dt.name.c_str(), (long long) dt.ne[0], (long long) dt.ne[1],
               (long long) dt.ne[2], (long long) dt.ne[3], mu, sd, amax);
        if (dump_dir) {
            std::string path = std::string(dump_dir) + "/" + dt.name + ".bin";
            FILE * f = fopen(path.c_str(), "wb");
            if (f) { fwrite(v.data(), 4, v.size(), f); fclose(f); }
        }
    }
    return rc;
}
