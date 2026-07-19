// 1280px collapse: does tiling fix it? (docs/ggml-upstream-issues.md #4)
// Validates vae_encode_tiled (diffusers-matching tile/blend/crop algorithm)
// against a real upstream reference generated WITH vae.enable_tiling(),
// same seed/input as reference_vae_encode_160_taps.bin's untiled version.
//
//   test_vae_encode_tiled <vae.gguf> <reference_vae_encode_160_tiled.bin>

#include "test_common.h"
#include "vae.h"

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s vae.gguf reference.bin\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);

    Model m;
    if (!st_load(m, argv[1])) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
    // production (pipe_load, pipeline.cpp) does not enable direct_conv/
    // conv_row_chunk for the VAE load -- st_load applies the env-var knobs
    // unconditionally, so reset here to match production.
    m.direct_conv = false;
    m.conv_row_chunk = false;
    printf("weights: %zu tensors\n", m.weights.size());

    std::vector<NpyArray> ref;   // x, z
    if (!read_ref(argv[2], ref, 2)) { fprintf(stderr, "failed to read %s\n", argv[2]); return 1; }
    const NpyArray & rx = ref[0], & rz = ref[1];
    const int64_t RES = rx.shape[3];

    // 16 tiles (4x4 at res=1280) each running the full encoder -- generous
    // node budget.
    init_graph_ctx(m, 65536);
    ggml_tensor * x = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, RES, RES, 3, 1);
    ggml_set_input(x);

    ggml_tensor * z = vae_encode_tiled(m, x);
    ggml_set_output(z);

    if (!compute_cpu_multi(m, { z }, 65536, [&] {
            ggml_backend_tensor_set(x, rx.data.data(), 0, rx.data.size() * 4);
        })) { return 1; }

    return compare_ref(z, rz);
}
