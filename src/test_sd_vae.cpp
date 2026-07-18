// Milestone 2 of see-through.cpp: the stock SD/SDXL VAE decoder (diffusers
// AutoencoderKL) as a ggml graph, validated numerically against a PyTorch
// reference. Covers both layerdiff-vae (SDXL) and marigold-vae (SD).
//
//   test_sd_vae <vae.gguf> <reference_sd_vae.bin>

#include "test_common.h"
#include "vae.h"

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s vae.gguf reference.bin\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);

    Model m;
    if (!m.load(argv[1])) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
    printf("weights: %zu tensors\n", m.weights.size());

    std::vector<NpyArray> ref;   // z, y
    if (!read_ref(argv[2], ref, 2)) { fprintf(stderr, "failed to read %s\n", argv[2]); return 1; }
    const NpyArray & rz = ref[0], & ry = ref[1];
    const int64_t ZRES = rz.shape[3];

    init_graph_ctx(m, 4096);
    ggml_tensor * z = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, ZRES, ZRES, 4, 1);
    ggml_set_input(z);

    ggml_tensor * y = vae_decode(m, z);
    ggml_set_output(y);

    if (!compute_cpu(m, y, 4096, [&] {
            ggml_backend_tensor_set(z, rz.data.data(), 0, rz.data.size() * 4);
        })) return 1;

    return compare_ref(y, ry);
}
