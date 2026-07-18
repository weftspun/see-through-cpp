// Milestone 3 of see-through.cpp: the full TransparentVAE decode chain as one
// ggml graph - SDXL VAE decode -> pixel*0.5+0.5 -> UNet1024 alpha head ->
// clip(0,1) - validated against upstream TransparentVAEDecoder.forward.
//
//   test_trans_vae_full <layerdiff-vae.gguf> <trans-vae.gguf> <reference_trans_vae_full.bin>

#include "test_common.h"
#include "vae.h"

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s layerdiff-vae.gguf trans-vae.gguf reference.bin\n", argv[0]);
        return 1;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);

    Model m;
    if (!st_load(m, argv[1]) || !st_load(m, argv[2])) { fprintf(stderr, "failed to load weights\n"); return 1; }
    printf("weights: %zu tensors (both ggufs)\n", m.weights.size());

    std::vector<NpyArray> ref;   // z, y
    if (!read_ref(argv[3], ref, 2)) { fprintf(stderr, "failed to read %s\n", argv[3]); return 1; }
    const NpyArray & rz = ref[0], & ry = ref[1];
    const int64_t ZRES = rz.shape[3];

    init_graph_ctx(m, 32768);
    ggml_tensor * z = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, ZRES, ZRES, 4, 1);
    ggml_set_input(z);

    ggml_tensor * y = trans_vae_decode(m, z);
    ggml_set_output(y);

    if (!compute_cpu(m, y, 32768, [&] {
            ggml_backend_tensor_set(z, rz.data.data(), 0, rz.data.size() * 4);
        })) return 1;

    return compare_ref(y, ry);
}
