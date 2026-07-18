// Milestone 1 of see-through.cpp: the TransparentVAE decoder head (UNet1024)
// as a ggml graph, validated numerically against a PyTorch reference.
//
//   test_trans_vae <trans-vae.gguf> <reference_trans_vae.bin>

#include "test_common.h"
#include "vae.h"

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s trans-vae.gguf reference.bin\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);

    Model m;
    if (!m.load(argv[1])) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
    printf("weights: %zu tensors\n", m.weights.size());

    std::vector<NpyArray> ref;   // x, latent, y
    if (!read_ref(argv[2], ref, 3)) { fprintf(stderr, "failed to read %s\n", argv[2]); return 1; }
    const NpyArray & rx = ref[0], & rl = ref[1], & ry = ref[2];
    const int64_t RES = rx.shape[3], IN_CH = rx.shape[1];

    init_graph_ctx(m, 16384);
    ggml_tensor * x      = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, RES, RES, IN_CH, 1);
    ggml_tensor * latent = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, RES / 8, RES / 8, 4, 1);
    ggml_set_input(x);
    ggml_set_input(latent);

    ggml_tensor * y = unet1024(m, x, latent);
    ggml_set_output(y);

    if (!compute_cpu(m, y, 16384, [&] {
            ggml_backend_tensor_set(x,      rx.data.data(), 0, rx.data.size() * 4);
            ggml_backend_tensor_set(latent, rl.data.data(), 0, rl.data.size() * 4);
        })) return 1;

    return compare_ref(y, ry);
}
