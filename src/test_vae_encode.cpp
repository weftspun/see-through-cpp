// M3 of see-through.cpp: the AutoencoderKL encoder (image -> posterior latent
// mean), validated against diffusers for both VAEs.
//
//   test_vae_encode <vae.gguf> <reference_vae_encode_*.bin>

#include "test_common.h"
#include "vae.h"

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s vae.gguf reference.bin\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);

    Model m;
    if (!m.load(argv[1])) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
    printf("weights: %zu tensors\n", m.weights.size());

    std::vector<NpyArray> ref;   // x, z
    if (!read_ref(argv[2], ref, 2)) { fprintf(stderr, "failed to read %s\n", argv[2]); return 1; }
    const NpyArray & rx = ref[0], & rz = ref[1];
    const int64_t RES = rx.shape[3];

    init_graph_ctx(m, 4096);
    ggml_tensor * x = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, RES, RES, 3, 1);
    ggml_set_input(x);

    ggml_tensor * z = vae_encode(m, x);
    ggml_set_output(z);

    if (!compute_cpu(m, z, 4096, [&] {
            ggml_backend_tensor_set(x, rx.data.data(), 0, rx.data.size() * 4);
        })) return 1;

    return compare_ref(z, rz);
}
