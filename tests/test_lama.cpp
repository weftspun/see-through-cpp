// M9 of see-through.cpp: LaMa FFC inpainting (2-D FFT via matmul against a
// precomputed DFT basis, so the whole graph runs on GPU) vs the upstream
// generator. SEETHROUGH_DEVICE=cpu forces the CPU backend (both st_load and
// compute_cpu honor it) for a quick math-only sanity check; GPU (Vulkan) is
// the default and the actual production path.
//
//   test_lama <lama.gguf> <reference_lama.bin>

#include "test_common.h"
#include "lama.h"

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s lama.gguf reference.bin\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);

    Model m;
    if (!st_load(m, argv[1])) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
    printf("weights: %zu tensors\n", m.weights.size());
    lama_prepare_gpu_weights(m);

    std::vector<NpyArray> ref;   // img (1,3,H,W), mask (1,1,H,W), out (1,3,H,W)
    if (!read_ref(argv[2], ref, 3)) { fprintf(stderr, "failed to read %s\n", argv[2]); return 1; }
    const int64_t RES = ref[0].shape[3];

    init_graph_ctx(m, 8192);
    ggml_tensor * img  = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, RES, RES, 3, 1);
    ggml_tensor * mask = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, RES, RES, 1, 1);
    ggml_set_input(img);
    ggml_set_input(mask);

    LamaExtraInputs extra;
    ggml_tensor * out = lama_inpaint_graph(m, img, mask, &extra);
    ggml_set_output(out);

    if (!compute_cpu(m, out, 8192, [&] {
            ggml_backend_tensor_set(img,  ref[0].data.data(), 0, ref[0].data.size() * 4);
            ggml_backend_tensor_set(mask, ref[1].data.data(), 0, ref[1].data.size() * 4);
            lama_set_extra_inputs(extra, RES);
        })) return 1;

    return compare_ref(out, ref[2]);
}
