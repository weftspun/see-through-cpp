// 1280px collapse: isolate vae_decode_tiled from unet1024 (docs/ggml-
// upstream-issues.md #4). Validates vae_decode_tiled alone against a real
// upstream reference generated with vae.enable_tiling(); vae.decode(z).
//
//   test_vae_decode_tiled <vae.gguf> <reference_vae_decode_160_tiled.bin>

#include "test_common.h"
#include "vae.h"
#include "image_utils.h"

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s vae.gguf reference.bin\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);

    Model m;
    if (!st_load(m, argv[1])) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
    m.direct_conv = false;
    m.conv_row_chunk = false;
    printf("weights: %zu tensors\n", m.weights.size());

    std::vector<NpyArray> ref;   // z, pixel
    if (!read_ref(argv[2], ref, 2)) { fprintf(stderr, "failed to read %s\n", argv[2]); return 1; }
    const NpyArray & rz = ref[0], & rpixel = ref[1];
    const int64_t ZR = rz.shape[3];

    init_graph_ctx(m, 131072);
    ggml_tensor * z = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, ZR, ZR, 4, 1);
    ggml_set_input(z);

    ggml_tensor * pixel = vae_decode_tiled(m, z);
    ggml_set_output(pixel);

    if (!compute_cpu_multi(m, { pixel }, 131072, [&] {
            ggml_backend_tensor_set(z, rz.data.data(), 0, rz.data.size() * 4);
        })) { return 1; }

    if (getenv("SEETHROUGH_DUMP_PIXEL_PNG")) {
        std::vector<float> v(ggml_nelements(pixel));
        ggml_backend_tensor_get(pixel, v.data(), 0, v.size() * 4);
        const int64_t W = pixel->ne[0], H = pixel->ne[1];
        Image im; im.w = (int) W; im.h = (int) H; im.c = 3;
        im.data.resize((size_t) W * H * 3);
        for (int64_t y = 0; y < H; y++) {
            for (int64_t x = 0; x < W; x++) {
                for (int64_t c = 0; c < 3; c++) {
                    float val = v[(size_t) c * W * H + (size_t) y * W + x] * 0.5f + 0.5f;
                    im.data[((size_t) y * W + x) * 3 + c] = std::min(1.0f, std::max(0.0f, val));
                }
            }
        }
        save_image(getenv("SEETHROUGH_DUMP_PIXEL_PNG"), im);
    }

    return compare_ref(pixel, rpixel);
}
