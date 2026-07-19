// M5 of see-through.cpp: full LayerDiff UNetFrameConditionModel forward
// (13 frames, group 0, t=999) at 512px latent vs upstream.
//
//   test_unet_forward <layerdiff-unet.gguf> <reference_unet_forward.bin>

#include "test_common.h"
#include "unet_frame.h"

#include <string>

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s layerdiff-unet.gguf reference.bin\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);

    Model m;
    if (!st_load(m, argv[1])) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
    printf("weights: %zu tensors\n", m.weights.size());

    // sample, ehs, text_embeds, out, then taps: conv_in, down0.resnet0, down x3, mid
    std::vector<NpyArray> ref;
    if (!read_ref(argv[2], ref, 10)) { fprintf(stderr, "failed to read %s\n", argv[2]); return 1; }
    const int64_t F = ref[0].shape[1], ZRES = ref[0].shape[4];

    init_graph_ctx(m, 98304);
    ggml_context * ctx = m.ctx_g;

    ggml_tensor * sample = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ZRES, ZRES, 8, F);
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

    const bool no_taps = argc > 3 && std::string(argv[3]) == "--no-taps";
    std::vector<ggml_tensor *> taps;
    ggml_tensor * out = unet_frame_forward(m, sample, emb, ehs2, no_taps ? nullptr : &taps,
                                            /*fine_taps_down0=*/true);
    ggml_set_output(out);
    for (ggml_tensor * t : taps) ggml_set_output(t);

    std::vector<ggml_tensor *> outs = { out };
    outs.insert(outs.end(), taps.begin(), taps.end());
    if (!compute_cpu_multi(m, outs, 98304, [&] {
            // reference sample is (1, F, 8, 64, 64) — same memory order as
            // our (64, 64, 8, F)
            ggml_backend_tensor_set(sample, ref[0].data.data(), 0, ref[0].data.size() * 4);
            ggml_backend_tensor_set(ehs,  ref[1].data.data(), 0, ref[1].data.size() * 4);
            ggml_backend_tensor_set(text, ref[2].data.data(), 0, ref[2].data.size() * 4);
            std::vector<float> v(6 * F);
            for (int f = 0; f < F; f++) {
                const float ids[6] = { 1280, 1280, 0, 0, 1280, 1280 };
                for (int i = 0; i < 6; i++) { v[f * 6 + i] = ids[i]; }
            }
            ggml_backend_tensor_set(tids, v.data(), 0, v.size() * 4);
            std::vector<float> t999(F, 999.0f);
            ggml_backend_tensor_set(ts, t999.data(), 0, F * 4);
        })) { return 1; }

    static const char * tap_names[] = { "conv_in", "down_block 0 resnet0", "down_block 0",
                                        "down_block 1", "down_block 2", "mid_block" };
    for (size_t i = 0; i < taps.size() && i + 4 < ref.size(); i++) {
        printf("-- %s --\n", tap_names[i]);
        compare_ref(taps[i], ref[4 + i]);
    }
    printf("-- final --\n");
    return compare_ref(out, ref[3]);
}
