// Head-group (group_index=1, the pass that produces "eyebrow"/"eyelash"/etc)
// UNetFrameConditionModel forward vs upstream, using REAL production inputs
// captured from an actual apply_layerdiff run (gen_reference_ahri_real_taps.py)
// rather than synthetic random tensors -- gen_reference_unet_forward_160.py's
// synthetic-weight test already showed a healthy single forward pass, but
// that doesn't rule out a real-content-specific divergence, and real upstream
// (both the live HF demo and a local run with the real checkpoint) reliably
// produces two separate connected components for eyebrow/eyelash/eyewhite/
// irides/ears/handwear where our ggml port produces one. This isolates
// exactly where (if anywhere) our ggml unet_frame_forward diverges from
// upstream for the identical real conditioning/latent at the real first
// denoising step (t=961, confirmed matching our own CLI's own first step).
//
//   test_unet_forward_head_real <layerdiff-unet.gguf> <reference_unet_head_real.bin>

#include "test_common.h"
#include "unet_frame.h"

#include <string>

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s layerdiff-unet.gguf reference.bin\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);

    Model m;
    if (!st_load(m, argv[1])) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
    printf("weights: %zu tensors\n", m.weights.size());

    // sample, ehs, text_embeds, time_ids, out, then taps: conv_in, down 0-2,
    // mid.resnet0, mid.attn, mid.resnet1
    std::vector<NpyArray> ref;
    if (!read_ref(argv[2], ref, 12)) { fprintf(stderr, "failed to read %s\n", argv[2]); return 1; }
    const int64_t F = ref[0].shape[1], ZRES = ref[0].shape[3];

    init_graph_ctx(m, 98304);
    ggml_context * ctx = m.ctx_g;

    ggml_tensor * sample = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ZRES, ZRES, 8, F);
    ggml_tensor * ehs    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 77, F);
    ggml_tensor * text   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1280, F);
    ggml_tensor * tids   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6, F);
    ggml_tensor * ts     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, F);
    for (ggml_tensor * t : { sample, ehs, text, tids, ts }) ggml_set_input(t);

    // group_index=1 (head pass): "group_embeds2.1"/"group_embeds.1", matching
    // pipeline.cpp's layerdiff_pass exactly (gi = std::to_string(group_index))
    ggml_tensor * ehs2 = ggml_add(ctx, ehs, group_embedding(m, ehs, "group_embeds2.1"));
    ggml_tensor * emb = time_embed_mlp(m, ggml_timestep_embedding(ctx, ts, 320, 10000),
                                       "time_embedding");
    ggml_tensor * aug_text = ggml_add(ctx, text, group_embedding(m, text, "group_embeds.1"));
    emb = ggml_add(ctx, emb, sdxl_add_embed(m, aug_text, tids));

    // --final-only: skip pinning every tap as a graph output (matches
    // production, which never does this) -- lets gallocr free/reuse
    // intermediate buffers instead of keeping all of them resident for the
    // whole graph, which inflates peak memory well past what a real
    // production run needs.
    const bool final_only = argc > 3 && std::string(argv[3]) == "--final-only";
    std::vector<ggml_tensor *> taps;
    ggml_tensor * out = unet_frame_forward(m, sample, emb, ehs2, final_only ? nullptr : &taps,
                                           /*fine_taps_down0=*/false,
                                           /*fine_taps_mid=*/!final_only);
    ggml_set_output(out);
    if (!final_only) { for (ggml_tensor * t : taps) ggml_set_output(t); }

    std::vector<ggml_tensor *> outs = { out };
    outs.insert(outs.end(), taps.begin(), taps.end());
    if (!compute_cpu_multi(m, outs, 98304, [&] {
            // reference sample is (1, F, 8, ZR, ZR) -- same memory order as
            // our (ZR, ZR, 8, F)
            ggml_backend_tensor_set(sample, ref[0].data.data(), 0, ref[0].data.size() * 4);
            ggml_backend_tensor_set(ehs,  ref[1].data.data(), 0, ref[1].data.size() * 4);
            ggml_backend_tensor_set(text, ref[2].data.data(), 0, ref[2].data.size() * 4);
            ggml_backend_tensor_set(tids, ref[3].data.data(), 0, ref[3].data.size() * 4);
            std::vector<float> t961(F, 961.0f);
            ggml_backend_tensor_set(ts, t961.data(), 0, F * 4);
        })) { return 1; }

    static const char * tap_names[] = { "conv_in", "down_block 0", "down_block 1",
                                        "down_block 2", "mid.resnet0", "mid.attn",
                                        "mid.resnet1 (final mid_block)" };
    int rc = 0;
    for (size_t i = 0; i < taps.size() && i + 5 < ref.size(); i++) {
        printf("-- %s --\n", tap_names[i]);
        if (compare_ref(taps[i], ref[5 + i]) != 0) rc = 1;
    }
    printf("-- final --\n");
    if (compare_ref(out, ref[4]) != 0) rc = 1;
    return rc;
}
