// M4 of see-through.cpp: UNetFrameCondition building blocks vs upstream —
// resnet+temb, a full 2-layer Transformer3D (stock + cross-frame temporal
// blocks), and the conditioning path (time/add/group embeddings, group 0).
//
//   test_unet_blocks <layerdiff-unet.gguf> <reference_unet_blocks.bin>

#include "test_common.h"
#include "unet_frame.h"

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s layerdiff-unet.gguf reference.bin\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);

    Model m;
    if (!m.load(argv[1])) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
    printf("weights: %zu tensors\n", m.weights.size());

    // records: rx, rtemb, ry | tx, ehs, ty | text_embeds, emb, ehs2
    std::vector<NpyArray> ref;
    if (!read_ref(argv[2], ref, 9)) { fprintf(stderr, "failed to read %s\n", argv[2]); return 1; }
    const int64_t F = ref[0].shape[0];

    init_graph_ctx(m, 8192);
    ggml_context * ctx = m.ctx_g;
    m.gn_groups = 32; m.gn_eps = 1e-5f;

    ggml_tensor * x    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 16, 320, F);
    ggml_tensor * temb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1280, F);
    ggml_tensor * tx   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 16, 640, F);
    ggml_tensor * ehs  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 77, F);
    ggml_tensor * text = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1280, F);
    ggml_tensor * tids = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6, F);
    ggml_tensor * ts   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, F);
    for (ggml_tensor * t : { x, temb, tx, ehs, text, tids, ts }) ggml_set_input(t);

    ggml_tensor * resnet_out = resnet_block(m, x, "down_blocks.1.resnets.0", temb);
    ggml_tensor * t3d_out    = transformer3d(m, tx, ehs, "down_blocks.1.attentions.0", 10, 2);

    ggml_tensor * emb = time_embed_mlp(m, ggml_timestep_embedding(ctx, ts, 320, 10000),
                                       "time_embedding");
    ggml_tensor * aug_text = ggml_add(ctx, text, group_embedding(m, text, "group_embeds.0"));
    emb = ggml_add(ctx, emb, sdxl_add_embed(m, aug_text, tids));
    ggml_tensor * ehs2 = ggml_add(ctx, ehs, group_embedding(m, ehs, "group_embeds2.0"));

    for (ggml_tensor * t : { resnet_out, t3d_out, emb, ehs2 }) ggml_set_output(t);

    if (!compute_cpu_multi(m, { resnet_out, t3d_out, emb, ehs2 }, 8192, [&] {
            ggml_backend_tensor_set(x,    ref[0].data.data(), 0, ref[0].data.size() * 4);
            ggml_backend_tensor_set(temb, ref[1].data.data(), 0, ref[1].data.size() * 4);
            ggml_backend_tensor_set(tx,   ref[3].data.data(), 0, ref[3].data.size() * 4);
            ggml_backend_tensor_set(ehs,  ref[4].data.data(), 0, ref[4].data.size() * 4);
            ggml_backend_tensor_set(text, ref[6].data.data(), 0, ref[6].data.size() * 4);
            std::vector<float> v(6 * F);
            for (int f = 0; f < F; f++) {
                const float ids[6] = { 1280, 1280, 0, 0, 1280, 1280 };
                for (int i = 0; i < 6; i++) v[f * 6 + i] = ids[i];
            }
            ggml_backend_tensor_set(tids, v.data(), 0, v.size() * 4);
            std::vector<float> t999(F, 999.0f);
            ggml_backend_tensor_set(ts, t999.data(), 0, F * 4);
        })) return 1;

    int rc = 0;
    printf("-- resnet+temb --\n");        rc |= compare_ref(resnet_out, ref[2]);
    printf("-- transformer3d --\n");      rc |= compare_ref(t3d_out, ref[5]);
    printf("-- cond emb --\n");           rc |= compare_ref(emb, ref[7]);
    printf("-- group_embeds2 ehs --\n");  rc |= compare_ref(ehs2, ref[8]);
    return rc;
}
