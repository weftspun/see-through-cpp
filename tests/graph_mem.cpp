// VRAM diagnostics: build the layerdiff UNet graph at a given latent res /
// frame count with the low-VRAM knobs on, allocate with gallocr on host RAM,
// and report the buffer size plus the largest tensors.
//
//   graph_mem <layerdiff-unet.gguf> [latent_res=160] [frames=13]

#include "test_common.h"
#include "unet_frame.h"

#include <algorithm>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s layerdiff-unet.gguf [zres] [frames]\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int ZR = argc > 2 ? atoi(argv[2]) : 160;
    const int F  = argc > 3 ? atoi(argv[3]) : 13;

    Model m;
    if (!m.load(argv[1])) return 1;
    m.flash_attn = true;
    m.spatial_chunk = true;

    init_graph_ctx(m, 98304);
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * sample = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ZR, ZR, 8, F);
    ggml_tensor * ehs    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 77, F);
    ggml_tensor * text   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1280, F);
    ggml_tensor * tids   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6, F);
    ggml_tensor * ts     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, F);
    for (ggml_tensor * t : { sample, ehs, text, tids, ts }) ggml_set_input(t);

    ggml_tensor * ehs2 = ggml_add(ctx, ehs, group_embedding(m, ehs, "group_embeds2.0"));
    ggml_tensor * emb = time_embed_mlp(m, ggml_timestep_embedding(ctx, ts, 320, 10000),
                                       "time_embedding");
    ggml_tensor * aug = ggml_add(ctx, text, group_embedding(m, text, "group_embeds.0"));
    emb = ggml_add(ctx, emb, sdxl_add_embed(m, aug, tids));
    ggml_tensor * out = unet_frame_forward(m, sample, emb, ehs2);
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 98304, false);
    ggml_build_forward_expand(gf, out);
    printf("graph: %d nodes at zres=%d frames=%d\n", ggml_graph_n_nodes(gf), ZR, F);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(alloc, gf)) { fprintf(stderr, "reserve failed\n"); return 1; }
    printf("gallocr buffer: %.2f GB\n", ggml_gallocr_get_buffer_size(alloc, 0) / 1e9);

    // largest tensors by size with op names
    struct Row { size_t sz; const char * op; const char * name; };
    std::vector<Row> rows;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        rows.push_back({ ggml_nbytes(t), ggml_op_desc(t), t->name });
    }
    std::sort(rows.begin(), rows.end(), [](const Row & a, const Row & b) { return a.sz > b.sz; });
    printf("top tensors:\n");
    for (size_t i = 0; i < 25 && i < rows.size(); i++) {
        printf("  %7.3f GB  %-16s %s\n", rows[i].sz / 1e9, rows[i].op, rows[i].name);
    }
    return 0;
}
