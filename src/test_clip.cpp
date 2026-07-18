// M2 of see-through.cpp: the three CLIP text encoders (tokenizer + causal text
// transformer) validated against transformers, in the exact configurations
// See-Through uses them (see gen_reference_clip.py).
//
//   test_clip <te.gguf> <reference.bin> <te1|te2|marigold>

#include "test_common.h"
#include "clip.h"

static const char * PROMPT = "solo, 1girl, blue hair, cat ears, school uniform";

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s te.gguf reference.bin te1|te2|marigold\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string variant = argv[3];

    Model m;
    if (!m.load(argv[1])) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
    printf("weights: %zu tensors\n", m.weights.size());

    // pull config/vocab/merges KVs by suffix (keys are seethrough.<comp>.*)
    std::string cfg, vocab_js, merges;
    for (const auto & kv : m.config_json) {
        auto ends = [&](const char * s) {
            size_t n = strlen(s);
            return kv.first.size() > n && kv.first.compare(kv.first.size() - n, n, s) == 0;
        };
        if (ends(".config_json")) cfg = kv.second;
        if (ends(".vocab_json"))  vocab_js = kv.second;
        if (ends(".merges_txt"))  merges = kv.second;
    }
    ClipTokenizer tok;
    if (!tok.load(vocab_js, merges)) { fprintf(stderr, "tokenizer load failed\n"); return 1; }
    printf("tokenizer: %zu vocab, %zu merges\n", tok.vocab.size(), tok.merge_rank.size());
    ClipParams p = clip_params_from_config(cfg);
    printf("params: %dL/%dH d=%d %s\n", p.n_layer, p.n_head, p.d_model,
           p.quick_gelu ? "quick_gelu" : "gelu");

    const size_t n_ref = variant == "te2" ? 3 : 2;
    std::vector<NpyArray> ref;   // ids, hidden[, pooled]
    if (!read_ref(argv[2], ref, n_ref)) { fprintf(stderr, "failed to read %s\n", argv[2]); return 1; }

    // tokenize exactly as upstream: 77-token padded (te1 pads with EOS, te2
    // with "!" id 0); marigold = unpadded empty prompt [BOS, EOS]
    int eos_pos = 0;
    std::vector<int32_t> ids;
    if (variant == "marigold") {
        ids = { tok.bos_id, tok.eos_id };
        eos_pos = 1;
    } else {
        ids = tok.encode_padded(PROMPT, 77, variant == "te2" ? 0 : tok.eos_id, &eos_pos);
    }
    if (ids.size() != ref[0].data.size()) {
        fprintf(stderr, "TOKENIZER FAIL: %zu ids vs %zu reference\n", ids.size(), ref[0].data.size());
        return 1;
    }
    for (size_t i = 0; i < ids.size(); i++) {
        if (ids[i] != (int32_t) ref[0].data[i]) {
            fprintf(stderr, "TOKENIZER FAIL at %zu: %d vs %d\n", i, ids[i], (int32_t) ref[0].data[i]);
            return 1;
        }
    }
    printf("tokenizer: %zu ids match reference (eos at %d)\n", ids.size(), eos_pos);

    init_graph_ctx(m, 8192);
    ggml_tensor * ids_t = ggml_new_tensor_1d(m.ctx_g, GGML_TYPE_I32, ids.size());
    ggml_set_input(ids_t);

    ggml_tensor * penult = nullptr, * final_out = nullptr;
    clip_text_graph(m, ids_t, p, &penult, &final_out);
    ggml_tensor * hidden = variant == "marigold" ? final_out : penult;
    ggml_tensor * pooled = variant == "te2" ? clip_pooled_projection(m, final_out, eos_pos) : nullptr;
    ggml_set_output(hidden);
    ggml_tensor * out = pooled ? pooled : hidden;
    if (pooled) ggml_set_output(pooled);

    if (!compute_cpu(m, out, 8192, [&] {
            ggml_backend_tensor_set(ids_t, ids.data(), 0, ids.size() * 4);
        })) return 1;

    int rc = compare_ref(hidden, ref[1]);
    if (pooled) rc |= compare_ref(pooled, ref[2]);
    return rc;
}
