// CLIP text encoding for see-through.cpp: the byte-level BPE tokenizer
// (vocab/merges embedded in the text-encoder ggufs) and the transformers
// CLIPTextModel graph, covering the three variants used by See-Through:
//   layerdiff-te1  CLIP-L        d=768  12L/12H quick_gelu  (penultimate out)
//   layerdiff-te2  OpenCLIP-G    d=1280 32L/20H gelu        (penultimate out,
//                  + pooled: final-LN EOS token -> text_projection)
//   marigold-te    OpenCLIP-H    d=1024 23L/16H gelu        (final-LN out)
#pragma once

#include "ops.h"

#include <cstdint>

struct ClipTokenizer {
    // token string -> id, and BPE merge ranks
    std::map<std::string, int32_t> vocab;
    std::map<std::pair<std::string, std::string>, int> merge_rank;
    int32_t bos_id = 49406, eos_id = 49407;

    // build from the gguf KV payloads (vocab.json text + merges.txt text)
    bool load(const std::string & vocab_json, const std::string & merges_txt);

    // lowercased whitespace-split byte-level BPE; returns bare token ids
    // (no BOS/EOS/padding)
    std::vector<int32_t> encode(const std::string & text) const;

    // [BOS, tokens..., EOS], truncated then padded with pad_id to n_ctx;
    // eos_pos receives the index of the EOS token
    std::vector<int32_t> encode_padded(const std::string & text, int n_ctx,
                                       int32_t pad_id, int * eos_pos) const;
};

struct ClipParams {
    int   n_layer;
    int   n_head;
    int   d_model;
    bool  quick_gelu;      // CLIP-L uses x*sigmoid(1.702x); OpenCLIP uses gelu
};

// parse n_layer/n_head/d_model/hidden_act from the component config_json KV
ClipParams clip_params_from_config(const std::string & config_json);

// Run the text transformer over `ids` (int32, [n_tok]). Outputs:
//   *penultimate: hidden states after n_layer-1 layers, no final LN (d, n_tok)
//   *final:       last_hidden_state = final_layer_norm(all layers)   (d, n_tok)
// Either out-param may be null if unused. Weights under "text_model.".
void clip_text_graph(Model & m, ggml_tensor * ids, const ClipParams & p,
                     ggml_tensor ** penultimate, ggml_tensor ** final_out);

// pooled output for CLIPTextModelWithProjection: final_out row at eos_pos
// through "text_projection.weight" (no bias) -> (proj_dim, 1)
ggml_tensor * clip_pooled_projection(Model & m, ggml_tensor * final_out, int eos_pos);
