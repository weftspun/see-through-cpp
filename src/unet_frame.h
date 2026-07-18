// UNetFrameConditionModel building blocks (See-Through layerdiff3d /
// transformer3d): SDXL-style transformer blocks plus the parallel cross-frame
// temporal blocks and the LayerDiff group embeddings. Token activations are
// (C, T, B) with frames on the batch dim.
#pragma once

#include "ops.h"

// diffusers Attention over tokens: q from q_src (C, Tq, B), k/v from kv_src
// (Ck, Tk, B); no qkv bias, biased out proj, scale 1/sqrt(C/n_head)
ggml_tensor * attn_tokens(Model & m, ggml_tensor * q_src, ggml_tensor * kv_src,
                          const std::string & pre, int n_head);

// diffusers FeedForward("geglu"): net.0.proj -> GEGLU (erf) -> net.2
ggml_tensor * geglu_ff(Model & m, ggml_tensor * x, const std::string & pre);

// diffusers BasicTransformerBlock (layer_norm): LN -> self-attn -> LN ->
// cross-attn on ehs -> LN -> GEGLU ff, all residual
ggml_tensor * basic_transformer_block(Model & m, ggml_tensor * x, ggml_tensor * ehs,
                                      const std::string & pre, int n_head);

// CrossFrameTransformerBlock: x (C, S, F) tokens -> mix output (C, S, F).
// Internally permutes to (C, F, S) so attention runs across frames; returns
// the final FF output WITHOUT residual — the Transformer3D caller adds it.
ggml_tensor * cross_frame_block(Model & m, ggml_tensor * x, const std::string & pre,
                                int n_head);

// Transformer3DModel: x (W, H, C, F) + ehs (Ck, Tk, F) -> (W, H, C, F).
// GN32/1e-6 norm, linear proj_in/out, n_layers stock blocks with a temporal
// block after every `stride`-th one (stride 2 when n_layers >= 3), residual.
ggml_tensor * transformer3d(Model & m, ggml_tensor * x, ggml_tensor * ehs,
                            const std::string & pre, int n_head, int n_layers);

// diffusers TimestepEmbedding: linear_1 -> silu -> linear_2
ggml_tensor * time_embed_mlp(Model & m, ggml_tensor * t_emb, const std::string & pre);

// LayerDiff GroupEmbedding: linear(x + params), where params row f is added to
// frame f. x is (C, F) pooled or (C, T, F) sequence; params stored (C, n_cls).
ggml_tensor * group_embedding(Model & m, ggml_tensor * x, const std::string & pre);

// SDXL add_embedding input: concat(text_embeds (1280,F), 256-dim sinusoidal
// embedding of each of the 6 time_ids) -> add_embedding MLP -> (1280, F)
ggml_tensor * sdxl_add_embed(Model & m, ggml_tensor * text_embeds, ggml_tensor * time_ids);

// Full UNetFrameConditionModel forward (both layerdiff and marigold): sample
// (W, H, C_in, F), emb (1280, F) final time embedding, ehs (Ck, Tk, F) with
// any group embedding already applied. Self-configuring from weight presence
// (block/attention/layer counts, head count = C/64). GN32, norm_eps 1e-5.
ggml_tensor * unet_frame_forward(Model & m, ggml_tensor * sample, ggml_tensor * emb,
                                 ggml_tensor * ehs);
