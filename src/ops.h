// Shared ggml building blocks for see-through.cpp: the diffusers vocabulary
// (conv/norm/linear/resnet/attention) used by every ported graph, plus the
// multi-gguf weight container. All spatial tensors are ggml WHCN layout.
#pragma once

#include "ggml.h"

#include <map>
#include <string>
#include <vector>

// Weights merged from one or more ggufs (diffusers state-dict names, which do
// not collide across the See-Through components), plus the per-subgraph
// normalization/attention knobs the block helpers read.
struct Model {
    std::vector<ggml_context *> ctx_w;   // one per gguf
    ggml_context * ctx_g = nullptr;      // graph (no_alloc), set by the caller

    std::map<std::string, ggml_tensor *> weights;
    std::map<std::string, std::string>   config_json;  // per arch KV, if present

    // set before building a subgraph
    int   gn_groups = 32;
    float gn_eps    = 1e-6f;
    int   head_dim  = 0;      // spatial attn: 0 = one head of dim C

    bool load(const char * path);        // merge tensors from a gguf (host RAM)
    // load into a backend buffer (e.g. Vulkan VRAM); pass the buffer type
    // from ggml_backend_get_default_buffer_type(backend)
    bool load_backend(const char * path, struct ggml_backend_buffer_type * buft);
    std::vector<struct ggml_backend_buffer *> bufs;   // owned backend buffers
    ggml_tensor * get(const std::string & name) const;   // exits on miss
    bool has(const std::string & name) const;
};

// (1,1,C,1) reshape for NCHW-broadcast of a per-channel bias
ggml_tensor * bias4d(ggml_context * ctx, ggml_tensor * b);

// conv + bias; pad 0 for 1x1 convs
ggml_tensor * conv2d(Model & m, ggml_tensor * x, const std::string & pre,
                     int stride = 1, int pad = 1);

// GroupNorm(m.gn_groups, m.gn_eps) + affine
ggml_tensor * group_norm_affine(Model & m, ggml_tensor * x, const std::string & pre);

// LayerNorm over ne[0] + affine, for token-major (C, T) activations
ggml_tensor * layer_norm_affine(Model & m, ggml_tensor * x, const std::string & pre,
                                float eps = 1e-5f);

// y = W x + b for token-major (C_in, T) activations (bias optional: skipped
// when "<pre>.bias" is absent, e.g. diffusers attention qkv projections)
ggml_tensor * linear(Model & m, ggml_tensor * x, const std::string & pre);

// diffusers ResnetBlock2D, output_scale_factor=1; temb (C_t, F) is projected
// through "<pre>.time_emb_proj" and added after conv1 (nullptr = skip)
ggml_tensor * resnet_block(Model & m, ggml_tensor * x, const std::string & pre,
                           ggml_tensor * temb = nullptr);

// diffusers Attention: GroupNorm-prenorm spatial self-attention with residual;
// heads of dim m.head_dim (0 = single head of dim C)
ggml_tensor * attn_block(Model & m, ggml_tensor * x, const std::string & pre);
