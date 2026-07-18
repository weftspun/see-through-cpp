// Pipeline stages for the see-through CLI: CLIP tag encoding, the
// apply_layerdiff v3 two-pass diffusion (body + head crop), the Marigold
// depth stage, and PSD assembly. CPU-first; each stage loads and frees its
// own weights.
#pragma once

#include "image_utils.h"
#include "postproc.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct PipelineConfig {
    std::string model_dir = ".";
    int  steps = 30;
    int  res = 1280;
    int  depth_res = 768;
    int  depth_steps = 4;
    uint64_t seed = 42;
    int  threads = 8;
    bool verbose = true;
    std::string device = "cpu";   // "cpu" | "vulkan" (first GPU in the registry)
};

// tags (fixed v3 vocabulary)
extern const std::vector<std::string> BODY_TAGS_V3;   // 13, group 0
extern const std::vector<std::string> HEAD_TAGS_V3;   // 11, group 1
extern const std::vector<std::string> VALID_BODY_PARTS_V2;

// stage 1: dual-CLIP embeddings for a tag list -> ehs [F,77,2048] and pooled
// [F,1280] (flattened row-major, frame-major)
bool encode_tags(const PipelineConfig & cfg, const std::vector<std::string> & tags,
                 std::vector<float> & ehs, std::vector<float> & pooled);

// stage 2: one apply_layerdiff pass: page RGB (square, res x res) -> per-tag
// RGBA layers at page resolution. group_index selects the group embeddings.
bool layerdiff_pass(const PipelineConfig & cfg, const Image & page_rgb,
                    const std::vector<float> & ehs, const std::vector<float> & pooled,
                    int group_index, std::vector<Image> & layers_out);

// stage 3: Marigold depth for a layer list (last entry = fullpage);
// depths_out matches layers in [0,1]
bool marigold_depth(const PipelineConfig & cfg, const std::vector<Image> & layers_argb,
                    std::vector<Image> & depths_out);

// LaMa inpaint callback backed by lama.gguf (loads weights on first use)
InpaintFn make_lama_inpaint(const PipelineConfig & cfg);
