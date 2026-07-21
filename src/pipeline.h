// Pipeline stages for the see-through CLI: CLIP tag encoding, the
// apply_layerdiff v3 two-pass diffusion (body + head crop), the Marigold
// depth stage, and PSD assembly. GPU-only (whichever backend this build was
// compiled with, e.g. CUDA or Vulkan); each stage loads and frees its own
// weights.
#pragma once

#include "image_utils.h"
#include "otel_jsonl.h"
#include "postproc.h"

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct PipelineConfig {
    std::string model_dir = "models";
    int  steps = 30;
    int  res = 768;
    int  depth_res = 768;
    int  depth_steps = 4;
    uint64_t seed = 42;
    int  threads = 8;
    bool verbose = true;
    std::string device = "auto";  // any value other than "cpu" selects the first GPU ggml finds. GPU-only: "cpu" is rejected, not a fallback.
    std::string debug_dir;        // when set: dump per-stage stats + frames
    // when set: each span (see otel_jsonl.h) is appended + fflush()'d to
    // this JSONL file the instant it closes, not batched until the run
    // ends -- a crash/hang mid-run still leaves completed spans on disk.
    // Empty (default) disables file writes entirely, e.g. for library/C-ABI
    // callers that only want the in-memory SeeThroughResult::spans vector.
    std::string spans_path;

    // further_extr_parts heuristic control (see postproc.h PartsegFlags)
    unsigned partseg_flags = PARTSEG_DEPTH | PARTSEG_LR;
    // "topwear" defaults in too: garments with a draped/trailing portion
    // (e.g. a sash extending behind the legs) decode as one tag with two
    // genuinely different depths: front-body vs. the hallucinated occluded
    // drape. Left unsplit, that averages into one wrong mid-value; splitting
    // gives each its own honest depth_median (2026-07-19 field report).
    std::vector<std::string> depth_split_tags = { "hair", "topwear" };
    std::vector<std::string> lr_split_tags = {
        "handwear", "eyewhite", "irides", "eyelash", "eyebrow", "ears"
    };
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
                    int group_index, std::vector<Image> & layers_out,
                    const std::vector<std::string> & tags = {});

// stage 3: Marigold depth for a layer list (last entry = fullpage);
// depths_out matches layers in [0,1]
bool marigold_depth(const PipelineConfig & cfg, const std::vector<Image> & layers_argb,
                    std::vector<Image> & depths_out);

// LaMa inpaint callback backed by lama.gguf (loads weights on first use)
InpaintFn make_lama_inpaint(const PipelineConfig & cfg);

// full pipeline: input image -> depth-ordered per-tag layers (+ one PNG per
// layer, z-ordered back-to-front, tag as the id). Shared by the CLI
// (see_through.cpp) and the C ABI (see_through_capi.h) so both stay in sync
// with exactly one orchestration path.
struct SeeThroughResult {
    std::vector<std::pair<std::string, std::vector<uint8_t>>> png_layers;
    // 1-channel (grayscale) depth PNG per layer, same order/tag/crop as
    // png_layers -- upstream's dump_parts_psd writes this as a companion
    // "<name>_depth.psd" alongside the color PSD.
    std::vector<std::pair<std::string, std::vector<uint8_t>>> depth_layers;
    // per-entry {x0,y0,x1,y1} in canvas_w x canvas_h coords, same order/index
    // as png_layers/depth_layers -- carries each layer's placement for
    // consumers (e.g. PSD export) that want layer bytes + position.
    std::vector<std::array<int, 4>> layer_xyxy;
    // same order/index as png_layers -- upstream's dump_parts_psd carries
    // this (plus tag/xyxy) in the "<name>.psd.json" metadata sidecar.
    std::vector<double> layer_depth_median;
    int canvas_w = 0, canvas_h = 0;
    // OTel-style spans (root "run" span + one child per pipeline stage), for
    // the profiling data lake (see otel_jsonl.h) -- timed here rather than
    // left to external profilers (e.g. samply) so every run can log stage
    // timings without needing ETW/admin privileges.
    std::vector<Span> spans;
};
bool run_see_through(const PipelineConfig & cfg, const Image & input, SeeThroughResult & result);
