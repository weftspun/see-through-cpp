// see-through CLI: single anime illustration -> layered PSD (+ optional
// per-layer PNGs), the full upstream inference_psd.py flow on ggml.
//
//   see-through -m <model-dir> -i in.png -o out.psd
//               [--seed N] [--steps N] [--res N] [--depth-res N] [--threads N]
//               [--no-split-depth] [--no-split-lr]
//               [--split-depth-tags tag1,tag2,...] [--split-lr-tags tag1,tag2,...]

#include "pipeline.h"
#include "psd_write.h"

#include <cstdio>
#include <filesystem>
#include <fstream>

static std::string psd_json_escape(const std::string & s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; }
        out += c;
    }
    return out;
}

// matches upstream dump_parts_psd's "<psd_path>.json" sidecar: dict2json({
// 'parts': tag2pinfo, 'frame_size': frame_size}, psd_savep + '.json') after
// popping img/depth/mask off each part dict -- the only fields that survive
// are tag/xyxy/depth_median, which is all psd2partdicts reads back.
static std::string build_psd_json(const SeeThroughResult & result) {
    std::string j = "{\"parts\": {";
    for (size_t i = 0; i < result.png_layers.size(); i++) {
        const std::string & tag = result.png_layers[i].first;
        const std::array<int, 4> & bb = result.layer_xyxy[i];
        if (i) { j += ", "; }
        j += "\"" + psd_json_escape(tag) + "\": {\"tag\": \"" + psd_json_escape(tag) + "\", \"xyxy\": ["
           + std::to_string(bb[0]) + ", " + std::to_string(bb[1]) + ", "
           + std::to_string(bb[2]) + ", " + std::to_string(bb[3]) + "], \"depth_median\": "
           + std::to_string(result.layer_depth_median[i]) + "}";
    }
    j += "}, \"frame_size\": [" + std::to_string(result.canvas_h) + ", " + std::to_string(result.canvas_w) + "]}";
    return j;
}

static std::vector<std::string> split_csv(const std::string & s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        size_t comma = s.find(',', start);
        std::string tok = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!tok.empty()) out.push_back(tok);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

int main(int argc, char ** argv) {
    PipelineConfig cfg;
    std::string in_path, out_path = "out.psd", png_dir;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "-m") { cfg.model_dir = next(); }
        else if (a == "-i") { in_path = next(); }
        else if (a == "-o") { out_path = next(); }
        else if (a == "--seed") { cfg.seed = std::stoull(next()); }
        else if (a == "--steps") { cfg.steps = std::stoi(next()); }
        else if (a == "--res") { cfg.res = std::stoi(next()); }
        else if (a == "--depth-res") { cfg.depth_res = std::stoi(next()); }
        else if (a == "--threads") { cfg.threads = std::stoi(next()); }
        else if (a == "--device") { cfg.device = next(); }
        else if (a == "--debug-dir") { cfg.debug_dir = next(); }
        else if (a == "--png-dir") { png_dir = next(); }
        else if (a == "--no-split-depth") { cfg.partseg_flags &= ~PARTSEG_DEPTH; }
        else if (a == "--no-split-lr")    { cfg.partseg_flags &= ~PARTSEG_LR; }
        else if (a == "--split-depth-tags") { cfg.depth_split_tags = split_csv(next()); }
        else if (a == "--split-lr-tags")    { cfg.lr_split_tags = split_csv(next()); }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (in_path.empty()) {
        fprintf(stderr, "usage: see-through -m <model-dir> -i in.png -o out.psd "
                        "[--seed N] [--steps N] [--res N] [--depth-res N] [--threads N] "
                        "[--device vulkan] (GPU-only; --device cpu is not supported) "
                        "[--no-split-depth] [--no-split-lr] "
                        "[--split-depth-tags tag1,tag2,...] [--split-lr-tags tag1,tag2,...]\n");
        return 1;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);

    // --res / --depth-res must be a multiple of each VAE decoder's total
    // upsample stride (skip-connection concat requires matching down/up
    // spatial sizes at every level) -- trans-vae has 6 stages (2^6=64),
    // marigold-vae has 3 (2^3=8). Round up rather than let the decoder
    // hit a ggml_concat shape-mismatch assert deep in a multi-minute run.
    auto round_up = [](int v, int div) { return ((v + div - 1) / div) * div; };
    int res64 = round_up(cfg.res, 64);
    if (res64 != cfg.res) {
        fprintf(stderr, "note: --res %d is not a multiple of 64 (trans-vae's decoder needs "
                        "this for its 6-stage skip connections) -- rounding up to %d\n",
                cfg.res, res64);
        cfg.res = res64;
    }
    int depth8 = round_up(cfg.depth_res, 8);
    if (depth8 != cfg.depth_res) {
        fprintf(stderr, "note: --depth-res %d is not a multiple of 8 (marigold-vae's decoder "
                        "needs this for its 3-stage skip connections) -- rounding up to %d\n",
                cfg.depth_res, depth8);
        cfg.depth_res = depth8;
    }

    Image input;
    if (!load_image(in_path, input)) { fprintf(stderr, "failed to load %s\n", in_path.c_str()); return 1; }
    printf("input: %dx%d\n", input.w, input.h);

    // set before run_see_through(): spans are appended + flushed to this
    // file as each one closes (see PipelineConfig::spans_path), not
    // batched until the run ends -- the directory has to exist first.
    cfg.spans_path = "profiling/spans.jsonl";
    std::filesystem::create_directories("profiling");

    SeeThroughResult result;
    if (!run_see_through(cfg, input, result)) { return 1; }

    if (!png_dir.empty()) {
        std::filesystem::create_directories(png_dir);
        int pzi = 0;
        for (auto & [tag, png] : result.png_layers) {
            char name[32];
            std::snprintf(name, sizeof(name), "%03d_", pzi);
            std::string path = png_dir + "/" + name + tag + ".png";
            std::ofstream f(path, std::ios::binary);
            f.write(reinterpret_cast<const char *>(png.data()), (std::streamsize) png.size());
            pzi++;
        }
        printf("wrote %d PNG layers to %s\n", pzi, png_dir.c_str());
    }

    if (out_path.size() < 4 || out_path.compare(out_path.size() - 4, 4, ".psd") != 0) {
        fprintf(stderr, "-o must end in .psd (got %s)\n", out_path.c_str());
        return 1;
    }
    if (!write_psd(out_path, result.canvas_w, result.canvas_h, result.png_layers, result.layer_xyxy)) {
        fprintf(stderr, "failed to write %s\n", out_path.c_str());
        return 1;
    }
    // matches upstream dump_parts_psd's "<stem>_depth.psd" companion
    std::string depth_path = out_path.substr(0, out_path.size() - 4) + "_depth.psd";
    if (!write_psd_gray(depth_path, result.canvas_w, result.canvas_h, result.depth_layers, result.layer_xyxy)) {
        fprintf(stderr, "failed to write %s\n", depth_path.c_str());
        return 1;
    }
    printf("wrote %s\n", depth_path.c_str());
    // matches upstream's "<psd_path>.json" metadata sidecar
    std::ofstream json_f(out_path + ".json");
    json_f << build_psd_json(result);
    printf("wrote %s.json\n", out_path.c_str());
    printf("wrote %s (%zu layers)\n", out_path.c_str(), result.png_layers.size());
    // result.spans were already appended + flushed to cfg.spans_path
    // incrementally as each closed (see run_see_through) -- no batch write
    // needed here.
    printf("%zu spans written to %s\n", result.spans.size(), cfg.spans_path.c_str());
    return 0;
}
