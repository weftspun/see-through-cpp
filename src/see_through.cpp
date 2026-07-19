// see-through CLI: single anime illustration -> layered SVG (+ optional
// per-layer PNGs), the full upstream inference_psd.py flow on ggml.
//
//   see-through -m <model-dir> -i in.png -o out.svg
//               [--seed N] [--steps N] [--res N] [--depth-res N] [--threads N]

#include "pipeline.h"

#include <cstdio>
#include <filesystem>
#include <fstream>

int main(int argc, char ** argv) {
    PipelineConfig cfg;
    std::string in_path, out_path = "out.svg", png_dir;
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
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (in_path.empty()) {
        fprintf(stderr, "usage: see-through -m <model-dir> -i in.png -o out.psd "
                        "[--seed N] [--steps N] [--res N] [--depth-res N] [--threads N] "
                        "[--device cpu|vulkan]\n");
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

    std::ofstream svg(out_path);
    svg << result.svg;
    printf("wrote %s (%zu layers)\n", out_path.c_str(), result.png_layers.size());
    return 0;
}
