#include "see_through_capi.h"

#include "image_utils.h"
#include "pipeline.h"

#include <cstdlib>
#include <cstring>

namespace {

char * dup_str(const std::string & s) {
    char * p = static_cast<char *>(malloc(s.size() + 1));
    memcpy(p, s.data(), s.size());
    p[s.size()] = '\0';
    return p;
}

uint8_t * dup_bytes(const std::vector<uint8_t> & v) {
    if (v.empty()) { return nullptr; }
    uint8_t * p = static_cast<uint8_t *>(malloc(v.size()));
    memcpy(p, v.data(), v.size());
    return p;
}

} // namespace

int st_render(const char * model_dir, const uint8_t * image_data, size_t image_len,
              int steps, int res, int depth_res, uint64_t seed, const char * device,
              st_render_result * out) {
    if (!out) { return 1; }
    memset(out, 0, sizeof(*out));
    if (!model_dir || !image_data || image_len == 0) { return 1; }

    Image input;
    if (!load_image_from_memory(image_data, image_len, input, 4)) { return 2; }

    PipelineConfig cfg;
    cfg.model_dir = model_dir;
    cfg.steps = steps > 0 ? steps : cfg.steps;
    cfg.res = res > 0 ? res : cfg.res;
    cfg.depth_res = depth_res > 0 ? depth_res : cfg.depth_res;
    cfg.seed = seed;
    cfg.device = device ? device : cfg.device;
    cfg.verbose = false;

    // same rounding the CLI applies (trans-vae/marigold-vae skip-connection
    // stride requirements) -- silently corrects rather than asserting deep
    // inside a multi-minute run
    auto round_up = [](int v, int div) { return ((v + div - 1) / div) * div; };
    cfg.res = round_up(cfg.res, 64);
    cfg.depth_res = round_up(cfg.depth_res, 8);

    SeeThroughResult result;
    if (!run_see_through(cfg, input, result)) { return 3; }

    out->svg = dup_str(result.svg);
    out->svg_len = result.svg.size();
    out->num_layers = result.png_layers.size();
    out->layers = static_cast<st_layer *>(calloc(out->num_layers, sizeof(st_layer)));
    for (size_t i = 0; i < out->num_layers; i++) {
        out->layers[i].tag = dup_str(result.png_layers[i].first);
        out->layers[i].png = dup_bytes(result.png_layers[i].second);
        out->layers[i].png_len = result.png_layers[i].second.size();
    }
    return 0;
}

void st_free_result(st_render_result * r) {
    if (!r) { return; }
    free(r->svg);
    for (size_t i = 0; i < r->num_layers; i++) {
        free(r->layers[i].tag);
        free(r->layers[i].png);
    }
    free(r->layers);
    memset(r, 0, sizeof(*r));
}
