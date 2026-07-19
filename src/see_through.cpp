// see-through CLI: single anime illustration -> layered PSD (+ depth PSD +
// sidecar json), the full upstream inference_psd.py flow on ggml.
//
//   see-through -m <model-dir> -i in.png -o out.psd
//               [--seed N] [--steps N] [--res N] [--depth-res N] [--threads N]

#include "pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

// upstream img_alpha_blending (premultiplied=False): sequential "over"
static void blend_over(Image & dst, const Image & src) {
    for (size_t i = 0; i < (size_t) dst.w * dst.h; i++) {
        float a = src.data[i * 4 + 3];
        for (int c = 0; c < 3; c++) {
            dst.data[i * 4 + c] = src.data[i * 4 + c] * a + dst.data[i * 4 + c] * (1 - a);
        }
        dst.data[i * 4 + 3] = a + dst.data[i * 4 + 3] * (1 - a);
    }
}

static void alpha_floor(Image & img) {          // upstream: alpha < 15/255 -> 0
    for (size_t i = 3; i < img.data.size(); i += 4) {
        if (img.data[i] < 15.0f / 255.0f) img.data[i] = 0.0f;
    }
}

static bool bbox_alpha(const Image & img, float thr, int * x, int * y, int * w, int * h) {
    int x0 = img.w, y0 = img.h, x1 = -1, y1 = -1;
    for (int yy = 0; yy < img.h; yy++) {
        for (int xx = 0; xx < img.w; xx++) {
            if (img.px(xx, yy)[3] > thr) {
                x0 = std::min(x0, xx); y0 = std::min(y0, yy);
                x1 = std::max(x1, xx); y1 = std::max(y1, yy);
            }
        }
    }
    if (x1 < 0) return false;
    *x = x0; *y = y0; *w = x1 - x0 + 1; *h = y1 - y0 + 1;
    return true;
}

int main(int argc, char ** argv) {
    PipelineConfig cfg;
    std::string in_path, out_path = "out.svg", png_dir;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "-m") cfg.model_dir = next();
        else if (a == "-i") in_path = next();
        else if (a == "-o") out_path = next();
        else if (a == "--seed") cfg.seed = std::stoull(next());
        else if (a == "--steps") cfg.steps = std::stoi(next());
        else if (a == "--res") cfg.res = std::stoi(next());
        else if (a == "--depth-res") cfg.depth_res = std::stoi(next());
        else if (a == "--threads") cfg.threads = std::stoi(next());
        else if (a == "--device") cfg.device = next();
        else if (a == "--debug-dir") cfg.debug_dir = next();
        else if (a == "--png-dir") png_dir = next();
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

    const int RES = cfg.res;
    int pad_w, pad_h, pad_x, pad_y;
    Image fullpage = center_square_pad_resize(input, RES, 0.0f, &pad_w, &pad_h, &pad_x, &pad_y);
    const double scale = (double) pad_w / RES;

    // page RGB (alpha treated as 255) + page alpha for masking decoded layers
    Image page_rgb;
    page_rgb.w = page_rgb.h = RES; page_rgb.c = 3;
    page_rgb.data.resize((size_t) RES * RES * 3);
    std::vector<float> page_alpha((size_t) RES * RES);
    for (size_t i = 0; i < (size_t) RES * RES; i++) {
        for (int c = 0; c < 3; c++) page_rgb.data[i * 3 + c] = fullpage.data[i * 4 + c];
        page_alpha[i] = fullpage.data[i * 4 + 3];
    }

    if (!cfg.debug_dir.empty()) {
        Image dbg; dbg.w = dbg.h = RES; dbg.c = 1;
        dbg.data = page_alpha;
        std::ofstream f(cfg.debug_dir + "/page_alpha.png", std::ios::binary);
        std::vector<uint8_t> png = encode_png(dbg);
        f.write(reinterpret_cast<const char *>(png.data()), (std::streamsize) png.size());
        int rows_nonzero = 0;
        for (int y = 0; y < RES; y++) {
            bool any = false;
            for (int x = 0; x < RES; x++) if (page_alpha[(size_t) y * RES + x] > 0.5f) { any = true; break; }
            if (any) rows_nonzero++;
        }
        printf("[debug] page_alpha: %d/%d rows have any nonzero alpha (pad_w=%d pad_h=%d pad_x=%d pad_y=%d)\n",
               rows_nonzero, RES, pad_w, pad_h, pad_x, pad_y);
    }

    // ---- layerdiff pass 1: body ----
    std::vector<float> ehs, pooled;
    if (!encode_tags(cfg, BODY_TAGS_V3, ehs, pooled)) return 1;
    std::vector<Image> body_layers;
    if (!layerdiff_pass(cfg, page_rgb, ehs, pooled, 0, body_layers)) return 1;

    std::vector<float> pre_mul_alpha5;   // snapshot before the page_alpha multiply
    if (!cfg.debug_dir.empty() && body_layers.size() > 5) {
        Image & l = body_layers[5];
        pre_mul_alpha5.resize((size_t) l.w * l.h);
        for (size_t i = 0; i < pre_mul_alpha5.size(); i++) pre_mul_alpha5[i] = l.data[i * 4 + 3];
    }

    for (Image & l : body_layers) {
        for (size_t i = 0; i < page_alpha.size(); i++) l.data[i * 4 + 3] *= page_alpha[i];
    }

    if (!cfg.debug_dir.empty() && body_layers.size() > 5) {
        Image & l = body_layers[5];   // topwear, pre-alpha_floor
        printf("[debug] body_layers[5] (topwear), w=%d h=%d, page_alpha.size=%zu\n",
               l.w, l.h, page_alpha.size());
        for (int y = 0; y < l.h; y += (l.h / 20 > 0 ? l.h / 20 : 1)) {
            float pre_max = 0, pa_max = 0, post_max = 0;
            for (int x = 0; x < l.w; x++) {
                size_t i = (size_t) y * l.w + x;
                pre_max = std::max(pre_max, pre_mul_alpha5[i]);
                pa_max = std::max(pa_max, page_alpha[i]);
                post_max = std::max(post_max, l.px(x, y)[3]);
            }
            printf("[debug]   row %4d: pre_alpha=%.4f page_alpha=%.4f post_alpha=%.4f\n",
                   y, pre_max, pa_max, post_max);
        }
    }

    // ---- head crop + pass 2 ----
    std::map<std::string, Image> v3_layers;
    for (size_t i = 0; i < BODY_TAGS_V3.size(); i++) v3_layers[BODY_TAGS_V3[i]] = body_layers[i];

    {
        const Image & head = body_layers[2];
        int hx0, hy0, hw, hh;
        if (!bbox_alpha(head, 15.0f / 255.0f, &hx0, &hy0, &hw, &hh)) {
            fprintf(stderr, "no head region found — skipping head pass\n");
        } else {
            // map to original-image coords (upstream apply_layerdiff v3)
            int hx = (int) (hx0 * scale) - pad_x;
            int hy = (int) (hy0 * scale) - pad_y;
            hw = (int) (hw * scale);
            hh = (int) (hh * scale);
            int iw = input.w, ih = input.h;
            int x1 = hx, y1 = hy, x2 = hx + hw, y2 = hy + hh;
            if (hw < iw / 2) {
                int px = std::min(std::min(iw - hx - hw, hx), hw / 5);
                x1 = std::min(std::max(hx - px, 0), iw);
                x2 = std::min(std::max(hx + hw + px, 0), iw);
            }
            if (hh < ih / 2) {
                int py = std::min(std::min(ih - hy - hh, hy), hh / 5);
                y1 = std::min(std::max(hy - py, 0), ih);
                y2 = std::min(std::max(hy + hh + py, 0), ih);
            }
            int hx1 = (int) (x1 / scale + pad_x / scale);
            int hy1 = (int) (y1 / scale + pad_y / scale);

            Image head_crop;
            head_crop.w = x2 - x1; head_crop.h = y2 - y1; head_crop.c = 4;
            head_crop.data.resize((size_t) head_crop.w * head_crop.h * 4);
            for (int yy = y1; yy < y2; yy++) {
                std::copy(input.px(x1, yy), input.px(x1, yy) + (size_t) head_crop.w * 4,
                          head_crop.data.begin() + (size_t) (yy - y1) * head_crop.w * 4);
            }
            int hp_w, hp_h, hp_x, hp_y;
            Image head_page = center_square_pad_resize(head_crop, RES, 0.0f, &hp_w, &hp_h, &hp_x, &hp_y);

            Image head_rgb;
            head_rgb.w = head_rgb.h = RES; head_rgb.c = 3;
            head_rgb.data.resize((size_t) RES * RES * 3);
            std::vector<float> head_alpha((size_t) RES * RES);
            for (size_t i = 0; i < (size_t) RES * RES; i++) {
                for (int c = 0; c < 3; c++) head_rgb.data[i * 3 + c] = head_page.data[i * 4 + c];
                head_alpha[i] = head_page.data[i * 4 + 3];
            }

            std::vector<float> ehs2, pooled2;
            if (!encode_tags(cfg, HEAD_TAGS_V3, ehs2, pooled2)) return 1;
            std::vector<Image> head_layers;
            if (!layerdiff_pass(cfg, head_rgb, ehs2, pooled2, 1, head_layers)) return 1;

            // reproject each head layer to the fullpage canvas
            const int ch = head_crop.h, cw = head_crop.w;
            int py1 = (int) (hp_y / scale), py2 = (int) ((hp_y + ch) / scale);
            int px1 = (int) (hp_x / scale), px2 = (int) ((hp_x + cw) / scale);
            int sw = (int) (hp_w / scale), sh = (int) (hp_h / scale);
            for (size_t t = 0; t < HEAD_TAGS_V3.size(); t++) {
                Image & hl = head_layers[t];
                for (size_t i = 0; i < head_alpha.size(); i++) hl.data[i * 4 + 3] *= head_alpha[i];
                Image big = smart_resize(hl, sw, sh);
                Image canvas;
                canvas.w = canvas.h = RES; canvas.c = 4;
                canvas.data.assign((size_t) RES * RES * 4, 0.0f);
                for (int yy = py1; yy < py2 && yy < big.h; yy++) {
                    int cy = hy1 + (yy - py1);
                    if (cy < 0 || cy >= RES) continue;
                    for (int xx = px1; xx < px2 && xx < big.w; xx++) {
                        int cx = hx1 + (xx - px1);
                        if (cx < 0 || cx >= RES) continue;
                        std::copy(big.px(xx, yy), big.px(xx, yy) + 4, canvas.px(cx, cy));
                    }
                }
                v3_layers[HEAD_TAGS_V3[t]] = std::move(canvas);
            }
        }
    }
    for (auto & kv : v3_layers) alpha_floor(kv.second);

    if (!cfg.debug_dir.empty()) {
        // temporary: dump one full, UNCROPPED layer to see the raw alpha
        // spatial distribution before crop_part's bbox tightens it
        auto it = v3_layers.find("topwear");
        if (it != v3_layers.end()) {
            std::ofstream f(cfg.debug_dir + "/raw_topwear_full.png", std::ios::binary);
            std::vector<uint8_t> png = encode_png(it->second);
            f.write(reinterpret_cast<const char *>(png.data()), (std::streamsize) png.size());
            printf("[debug] dumped raw uncropped topwear layer (%dx%d)\n",
                   it->second.w, it->second.h);
        }
    }

    // ---- marigold: assemble the V2 list (compose eyes/hair), page last ----
    const std::map<std::string, std::vector<std::string>> COMPOSE = {
        { "eyes", { "eyewhite", "irides", "eyelash", "eyebrow" } },
        { "hair", { "back hair", "front hair" } },
    };
    std::vector<Image> v2_list;
    std::vector<float> blended_alpha((size_t) RES * RES, 0.0f);
    Image empty;
    empty.w = empty.h = RES; empty.c = 4;
    empty.data.assign((size_t) RES * RES * 4, 0.0f);

    for (const std::string & tag : VALID_BODY_PARTS_V2) {
        Image img = empty;
        auto cit = COMPOSE.find(tag);
        if (cit != COMPOSE.end()) {
            for (const std::string & sub : cit->second) {
                auto it = v3_layers.find(sub);
                if (it != v3_layers.end()) blend_over(img, it->second);
            }
        } else {
            auto it = v3_layers.find(tag);
            if (it != v3_layers.end()) img = it->second;
        }
        for (size_t i = 0; i < blended_alpha.size(); i++) blended_alpha[i] += img.data[i * 4 + 3];
        v2_list.push_back(std::move(img));
    }
    Image page_entry = fullpage;
    for (size_t i = 0; i < blended_alpha.size(); i++) {
        page_entry.data[i * 4 + 3] = std::min(1.0f, blended_alpha[i]);
    }
    v2_list.push_back(page_entry);

    std::vector<Image> depths;
    if (!marigold_depth(cfg, v2_list, depths)) return 1;
    for (Image & d : depths) {
        if (d.w != RES) d = smart_resize(d, RES, RES);
    }

    // ---- assign depths to v3 parts (compose depth_local reconstruction) ----
    std::map<std::string, Part> parts;
    auto make_part = [&](const std::string & tag, const Image & img, const Image & depth) {
        Part p;
        p.tag = tag;
        p.img = img;
        p.depth = depth;
        p.xyxy[0] = p.xyxy[1] = 0; p.xyxy[2] = RES; p.xyxy[3] = RES;
        crop_part(p);
        if (p.img.w > 0) parts[tag] = std::move(p);
    };
    for (size_t v = 0; v < VALID_BODY_PARTS_V2.size(); v++) {
        const std::string & tag = VALID_BODY_PARTS_V2[v];
        const Image & depth = depths[v];
        auto cit = COMPOSE.find(tag);
        if (cit == COMPOSE.end()) {
            auto it = v3_layers.find(tag);
            if (it != v3_layers.end()) make_part(tag, it->second, depth);
            continue;
        }
        // per-sub-tag depth_local: reverse order, occluded regions get the
        // visible-region median (upstream apply_marigold)
        std::vector<uint8_t> occl((size_t) RES * RES, 0);
        for (size_t i = 0; i < occl.size(); i++) occl[i] = blended_alpha[i] > 256.0f / 255.0f ? 1 : 0;
        std::vector<std::string> subs = cit->second;
        for (auto sit = subs.rbegin(); sit != subs.rend(); ++sit) {
            auto it = v3_layers.find(*sit);
            if (it == v3_layers.end()) continue;
            const Image & sub = it->second;
            Image dl;
            dl.w = dl.h = RES; dl.c = 1;
            dl.data.assign((size_t) RES * RES, 1.0f);
            std::vector<float> visible;
            for (size_t i = 0; i < dl.data.size(); i++) {
                bool local = sub.data[i * 4 + 3] > 15.0f / 255.0f;
                if (local) {
                    dl.data[i] = std::min(1.0f, std::max(0.0f, depth.data[i]));
                    if (!occl[i]) visible.push_back(dl.data[i]);
                }
            }
            std::sort(visible.begin(), visible.end());
            float med = visible.empty() ? 1.0f : visible[visible.size() / 2];
            for (size_t i = 0; i < dl.data.size(); i++) {
                bool local = sub.data[i * 4 + 3] > 15.0f / 255.0f;
                if (local && occl[i]) dl.data[i] = med;
                if (local) occl[i] = 1;
            }
            make_part(*sit, sub, dl);
        }
    }

    // ---- heuristics + PSD ----
    InpaintFn inpaint = make_lama_inpaint(cfg);
    further_extr_parts(parts, fullpage, inpaint);

    std::vector<const Part *> ordered;
    for (const auto & kv : parts) ordered.push_back(&kv.second);
    std::sort(ordered.begin(), ordered.end(),
              [](const Part * a, const Part * b) { return a->depth_median > b->depth_median; });

    // layered SVG: document order = z order (back to front); depth maps in
    // a hidden group; tag/depth_median/xyxy carried as data- attributes
    auto b64 = [](const std::vector<uint8_t> & d) {
        static const char * T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string o;
        o.reserve((d.size() + 2) / 3 * 4);
        for (size_t i = 0; i < d.size(); i += 3) {
            uint32_t v = d[i] << 16 | (i + 1 < d.size() ? d[i + 1] << 8 : 0)
                       | (i + 2 < d.size() ? d[i + 2] : 0);
            o += T[v >> 18]; o += T[(v >> 12) & 63];
            o += i + 1 < d.size() ? T[(v >> 6) & 63] : '=';
            o += i + 2 < d.size() ? T[v & 63] : '=';
        }
        return o;
    };
    // semantic identifier shared by --png-dir filenames and the SVG's
    // per-layer <image id="..."> attributes (e.g. ThorVG lookup by id)
    auto safe_id = [](const std::string & tag) {
        std::string s = tag;
        std::replace(s.begin(), s.end(), ' ', '_');
        return s;
    };
    // separate PNG layers with the same z order, one file per part
    if (!png_dir.empty()) {
        std::filesystem::create_directories(png_dir);
        int pzi = 0;
        for (const Part * p : ordered) {
            char name[32];
            std::snprintf(name, sizeof(name), "%03d_", pzi);
            std::string path = png_dir + "/" + name + safe_id(p->tag) + ".png";
            std::vector<uint8_t> png = encode_png(p->img);
            std::ofstream f(path, std::ios::binary);
            f.write(reinterpret_cast<const char *>(png.data()), (std::streamsize) png.size());
            pzi++;
        }
        printf("wrote %d PNG layers to %s\n", pzi, png_dir.c_str());
    }

    std::ofstream svg(out_path);
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" "
        << "xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"" << RES
        << "\" height=\"" << RES << "\">\n";
    int zi = 0;
    for (const Part * p : ordered) {
        std::string png = b64(encode_png(p->img));
        svg << "  <image id=\"" << safe_id(p->tag) << "\" x=\"" << p->xyxy[0]
            << "\" y=\"" << p->xyxy[1]
            << "\" width=\"" << p->img.w << "\" height=\"" << p->img.h
            << "\" data-tag=\"" << p->tag << "\" data-z=\"" << zi
            << "\" data-depth-median=\"" << p->depth_median
            << "\" xlink:href=\"data:image/png;base64," << png << "\"/>\n";
        zi++;
    }
    svg << "  <g id=\"depth\" display=\"none\">\n";
    zi = 0;
    for (const Part * p : ordered) {
        Image d1;
        d1.w = p->depth.w; d1.h = p->depth.h; d1.c = 1;
        d1.data = p->depth.data;
        svg << "    <image id=\"depth-" << safe_id(p->tag) << "\" x=\"" << p->xyxy[0]
            << "\" y=\"" << p->xyxy[1]
            << "\" width=\"" << d1.w << "\" height=\"" << d1.h
            << "\" data-tag=\"" << p->tag << "\" data-z=\"" << zi
            << "\" xlink:href=\"data:image/png;base64," << b64(encode_png(d1))
            << "\"/>\n";
        zi++;
    }
    svg << "  </g>\n</svg>\n";
    printf("wrote %s (%zu layers)\n", out_path.c_str(), ordered.size());
    return 0;
}
