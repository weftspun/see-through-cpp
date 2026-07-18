#include "image_utils.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <algorithm>
#include <cmath>

bool load_image(const std::string & path, Image & out, int channels) {
    int w, h, n;
    uint8_t * px = stbi_load(path.c_str(), &w, &h, &n, channels);
    if (!px) return false;
    out.w = w; out.h = h; out.c = channels;
    out.data.resize((size_t) w * h * channels);
    for (size_t i = 0; i < out.data.size(); i++) out.data[i] = px[i] / 255.0f;
    stbi_image_free(px);
    return true;
}

bool save_image(const std::string & path, const Image & img) {
    std::vector<uint8_t> px(img.data.size());
    for (size_t i = 0; i < px.size(); i++) {
        float v = img.data[i] * 255.0f;
        px[i] = (uint8_t) std::min(255.0f, std::max(0.0f, std::round(v)));
    }
    return stbi_write_png(path.c_str(), img.w, img.h, img.c, px.data(), img.w * img.c) != 0;
}

Image resize_linear(const Image & src, int tw, int th) {
    Image dst;
    dst.w = tw; dst.h = th; dst.c = src.c;
    dst.data.resize((size_t) tw * th * src.c);
    const double sx = (double) src.w / tw, sy = (double) src.h / th;
    for (int y = 0; y < th; y++) {
        double fy = (y + 0.5) * sy - 0.5;
        int y0 = (int) std::floor(fy);
        double wy = fy - y0;
        int y0c = std::min(std::max(y0, 0), src.h - 1);
        int y1c = std::min(std::max(y0 + 1, 0), src.h - 1);
        for (int x = 0; x < tw; x++) {
            double fx = (x + 0.5) * sx - 0.5;
            int x0 = (int) std::floor(fx);
            double wx = fx - x0;
            int x0c = std::min(std::max(x0, 0), src.w - 1);
            int x1c = std::min(std::max(x0 + 1, 0), src.w - 1);
            const float * p00 = src.px(x0c, y0c), * p10 = src.px(x1c, y0c);
            const float * p01 = src.px(x0c, y1c), * p11 = src.px(x1c, y1c);
            float * d = dst.px(x, y);
            for (int c = 0; c < src.c; c++) {
                d[c] = (float) ((p00[c] * (1 - wx) + p10[c] * wx) * (1 - wy) +
                                (p01[c] * (1 - wx) + p11[c] * wx) * wy);
            }
        }
    }
    return dst;
}

Image resize_area(const Image & src, int tw, int th) {
    Image dst;
    dst.w = tw; dst.h = th; dst.c = src.c;
    dst.data.resize((size_t) tw * th * src.c);
    const double sx = (double) src.w / tw, sy = (double) src.h / th;
    for (int y = 0; y < th; y++) {
        double fy0 = y * sy, fy1 = (y + 1) * sy;
        int iy0 = (int) std::floor(fy0), iy1 = std::min((int) std::ceil(fy1), src.h);
        for (int x = 0; x < tw; x++) {
            double fx0 = x * sx, fx1 = (x + 1) * sx;
            int ix0 = (int) std::floor(fx0), ix1 = std::min((int) std::ceil(fx1), src.w);
            float * d = dst.px(x, y);
            for (int c = 0; c < src.c; c++) d[c] = 0.0f;
            double area = 0;
            for (int yy = iy0; yy < iy1; yy++) {
                double wy = std::min((double) yy + 1, fy1) - std::max((double) yy, fy0);
                for (int xx = ix0; xx < ix1; xx++) {
                    double wx = std::min((double) xx + 1, fx1) - std::max((double) xx, fx0);
                    const float * s = src.px(xx, yy);
                    for (int c = 0; c < src.c; c++) d[c] += (float) (s[c] * wx * wy);
                    area += wx * wy;
                }
            }
            for (int c = 0; c < src.c; c++) d[c] = (float) (d[c] / area);
        }
    }
    return dst;
}

Image smart_resize(const Image & src, int tw, int th) {
    if (tw == src.w && th == src.h) return src;
    if ((int64_t) tw * th < (int64_t) src.w * src.h) return resize_area(src, tw, th);
    return resize_linear(src, tw, th);
}

Image center_square_pad_resize(const Image & src, int target, float pad_value,
                               int * pad_w, int * pad_h, int * pad_x, int * pad_y) {
    Image sq = src;
    int px1 = 0, py1 = 0;
    if (src.w != src.h) {
        int sz = std::max(src.w, src.h);
        px1 = (sz - src.w) / 2;
        py1 = (sz - src.h) / 2;
        sq.w = sq.h = sz;
        sq.data.assign((size_t) sz * sz * src.c, pad_value);
        for (int y = 0; y < src.h; y++) {
            std::copy(src.px(0, y), src.px(0, y) + (size_t) src.w * src.c,
                      sq.data.begin() + (((size_t) (y + py1) * sz + px1) * src.c));
        }
    }
    if (pad_w) *pad_w = sq.w;
    if (pad_h) *pad_h = sq.h;
    if (pad_x) *pad_x = px1;
    if (pad_y) *pad_y = py1;
    if (sq.w != target) sq = smart_resize(sq, target, target);
    return sq;
}

Image pad_rgb(const Image & rgba) {
    // pyramid of premultiplied color + alpha, INTER_AREA /1.2 down to 1px
    struct Level { Image pc, a; };
    std::vector<Level> pyr;
    Level cur;
    cur.pc.w = cur.a.w = rgba.w; cur.pc.h = cur.a.h = rgba.h;
    cur.pc.c = 3; cur.a.c = 1;
    cur.pc.data.resize((size_t) rgba.w * rgba.h * 3);
    cur.a.data.resize((size_t) rgba.w * rgba.h);
    for (size_t i = 0; i < cur.a.data.size(); i++) {
        float a = rgba.data[i * 4 + 3];
        cur.a.data[i] = a;
        for (int c = 0; c < 3; c++) cur.pc.data[i * 3 + c] = rgba.data[i * 4 + c] * a;
    }
    for (;;) {
        pyr.push_back(cur);
        int W = cur.pc.w, H = cur.pc.h;
        if (std::min(W, H) == 1) break;
        int nw = (int) (W / 1.2), nh = (int) (H / 1.2);
        cur.pc = resize_area(cur.pc, nw, nh);
        cur.a  = resize_area(cur.a, nw, nh);
    }

    // fg starts as the alpha-weighted mean color of the top level
    const Level & top = pyr.back();
    double sum_c[3] = { 0, 0, 0 }, sum_a = 0;
    for (size_t i = 0; i < top.a.data.size(); i++) {
        sum_a += top.a.data[i];
        for (int c = 0; c < 3; c++) sum_c[c] += top.pc.data[i * 3 + c];
    }
    sum_a = std::max(sum_a, 1e-8);
    Image fg;
    fg.w = fg.h = 1; fg.c = 3;
    fg.data = { (float) (sum_c[0] / sum_a), (float) (sum_c[1] / sum_a), (float) (sum_c[2] / sum_a) };

    for (auto it = pyr.rbegin(); it != pyr.rend(); ++it) {
        fg = resize_linear(fg, it->pc.w, it->pc.h);
        for (size_t i = 0; i < it->a.data.size(); i++) {
            float a = it->a.data[i];
            for (int c = 0; c < 3; c++) {
                fg.data[i * 3 + c] = it->pc.data[i * 3 + c] + fg.data[i * 3 + c] * (1.0f - a);
            }
        }
    }

    // keep_ori_pixel: composite original over the bleed
    Image out = fg;
    for (size_t i = 0; i < (size_t) rgba.w * rgba.h; i++) {
        float a = rgba.data[i * 4 + 3];
        for (int c = 0; c < 3; c++) {
            float v = a * rgba.data[i * 4 + c] + (1.0f - a) * fg.data[i * 3 + c];
            out.data[i * 3 + c] = std::min(1.0f, std::max(0.0f, v));
        }
    }
    return out;
}
