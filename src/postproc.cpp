#include "postproc.h"

#include <algorithm>
#include <cmath>
#include <numeric>

// ---------------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------------

int connected_components(const std::vector<uint8_t> & mask, int w, int h,
                         std::vector<int> & labels, std::vector<CCStats> & stats) {
    labels.assign((size_t) w * h, 0);
    std::vector<int> parent(1, 0);
    auto find = [&](int a) { while (parent[a] != a) a = parent[a] = parent[parent[a]]; return a; };

    // two-pass union-find, 8-connectivity
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (!mask[(size_t) y * w + x]) continue;
            int best = 0;
            const int nx[4] = { x - 1, x - 1, x, x + 1 };
            const int ny[4] = { y, y - 1, y - 1, y - 1 };
            for (int k = 0; k < 4; k++) {
                if (nx[k] < 0 || nx[k] >= w || ny[k] < 0) continue;
                int l = labels[(size_t) ny[k] * w + nx[k]];
                if (l && (best == 0 || find(l) < find(best))) {
                    if (best) { parent[find(std::max(l, best))] = find(std::min(l, best)); }
                    best = l;
                } else if (l && best && find(l) != find(best)) {
                    parent[find(std::max(l, best))] = find(std::min(l, best));
                }
            }
            if (!best) {
                best = (int) parent.size();
                parent.push_back(best);
            }
            labels[(size_t) y * w + x] = best;
        }
    }
    // flatten + renumber (background 0 first, then discovery order like cv2)
    std::vector<int> remap(parent.size(), -1);
    int next = 1;
    for (size_t i = 0; i < labels.size(); i++) {
        if (!labels[i]) continue;
        int r = find(labels[i]);
        if (remap[r] < 0) remap[r] = next++;
        labels[i] = remap[r];
    }
    stats.assign(next, { w, h, 0, 0, 0 });
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int l = labels[(size_t) y * w + x];
            CCStats & s = stats[l];
            s.x = std::min(s.x, x); s.y = std::min(s.y, y);
            s.w = std::max(s.w, x); s.h = std::max(s.h, y);
            s.area++;
        }
    }
    for (CCStats & s : stats) { s.w = s.w - s.x + 1; s.h = s.h - s.y + 1; }
    if (next > 0) stats[0] = { 0, 0, w, h, stats[0].area };   // background full frame
    return next;
}

double kmeans2_split(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    if (n < 2) return v.empty() ? 0.0 : v[0];
    std::vector<double> pre(n + 1, 0), pre2(n + 1, 0);
    for (size_t i = 0; i < n; i++) {
        pre[i + 1] = pre[i] + v[i];
        pre2[i + 1] = pre2[i] + (double) v[i] * v[i];
    }
    double best = 1e300;
    size_t bi = 1;
    for (size_t i = 1; i < n; i++) {           // split: [0,i) | [i,n)
        double s1 = pre[i], s2 = pre[n] - pre[i];
        double q1 = pre2[i], q2 = pre2[n] - pre2[i];
        double cost = (q1 - s1 * s1 / i) + (q2 - s2 * s2 / (n - i));
        if (cost < best) { best = cost; bi = i; }
    }
    return (v[bi - 1] + v[bi]) / 2.0;
}

void dilate_ellipse(std::vector<uint8_t> & mask, int w, int h, int k) {
    const int d = 2 * k + 1;
    // cv2 ellipse kernel membership
    std::vector<uint8_t> ker((size_t) d * d, 0);
    double r = k + 0.5;
    for (int y = 0; y < d; y++) {
        double dy = y - k;
        double dx = r * sqrt(std::max(0.0, 1.0 - dy * dy / (r * r)));
        int x0 = (int) std::lround(k - dx), x1 = (int) std::lround(k + dx);
        for (int x = std::max(0, x0); x <= std::min(d - 1, x1); x++) ker[(size_t) y * d + x] = 1;
    }
    std::vector<uint8_t> out((size_t) w * h, 0);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (!mask[(size_t) y * w + x]) continue;
            for (int ky = -k; ky <= k; ky++) {
                int yy = y + ky;
                if (yy < 0 || yy >= h) continue;
                for (int kx = -k; kx <= k; kx++) {
                    int xx = x + kx;
                    if (xx < 0 || xx >= w) continue;
                    if (ker[(size_t) (ky + k) * d + (kx + k)]) out[(size_t) yy * w + xx] = 255;
                }
            }
        }
    }
    mask = std::move(out);
}

void gaussian3_blur(std::vector<float> & buf, int w, int h) {
    // cv2 sigma 0.7 3-tap kernel
    const double s = 0.7;
    double k0 = exp(-0.5 / (s * s)), kc = 1.0;
    double norm = kc + 2 * k0;
    const double kk[3] = { k0 / norm, kc / norm, k0 / norm };
    std::vector<float> tmp(buf.size());
    auto clampi = [](int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; };
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double a = 0;
            for (int t = -1; t <= 1; t++) a += kk[t + 1] * buf[(size_t) y * w + clampi(x + t, 0, w - 1)];
            tmp[(size_t) y * w + x] = (float) a;
        }
    }
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double a = 0;
            for (int t = -1; t <= 1; t++) a += kk[t + 1] * tmp[(size_t) clampi(y + t, 0, h - 1) * w + x];
            buf[(size_t) y * w + x] = (float) a;
        }
    }
}

static double median_of(std::vector<float> v) {
    if (v.empty()) return 1.0;
    size_t k = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + k, v.end());
    double m = v[k];
    if (v.size() % 2 == 0) {
        double lo = *std::max_element(v.begin(), v.begin() + k);
        m = (m + lo) / 2.0;
    }
    return m;
}

void crop_part(Part & p) {
    // largest-connected-component bbox, not naive min/max: the raw decode
    // carries a small (~10px, low-alpha) corner artifact in every layer (see
    // pipeline.cpp's bbox_alpha_largest_cc), which otherwise inflates this
    // tag's crop rectangle out to the canvas corner -- the same failure mode
    // that corrupted the head-crop bbox, just at this call site instead.
    std::vector<uint8_t> mask((size_t) p.img.w * p.img.h, 0);
    for (int y = 0; y < p.img.h; y++) {
        for (int x = 0; x < p.img.w; x++) {
            if (p.img.px(x, y)[3] > 10.0f / 255.0f) { mask[(size_t) y * p.img.w + x] = 1; }
        }
    }
    std::vector<int> labels;
    std::vector<CCStats> stats;
    int n = connected_components(mask, p.img.w, p.img.h, labels, stats);
    int best = 0;
    for (int l = 1; l < n; l++) {
        if (stats[l].area > (best ? stats[best].area : 0)) { best = l; }
    }
    std::vector<float> dvals;
    int x0 = 0, y0 = 0, x1 = -1, y1 = -1;
    if (best) {
        CCStats & s = stats[best];
        x0 = s.x; y0 = s.y; x1 = s.x + s.w - 1; y1 = s.y + s.h - 1;
        for (int y = s.y; y < s.y + s.h; y++) {
            for (int x = s.x; x < s.x + s.w; x++) {
                if (labels[(size_t) y * p.img.w + x] == best) { dvals.push_back(p.depth.px(x, y)[0]); }
            }
        }
    }
    p.depth_median = median_of(std::move(dvals));
    if (x1 < 0) return;
    x1++; y1++;
    Image img2, dep2;
    img2.w = x1 - x0; img2.h = y1 - y0; img2.c = 4;
    dep2.w = img2.w; dep2.h = img2.h; dep2.c = 1;
    img2.data.resize((size_t) img2.w * img2.h * 4);
    dep2.data.resize((size_t) dep2.w * dep2.h);
    for (int y = y0; y < y1; y++) {
        std::copy(p.img.px(x0, y), p.img.px(x0, y) + (size_t) img2.w * 4,
                  img2.data.begin() + (size_t) (y - y0) * img2.w * 4);
        std::copy(p.depth.px(x0, y), p.depth.px(x0, y) + dep2.w,
                  dep2.data.begin() + (size_t) (y - y0) * dep2.w);
    }
    p.xyxy[2] = p.xyxy[0] + x1; p.xyxy[3] = p.xyxy[1] + y1;
    p.xyxy[0] += x0; p.xyxy[1] += y0;
    p.img = std::move(img2);
    p.depth = std::move(dep2);

    // upstream load_part (io_utils.py:521-595, further_extr's actual entry
    // point -- every part passes through this before anything else): after
    // cropping, trim the trailing max(w,h)//10 rows+cols and require >4
    // real (alpha>10/255) pixels in what remains, else the part is dropped
    // entirely (`return None`) -- a crop that's only real near its own
    // trailing edge is a corner-hugging artifact, not real content.
    // Faithfully reproduces Python's mask[:-p_test, :-p_test] slicing: when
    // p_test == 0 (crop under 10px in its largest dimension -- exactly the
    // case for a spurious few-px corner-artifact "part") that slice is
    // mask[:0, :0], i.e. EMPTY, not "no trim" -- so any such tiny crop
    // unconditionally fails and gets dropped, regardless of its content.
    int p_test = std::max(p.img.w, p.img.h) / 10;
    int iw = p_test > 0 ? std::max(0, p.img.w - p_test) : 0;
    int ih = p_test > 0 ? std::max(0, p.img.h - p_test) : 0;
    int cnt = 0;
    for (int y = 0; y < ih && cnt <= 4; y++) {
        for (int x = 0; x < iw && cnt <= 4; x++) {
            if (p.img.px(x, y)[3] > 10.0f / 255.0f) { cnt++; }
        }
    }
    if (cnt <= 4) { p.img.w = 0; p.img.h = 0; p.img.data.clear(); }
}

// ---------------------------------------------------------------------------
// upstream flows
// ---------------------------------------------------------------------------

// process_cuts: crop img/depth to a component bbox, optionally masking with a
// component mask (dilate 1) and recomputing depth
static Part process_cuts(const Part & src, const CCStats & bb,
                         const std::vector<uint8_t> * comp_mask, const std::string & tag) {
    Part out;
    out.tag = tag;
    int tx1 = bb.x, ty1 = bb.y, tx2 = bb.x + bb.w, ty2 = bb.y + bb.h;
    out.img.w = tx2 - tx1; out.img.h = ty2 - ty1; out.img.c = 4;
    out.depth.w = out.img.w; out.depth.h = out.img.h; out.depth.c = 1;
    out.img.data.resize((size_t) out.img.w * out.img.h * 4);
    out.depth.data.resize((size_t) out.depth.w * out.depth.h);
    for (int y = ty1; y < ty2; y++) {
        std::copy(src.img.px(tx1, y), src.img.px(tx1, y) + (size_t) out.img.w * 4,
                  out.img.data.begin() + (size_t) (y - ty1) * out.img.w * 4);
        std::copy(src.depth.px(tx1, y), src.depth.px(tx1, y) + out.depth.w,
                  out.depth.data.begin() + (size_t) (y - ty1) * out.depth.w);
    }
    out.depth_median = 1.0;
    if (comp_mask) {
        std::vector<uint8_t> m((size_t) out.img.w * out.img.h);
        for (int y = 0; y < out.img.h; y++) {
            for (int x = 0; x < out.img.w; x++) {
                m[(size_t) y * out.img.w + x] =
                    (*comp_mask)[(size_t) (y + ty1) * src.img.w + (x + tx1)] > 15 ? 1 : 0;
            }
        }
        dilate_ellipse(m, out.img.w, out.img.h, 1);
        std::vector<float> dvals;
        for (size_t i = 0; i < m.size(); i++) {
            float mm = m[i] ? 1.0f : 0.0f;
            out.img.data[i * 4 + 3] *= mm;
            out.depth.data[i] = 1.0f - (1.0f - out.depth.data[i]) * mm;
            if (m[i]) dvals.push_back(out.depth.data[i]);
        }
        out.depth_median = median_of(std::move(dvals));
    }
    out.xyxy[0] = tx1 + src.xyxy[0]; out.xyxy[1] = ty1 + src.xyxy[1];
    out.xyxy[2] = tx2 + src.xyxy[0]; out.xyxy[3] = ty2 + src.xyxy[1];
    return out;
}

// masks for the two largest components ordered left/right by center x
struct LrSplit {
    std::vector<uint8_t> mask_l, mask_r;
    CCStats stat_l, stat_r;
};

static bool label_lr_split(const std::vector<int> & labels, const std::vector<CCStats> & stats,
                           int id1, int id2, int w, int h, LrSplit & out) {
    double x1 = stats[id1].x + stats[id1].w / 2.0;
    double x2 = stats[id2].x + stats[id2].w / 2.0;
    int li = id1, ri = id2;
    if (x2 < x1) std::swap(li, ri);
    out.mask_l.assign((size_t) w * h, 0);
    out.mask_r.assign((size_t) w * h, 0);
    for (size_t i = 0; i < labels.size(); i++) {
        if (labels[i] == li) out.mask_l[i] = 255;
        if (labels[i] == ri) out.mask_r[i] = 255;
    }
    out.stat_l = stats[li];
    out.stat_r = stats[ri];
    return true;
}

// labels sorted by area desc, background (largest, label of full frame) dropped
static std::vector<int> area_order(const std::vector<CCStats> & stats) {
    std::vector<int> idx(stats.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
                     [&](int a, int b) { return stats[a].area > stats[b].area; });
    return std::vector<int>(idx.begin() + 1, idx.end());
}

static void part_mask(const Part & p, std::vector<uint8_t> & m) {
    m.resize((size_t) p.img.w * p.img.h);
    for (size_t i = 0; i < m.size(); i++) {
        m[i] = p.img.data[i * 4 + 3] > 10.0f / 255.0f ? 255 : 0;
    }
}

static void tag_lr_split(const std::string & tag, std::map<std::string, Part> & parts) {
    auto it = parts.find(tag);
    if (it == parts.end()) return;
    Part p = it->second;
    std::vector<uint8_t> m;
    part_mask(p, m);
    std::vector<int> labels;
    std::vector<CCStats> stats;
    int n = connected_components(m, p.img.w, p.img.h, labels, stats);
    if (n <= 2) return;                        // keep as one part
    parts.erase(it);
    std::vector<int> order = area_order(stats);
    LrSplit lr;
    label_lr_split(labels, stats, order[0], order[1], p.img.w, p.img.h, lr);
    parts[tag + "-r"] = process_cuts(p, lr.stat_l, &lr.mask_l, tag + "-r");
    parts[tag + "-l"] = process_cuts(p, lr.stat_r, &lr.mask_r, tag + "-l");
}

std::vector<Part> cluster_inpaint_part(const Part & part, const InpaintFn & inpaint) {
    const int w = part.img.w, h = part.img.h;
    std::vector<uint8_t> mask;
    part_mask(part, mask);

    // normalized depth (uint8-quantized like upstream) + exact 2-means
    std::vector<float> dvals;
    float dmin = 1e9f, dmax = -1e9f;
    for (size_t i = 0; i < mask.size(); i++) {
        if (mask[i]) { dmin = std::min(dmin, part.depth.data[i]); dmax = std::max(dmax, part.depth.data[i]); }
    }
    std::vector<uint8_t> dq(mask.size());
    for (size_t i = 0; i < mask.size(); i++) {
        dq[i] = (uint8_t) std::lround((part.depth.data[i] - dmin) / (dmax - dmin + 1e-6f) * 255.0f);
        if (mask[i]) dvals.push_back(dq[i] / 255.0f);
    }
    double thr = kmeans2_split(dvals);

    Image rgb;
    rgb.w = w; rgb.h = h; rgb.c = 3;
    rgb.data.resize((size_t) w * h * 3);
    std::vector<float> alpha((size_t) w * h);
    for (size_t i = 0; i < alpha.size(); i++) {
        alpha[i] = part.img.data[i * 4 + 3];
        for (int c = 0; c < 3; c++) rgb.data[i * 3 + c] = part.img.data[i * 4 + c];
    }

    std::vector<Part> out;

    // front cluster (lower depth center) is masked out and inpainted away
    std::vector<uint8_t> to_mask(mask.size(), 0);
    for (size_t i = 0; i < mask.size(); i++) {
        if (mask[i] && dq[i] / 255.0f < thr) to_mask[i] = 255;
    }
    std::vector<uint8_t> imask_inpaint = to_mask;
    dilate_ellipse(imask_inpaint, w, h, 3);
    std::vector<float> imask(to_mask.size());
    for (size_t i = 0; i < imask.size(); i++) imask[i] = to_mask[i];
    gaussian3_blur(imask, w, h);

    // extracted front part: original rgb + blurred cluster mask as alpha
    Part front;
    front.tag = "front";
    front.img.w = w; front.img.h = h; front.img.c = 4;
    front.img.data.resize((size_t) w * h * 4);
    front.depth = part.depth;
    std::vector<float> dmed_front;
    for (size_t i = 0; i < imask.size(); i++) {
        for (int c = 0; c < 3; c++) front.img.data[i * 4 + c] = rgb.data[i * 3 + c];
        front.img.data[i * 4 + 3] = imask[i] / 255.0f;
        if (to_mask[i]) dmed_front.push_back(part.depth.data[i]);
    }
    front.depth_median = median_of(std::move(dmed_front));
    std::copy(part.xyxy, part.xyxy + 4, front.xyxy);
    out.push_back(std::move(front));

    // fill the removed region: composite over black/white, LaMa-inpaint, and
    // rebuild alpha from the distance to the fill color
    std::vector<uint8_t> valid(mask.size());
    double rgb_mean = 0;
    size_t nvalid = 0;
    for (size_t i = 0; i < mask.size(); i++) {
        double a255 = alpha[i] * 255.0, m255 = imask[i];
        valid[i] = std::max(0.0, a255 - m255) > 50 ? 1 : 0;
        if (valid[i]) {
            rgb_mean += (rgb.data[i * 3] + rgb.data[i * 3 + 1] + rgb.data[i * 3 + 2]) / 3.0 * 255.0;
            nvalid++;
        }
    }
    const float fill = (nvalid ? rgb_mean / nvalid : 255.0) < 100.0 ? 1.0f : 0.0f;
    for (size_t i = 0; i < alpha.size(); i++) {
        for (int c = 0; c < 3; c++) {
            float v = rgb.data[i * 3 + c] * alpha[i] + (1.0f - alpha[i]) * fill;
            rgb.data[i * 3 + c] = std::round(v * 255.0f) / 255.0f;
        }
    }
    rgb = inpaint(rgb, imask_inpaint);
    for (size_t i = 0; i < alpha.size(); i++) {
        double dist = 0;
        for (int c = 0; c < 3; c++) dist += fabs(rgb.data[i * 3 + c] - fill) * 255.0 / 3.0;
        double a = dist > 15.0 ? 255.0 : dist;
        if (imask_inpaint[i] <= 127) a = alpha[i] * 255.0;
        alpha[i] = (float) (std::lround(a) / 255.0);
    }

    // back part = inpainted rgb + reconstructed alpha; depth of the removed
    // region replaced by the valid-region median
    Part back;
    back.tag = "back";
    back.img.w = w; back.img.h = h; back.img.c = 4;
    back.img.data.resize((size_t) w * h * 4);
    back.depth = part.depth;
    std::vector<float> dvalid, dback;
    for (size_t i = 0; i < mask.size(); i++) {
        if (valid[i]) dvalid.push_back(dq[i] / 255.0f);
    }
    double dmed_valid = median_of(std::move(dvalid));
    for (size_t i = 0; i < mask.size(); i++) {
        for (int c = 0; c < 3; c++) back.img.data[i * 4 + c] = rgb.data[i * 3 + c];
        back.img.data[i * 4 + 3] = alpha[i];
        float dqv = imask_inpaint[i] > 127 ? (float) dmed_valid : dq[i] / 255.0f;
        back.depth.data[i] = dqv * (dmax - dmin + 1e-6f) + dmin;
        if (mask[i] && dq[i] / 255.0f >= thr) dback.push_back(part.depth.data[i]);
    }
    back.depth_median = median_of(std::move(dback));
    std::copy(part.xyxy, part.xyxy + 4, back.xyxy);
    out.push_back(std::move(back));
    return out;
}

void further_extr_parts(std::map<std::string, Part> & parts, const Image & fullpage,
                        const InpaintFn & inpaint, unsigned partseg_flags,
                        const std::vector<std::string> & depth_split_tags,
                        const std::vector<std::string> & lr_split_tags) {
    if (partseg_flags & PARTSEG_LR) {
        for (const std::string & tag : lr_split_tags) {
            tag_lr_split(tag, parts);
        }
    }

    if (partseg_flags & PARTSEG_DEPTH) {
        // missing tag / degenerate 2-means (e.g. a single-pixel target) never
        // crashes: cluster_inpaint_part always returns exactly two Parts,
        // each a valid (possibly fully-transparent) w*h buffer -- no bbox
        // recompute happens on this path, unlike the LR-split's process_cuts.
        for (const std::string & tag : depth_split_tags) {
            auto it = parts.find(tag);
            if (it == parts.end()) continue;
            Part p = it->second;
            parts.erase(it);
            std::vector<Part> split = cluster_inpaint_part(p, inpaint);
            std::sort(split.begin(), split.end(),
                      [](const Part & a, const Part & b) { return a.depth_median < b.depth_median; });
            // "hair" keeps its historical hairf/hairb tag names (external
            // consumers of already-shipped SVGs may depend on them); any
            // other target tag gets the generic <tag>-front/<tag>-back names.
            std::string front_tag = (tag == "hair") ? "hairf" : tag + "-front";
            std::string back_tag  = (tag == "hair") ? "hairb" : tag + "-back";
            split[0].tag = front_tag;
            split[1].tag = back_tag;
            parts[front_tag] = std::move(split[0]);
            parts[back_tag]  = std::move(split[1]);
        }
    }

    for (const char * tag : { "nose", "mouth" }) {
        auto it = parts.find(tag);
        if (it == parts.end()) continue;
        Part & p = it->second;
        for (int y = 0; y < p.img.h; y++) {
            for (int x = 0; x < p.img.w; x++) {
                const float * src = fullpage.px(p.xyxy[0] + x, p.xyxy[1] + y);
                float * dst = p.img.px(x, y);
                for (int c = 0; c < 3; c++) dst[c] = src[c];
            }
        }
    }

    auto face = parts.find("face");
    if (face != parts.end()) {
        // "eyes" is never a key by this point -- tag_lr_split() above already
        // explodes eyewhite/irides/eyelash/eyebrow into their own parts (and
        // further into -l/-r variants when it finds two components), so the
        // clamp has to name every one of those or it silently no-ops and
        // leaves eye layers at their raw (often noisy-for-small-regions)
        // marigold depth median, letting them sort behind the face. If a
        // caller's lr_split_tags excludes one of these tags (or PARTSEG_LR
        // is off), the corresponding parts.find() lookups below simply miss
        // and no-op -- same as any other missing tag, not a special case.
        // Clamped tags used to all collapse onto the SAME depth_median
        // (face_median - 0.001), which erased their relative depth signal:
        // the final sort (pipeline.cpp) then had to break ties among
        // identically-valued parts, and since it used std::sort (not
        // stable), that tie-break was unspecified rather than reflecting
        // any real front/back relationship -- it only happened to look like
        // "alphabetical by tag name" because it rode on parts' pre-sort
        // std::map key order. Give each tag in the clamp group a distinct
        // offset instead, ordered back-to-front by anatomy: eyewhite (base
        // of the eye) is furthest back, eyebrow is frontmost; nose/mouth
        // don't overlap the eye stack so their relative slot doesn't matter
        // visually, but still gets a distinct value for determinism.
        const char * clamp_order[] = {
            "eyewhite", "eyewhite-l", "eyewhite-r",
            "irides", "irides-l", "irides-r",
            "eyelash", "eyelash-l", "eyelash-r",
            "eyebrow", "eyebrow-l", "eyebrow-r",
            "nose", "mouth",
        };
        const double eps = 1e-4;
        int rank = 0;
        for (const char * t : clamp_order) {
            auto it = parts.find(t);
            if (it != parts.end() && it->second.depth_median > face->second.depth_median) {
                it->second.depth_median = face->second.depth_median - 0.001 - rank * eps;
            }
            rank++;
        }
        const char * ear_order[] = { "ears", "earl", "earr" };
        rank = 0;
        for (const char * t : ear_order) {
            auto it = parts.find(t);
            if (it != parts.end()) {
                it->second.depth_median = face->second.depth_median + 0.001 + rank * eps;
            }
            rank++;
        }
    }
}
