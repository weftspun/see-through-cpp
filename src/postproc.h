// further_extr heuristics (upstream inference_utils/torchcv): connected
// components, exact 1-D 2-means, L/R part splitting, hair depth-cluster split
// with LaMa inpainting, nose/mouth copyback, depth-median ordering.
#pragma once

#include "image_utils.h"

#include <functional>
#include <map>
#include <string>
#include <vector>

// bit flags for further_extr_parts's `partseg_flags` parameter
enum PartsegFlags : unsigned {
    PARTSEG_DEPTH = 1u << 0,   // enable hair-style depth (2-means) split
    PARTSEG_LR    = 1u << 1,   // enable left/right connected-component split
};

struct Part {
    std::string tag;
    Image  img;                 // RGBA float [0,1], cropped to xyxy
    Image  depth;               // 1ch float [0,1], same crop
    int    xyxy[4] = { 0, 0, 0, 0 };   // in fullpage coords
    double depth_median = 1.0;
};

// cv2.connectedComponentsWithStats(8-conn) over mask>0: label 0 = background;
// stats rows = {x, y, w, h, area} per label
struct CCStats { int x, y, w, h, area; };
int connected_components(const std::vector<uint8_t> & mask, int w, int h,
                         std::vector<int> & labels, std::vector<CCStats> & stats);

// exact 1-D 2-means (deterministic replacement for sklearn KMeans, documented
// divergence): returns the split threshold; values < threshold are cluster 0
// (lower center)
double kmeans2_split(std::vector<float> values);

// binary dilate with the cv2 MORPH_ELLIPSE structuring element of "ksize" k
// (kernel (2k+1)x(2k+1))
void dilate_ellipse(std::vector<uint8_t> & mask, int w, int h, int k);

// 3x3 Gaussian blur, sigma 0.7, on a [0,255] float buffer (cv2.GaussianBlur)
void gaussian3_blur(std::vector<float> & buf, int w, int h);

// inpaint callback: (rgb float [0,1] HWC interleaved 3ch, mask uint8 0/255) ->
// inpainted rgb, both w*h; wired to the LaMa graph by the caller
using InpaintFn = std::function<Image(const Image & rgb, const std::vector<uint8_t> & mask)>;

// upstream cluster_inpaint_part: KMeans-on-depth split of a part into
// front/back with LaMa inpainting of the removed region
std::vector<Part> cluster_inpaint_part(const Part & part, const InpaintFn & inpaint);

// full further_extr heuristic pass over decoded parts (v3 tags), in place:
// L/R connected-component split (if PARTSEG_LR set, over lr_split_tags),
// depth-cluster front/back split with inpaint (if PARTSEG_DEPTH set, over
// depth_split_tags), nose/mouth RGB copyback from fullpage, face depth
// nudges (always runs, independent of the flags).
void further_extr_parts(std::map<std::string, Part> & parts, const Image & fullpage,
                        const InpaintFn & inpaint, unsigned partseg_flags,
                        const std::vector<std::string> & depth_split_tags,
                        const std::vector<std::string> & lr_split_tags);

// crop a part to its alpha>10/255 bounding box and set depth_median
void crop_part(Part & p);
