// Image containers and cv2-compatible primitives for the pipeline: stb PNG
// I/O, INTER_LINEAR / INTER_AREA resizes (matching OpenCV's sampling),
// smart_resize / center_square_pad_resize, and the LaMa-style pyramid RGB
// bleed (pad_rgb).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// interleaved float image, values [0,1]
struct Image {
    int w = 0, h = 0, c = 0;
    std::vector<float> data;      // h*w*c

    float * px(int x, int y) { return data.data() + ((size_t) y * w + x) * c; }
    const float * px(int x, int y) const { return data.data() + ((size_t) y * w + x) * c; }
};

bool  load_image(const std::string & path, Image & out, int channels = 4);
bool  load_image_from_memory(const uint8_t * data, size_t len, Image & out, int channels = 4);
bool  save_image(const std::string & path, const Image & img);
// PNG-encode in memory (for data: URIs)
std::vector<uint8_t> encode_png(const Image & img);

// cv2.resize equivalents
Image resize_linear(const Image & src, int tw, int th);
Image resize_area(const Image & src, int tw, int th);
// INTER_AREA when shrinking, INTER_LINEAR when growing (upstream smart_resize)
Image smart_resize(const Image & src, int tw, int th);

// pad to square with pad_value then smart_resize to target; pad_pos receives
// the (x,y) offset of the original image inside the square
Image center_square_pad_resize(const Image & src, int target, float pad_value,
                               int * pad_w = nullptr, int * pad_h = nullptr,
                               int * pad_x = nullptr, int * pad_y = nullptr);

// upstream pad_rgb (keep_ori_pixel=true): bleed RGB into transparent regions
// via an INTER_AREA pyramid (factor 1.2); rgba in, 3-channel rgb out
Image pad_rgb(const Image & rgba);
