// Minimal 8BPS writer: version-1 RGBA PSD with named pixel layers (PackBits
// RLE channels, unicode 'luni' names) and a raw merged composite — the
// psd-tools-compatible subset produced by upstream save_psd.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct PsdLayer {
    std::string name;
    int top = 0, left = 0;        // canvas offset
    int w = 0, h = 0;
    std::vector<uint8_t> rgba;    // interleaved RGBA, h*w*4
};

// composite: interleaved RGBA canvas_w*canvas_h*4, or empty -> transparent
bool write_psd(const std::string & path, int canvas_w, int canvas_h,
               const std::vector<PsdLayer> & layers,
               const std::vector<uint8_t> & composite);
