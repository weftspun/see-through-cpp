// Upstream-parity writer: dumps our own per-layer PNGs into a flat (no
// groups), 8-bit RGB PSD -- same shape as upstream's save_psd() (io_utils.py):
// one pixel layer per tag, positioned at its own tight bbox, full opacity,
// document order = z order back-to-front. Counterpart to tests/psd_layers.cpp
// (that one reads; this one writes).
#pragma once

#include "pipeline.h"

// layers/xyxy must be same length, same order (back to front); layers[i] is
// {name, png_bytes} and xyxy[i] is {x0,y0,x1,y1} in canvas_w x canvas_h coords
// -- exactly SeeThroughResult::png_layers / layer_xyxy.
bool write_psd(const std::string & path, int canvas_w, int canvas_h,
               const std::vector<std::pair<std::string, std::vector<uint8_t>>> & layers,
               const std::vector<std::array<int, 4>> & xyxy);

// same shape, single-channel (GRAY, no alpha) PSD -- upstream's dump_parts_psd
// companion "<name>_depth.psd" (SeeThroughResult::depth_layers / layer_xyxy).
bool write_psd_gray(const std::string & path, int canvas_w, int canvas_h,
                    const std::vector<std::pair<std::string, std::vector<uint8_t>>> & layers,
                    const std::vector<std::array<int, 4>> & xyxy);
