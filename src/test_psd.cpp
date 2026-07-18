// M11 of see-through.cpp: PSD writer round-trip. Writes deterministic layers
// + a raw dump of the same pixels; check_psd.py (psd-tools) verifies names,
// offsets and exact pixel equality.
//
//   test_psd <out.psd> <out_raw.bin>

#include "psd_writer.h"

#include <cstdio>

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s out.psd out_raw.bin\n", argv[0]); return 1; }

    const int W = 96, H = 80;
    std::vector<PsdLayer> layers(2);
    layers[0] = { "front hair", 4, 6, 40, 32 };
    layers[1] = { "topwear", 20, 10, 64, 48 };
    for (PsdLayer & l : layers) {
        l.rgba.resize((size_t) l.w * l.h * 4);
        for (int y = 0; y < l.h; y++) {
            for (int x = 0; x < l.w; x++) {
                size_t i = ((size_t) y * l.w + x) * 4;
                l.rgba[i + 0] = (uint8_t) (x * 5 + 1);
                l.rgba[i + 1] = (uint8_t) (y * 3 + 7);
                l.rgba[i + 2] = (uint8_t) ((x + y) * 2);
                l.rgba[i + 3] = (uint8_t) (x > l.w / 2 ? 255 : x * 4);
            }
        }
    }
    std::vector<uint8_t> comp((size_t) W * H * 4, 0);
    for (size_t i = 0; i < comp.size(); i += 4) comp[i + 3] = 255;

    if (!write_psd(argv[1], W, H, layers, comp)) { fprintf(stderr, "write failed\n"); return 1; }

    FILE * f = fopen(argv[2], "wb");
    for (const PsdLayer & l : layers) {
        int32_t hdr[5] = { (int32_t) l.name.size(), l.top, l.left, l.w, l.h };
        fwrite(hdr, 4, 5, f);
        fwrite(l.name.data(), 1, l.name.size(), f);
        fwrite(l.rgba.data(), 1, l.rgba.size(), f);
    }
    fclose(f);
    printf("wrote %s + raw dump\n", argv[1]);
    return 0;
}
