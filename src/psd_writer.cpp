#include "psd_writer.h"

#include <cstdio>
#include <cstring>

// big-endian byte sink
struct BeBuf {
    std::vector<uint8_t> d;
    void u8(uint8_t v) { d.push_back(v); }
    void u16(uint16_t v) { d.push_back(v >> 8); d.push_back(v & 0xff); }
    void u32(uint32_t v) { for (int s = 24; s >= 0; s -= 8) d.push_back((v >> s) & 0xff); }
    void i16(int16_t v) { u16((uint16_t) v); }
    void i32(int32_t v) { u32((uint32_t) v); }
    void raw(const void * p, size_t n) {
        d.insert(d.end(), (const uint8_t *) p, (const uint8_t *) p + n);
    }
    void tag(const char * t) { raw(t, 4); }
};

// PackBits one row
void packbits(const uint8_t * row, int n, std::vector<uint8_t> & out) {
    // fold only runs >= 3 into RLE — breaking a literal for a 2-run costs
    // more than it saves and breaks the n + ceil(n/128) expansion bound
    int i = 0;
    while (i < n) {
        int run = 1;
        while (i + run < n && run < 128 && row[i + run] == row[i]) run++;
        if (run >= 3) {
            out.push_back((uint8_t) (int8_t) (1 - run));
            out.push_back(row[i]);
            i += run;
        } else {
            int lit = 0;
            while (i + lit < n && lit < 128) {
                int r = 1;
                while (i + lit + r < n && r < 3 && row[i + lit + r] == row[i + lit]) r++;
                if (r >= 3) break;
                lit++;
            }
            out.push_back((uint8_t) (lit - 1));
            out.insert(out.end(), row + i, row + i + lit);
            i += lit;
        }
    }
}

// one channel of a layer -> compression header + row table + streams
static void channel_rle(const uint8_t * rgba, int w, int h, int comp,
                        std::vector<uint8_t> & out) {
    BeBuf b;
    b.u16(1);                                     // RLE
    std::vector<std::vector<uint8_t>> rows(h);
    std::vector<uint8_t> plane(w);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) plane[x] = rgba[(y * (size_t) w + x) * 4 + comp];
        packbits(plane.data(), w, rows[y]);
        b.u16((uint16_t) rows[y].size());
    }
    for (auto & r : rows) b.raw(r.data(), r.size());
    out = std::move(b.d);
}

bool write_psd(const std::string & path, int W, int H,
               const std::vector<PsdLayer> & layers,
               const std::vector<uint8_t> & composite) {
    static const int comp_of_chan[4] = { 0, 1, 2, 3 };   // R,G,B,A
    static const int16_t chan_ids[4] = { 0, 1, 2, -1 };

    BeBuf f;
    f.tag("8BPS"); f.u16(1);
    for (int i = 0; i < 6; i++) f.u8(0);
    f.u16(4); f.u32(H); f.u32(W); f.u16(8); f.u16(3);    // RGBA, 8-bit, RGB mode
    f.u32(0);                                            // color mode data
    f.u32(0);                                            // image resources

    // layer records + channel data
    BeBuf li;
    li.i16((int16_t) layers.size());
    std::vector<std::vector<uint8_t>> chan_data;
    for (const PsdLayer & l : layers) {
        li.i32(l.top); li.i32(l.left); li.i32(l.top + l.h); li.i32(l.left + l.w);
        li.u16(4);
        for (int c = 0; c < 4; c++) {
            std::vector<uint8_t> cd;
            channel_rle(l.rgba.data(), l.w, l.h, comp_of_chan[c], cd);
            li.i16(chan_ids[c]);
            li.u32((uint32_t) cd.size());
            chan_data.push_back(std::move(cd));
        }
        li.tag("8BIM"); li.tag("norm");
        li.u8(255); li.u8(0); li.u8(0); li.u8(0);        // opacity, clip, flags, filler

        BeBuf ex;
        ex.u32(0);                                       // mask
        ex.u32(0);                                       // blending ranges
        size_t nlen = l.name.size() > 255 ? 255 : l.name.size();
        ex.u8((uint8_t) nlen);
        ex.raw(l.name.data(), nlen);
        while ((nlen + 1) % 4) { ex.u8(0); nlen++; }     // pascal pad to 4
        BeBuf uni;                                       // 'luni' unicode name
        uni.u32((uint32_t) l.name.size());
        for (char c : l.name) uni.u16((uint8_t) c);      // ascii tags -> utf16be
        ex.tag("8BIM"); ex.tag("luni");
        ex.u32((uint32_t) uni.d.size());
        ex.raw(uni.d.data(), uni.d.size());
        if (uni.d.size() % 2) ex.u8(0);

        li.u32((uint32_t) ex.d.size());
        li.raw(ex.d.data(), ex.d.size());
    }
    for (auto & cd : chan_data) li.raw(cd.data(), cd.size());
    if (li.d.size() % 2) li.u8(0);

    BeBuf lm;                                            // layer & mask section
    lm.u32((uint32_t) li.d.size());
    lm.raw(li.d.data(), li.d.size());
    lm.u32(0);                                           // global layer mask
    f.u32((uint32_t) lm.d.size());
    f.raw(lm.d.data(), lm.d.size());

    // merged composite: raw planar RGBA
    f.u16(0);
    std::vector<uint8_t> zero;
    const uint8_t * comp = composite.data();
    if (composite.empty()) { zero.assign((size_t) W * H * 4, 0); comp = zero.data(); }
    std::vector<uint8_t> plane((size_t) W * H);
    for (int c = 0; c < 4; c++) {
        for (size_t i = 0; i < (size_t) W * H; i++) plane[i] = comp[i * 4 + c];
        f.raw(plane.data(), plane.size());
    }

    FILE * fp = fopen(path.c_str(), "wb");
    if (!fp) return false;
    size_t n = fwrite(f.d.data(), 1, f.d.size(), fp);
    fclose(fp);
    return n == f.d.size();
}
