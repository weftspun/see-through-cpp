// Property-based tests (rapidcheck) for the pure-logic components: PackBits,
// the CLIP tokenizer, both schedulers, resize/pad, and the postproc
// primitives. The numeric reference gates (test_clip, test_unet_*, ...)
// remain characterization tests against PyTorch by design.
//
//   test_properties [layerdiff-te1.gguf]   (tokenizer properties need a gguf)

#include "image_utils.h"
#include "clip.h"
#include "postproc.h"
#include "psd_writer.h"
#include "scheduler.h"

#include <rapidcheck.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

static int failures = 0;

template <typename Fn>
static void prop(const char * name, Fn fn) {
    printf("-- %s --\n", name);
    if (!rc::check(name, fn)) failures++;
}

// PackBits decoder (trivial, test-only)
static std::vector<uint8_t> unpackbits(const std::vector<uint8_t> & in, int n) {
    std::vector<uint8_t> out;
    size_t i = 0;
    while (i < in.size() && (int) out.size() < n) {
        int8_t h = (int8_t) in[i++];
        if (h >= 0) {
            for (int k = 0; k <= h; k++) out.push_back(in[i++]);
        } else if (h != -128) {
            uint8_t v = in[i++];
            for (int k = 0; k < 1 - h; k++) out.push_back(v);
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    prop("packbits round-trips arbitrary rows", [](const std::vector<uint8_t> & row) {
        RC_PRE(!row.empty());
        std::vector<uint8_t> enc;
        packbits(row.data(), (int) row.size(), enc);
        RC_ASSERT(unpackbits(enc, (int) row.size()) == row);
        // worst-case expansion bound: 1 header per 128 literals
        RC_ASSERT(enc.size() <= row.size() + (row.size() + 127) / 128);
    });

    prop("dpm timesteps strictly descending in [1,1000)", []() {
        const int n = *rc::gen::inRange(1, 60);
        DpmSolverSDE s;
        s.set_timesteps(n);
        RC_ASSERT((int) s.timesteps.size() == n);
        for (size_t i = 0; i < s.timesteps.size(); i++) {
            RC_ASSERT(s.timesteps[i] >= 1);
            RC_ASSERT(s.timesteps[i] < 1000);
            if (i) RC_ASSERT(s.timesteps[i] < s.timesteps[i - 1]);
        }
        for (size_t i = 0; i + 1 < s.sigmas.size(); i++) {
            RC_ASSERT(s.sigmas[i] > s.sigmas[i + 1]);
        }
        RC_ASSERT(s.sigmas.back() == 0.0);
    });

    prop("dpm step keeps finite samples finite", []() {
        const int n = *rc::gen::inRange(2, 20);
        DpmSolverSDE s;
        s.set_timesteps(n);
        std::vector<float> x = *rc::gen::container<std::vector<float>>(
            8, rc::gen::map(rc::gen::inRange(-1000, 1000), [](int v) { return v / 100.0f; }));
        std::vector<float> eps = x, noise(8, 0.0f);
        for (int k = 0; k < n; k++) {
            s.step(x, eps, noise);
            for (float v : x) RC_ASSERT(std::isfinite(v));
        }
    });

    prop("ddim timesteps descending, trailing ends at 999", []() {
        const int n = *rc::gen::inRange(1, 50);
        DdimTrailing s;
        s.set_timesteps(n);
        RC_ASSERT(s.timesteps.front() == 999);
        for (size_t i = 0; i < s.timesteps.size(); i++) {
            RC_ASSERT(s.timesteps[i] >= 0);
            if (i) RC_ASSERT(s.timesteps[i] < s.timesteps[i - 1]);
        }
    });

    prop("resize preserves constant images", []() {
        const int w = *rc::gen::inRange(1, 40), h = *rc::gen::inRange(1, 40);
        const int tw = *rc::gen::inRange(1, 40), th = *rc::gen::inRange(1, 40);
        const float v = (float) (*rc::gen::inRange(0, 255)) / 255.0f;
        Image img;
        img.w = w; img.h = h; img.c = 3;
        img.data.assign((size_t) w * h * 3, v);
        Image out = smart_resize(img, tw, th);
        RC_ASSERT(out.w == tw);
        RC_ASSERT(out.h == th);
        for (float o : out.data) RC_ASSERT(std::fabs(o - v) < 1e-5f);
    });

    prop("resize output stays within input range", []() {
        const int w = *rc::gen::inRange(2, 24), h = *rc::gen::inRange(2, 24);
        const int tw = *rc::gen::inRange(1, 48), th = *rc::gen::inRange(1, 48);
        Image img;
        img.w = w; img.h = h; img.c = 1;
        img.data = *rc::gen::container<std::vector<float>>(
            (size_t) w * h, rc::gen::map(rc::gen::inRange(0, 1000), [](int v) { return v / 1000.0f; }));
        float lo = *std::min_element(img.data.begin(), img.data.end());
        float hi = *std::max_element(img.data.begin(), img.data.end());
        Image out = smart_resize(img, tw, th);
        for (float o : out.data) {
            RC_ASSERT(o >= lo - 1e-5f);
            RC_ASSERT(o <= hi + 1e-5f);
        }
    });

    prop("center_square_pad_resize returns the requested square", []() {
        const int w = *rc::gen::inRange(1, 64), h = *rc::gen::inRange(1, 64);
        const int target = *rc::gen::inRange(1, 64);
        Image img;
        img.w = w; img.h = h; img.c = 4;
        img.data.assign((size_t) w * h * 4, 0.5f);
        int pw, ph, px, py;
        Image out = center_square_pad_resize(img, target, 0.0f, &pw, &ph, &px, &py);
        RC_ASSERT(out.w == target);
        RC_ASSERT(out.h == target);
        RC_ASSERT(pw == ph);
        RC_ASSERT(pw == std::max(w, h));
        RC_ASSERT(px == (pw - w) / 2);
        RC_ASSERT(py == (ph - h) / 2);
    });

    prop("connected components partition the mask", []() {
        const int w = *rc::gen::inRange(1, 24), h = *rc::gen::inRange(1, 24);
        std::vector<uint8_t> mask = *rc::gen::container<std::vector<uint8_t>>(
            (size_t) w * h, rc::gen::map(rc::gen::inRange(0, 2), [](int v) { return (uint8_t) (v * 255); }));
        std::vector<int> labels;
        std::vector<CCStats> stats;
        int n = connected_components(mask, w, h, labels, stats);
        RC_ASSERT(n >= 1);
        size_t on = 0;
        for (size_t i = 0; i < mask.size(); i++) {
            if (mask[i]) {
                RC_ASSERT(labels[i] >= 1);
                RC_ASSERT(labels[i] < n);
                on++;
            } else {
                RC_ASSERT(labels[i] == 0);
            }
        }
        size_t area_sum = 0;
        for (int l = 1; l < n; l++) area_sum += stats[l].area;
        RC_ASSERT(area_sum == on);
    });

    prop("kmeans2 split lies within sample range", []() {
        std::vector<float> v = *rc::gen::container<std::vector<float>>(
            rc::gen::map(rc::gen::inRange(0, 1000), [](int x) { return x / 1000.0f; }));
        RC_PRE(v.size() >= 2);
        double thr = kmeans2_split(v);
        RC_ASSERT(thr >= *std::min_element(v.begin(), v.end()) - 1e-9);
        RC_ASSERT(thr <= *std::max_element(v.begin(), v.end()) + 1e-9);
    });

    if (argc > 1) {
        Model m;
        if (!m.load(argv[1])) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
        std::string vocab, merges;
        for (const auto & kv : m.config_json) {
            if (kv.first.find("vocab_json") != std::string::npos) vocab = kv.second;
            if (kv.first.find("merges_txt") != std::string::npos) merges = kv.second;
        }
        static ClipTokenizer tok;
        if (!tok.load(vocab, merges)) { fprintf(stderr, "tokenizer load failed\n"); return 1; }

        prop("tokenizer pads to exactly n_ctx with BOS/EOS invariants", []() {
            std::string s = *rc::gen::container<std::string>(
                rc::gen::map(rc::gen::inRange(32, 127), [](int c) { return (char) c; }));
            int eos = -1;
            std::vector<int32_t> ids = tok.encode_padded(s, 77, tok.eos_id, &eos);
            RC_ASSERT((int) ids.size() == 77);
            RC_ASSERT(ids[0] == tok.bos_id);
            RC_ASSERT(eos >= 1);
            RC_ASSERT(eos < 77);
            RC_ASSERT(ids[eos] == tok.eos_id);
            for (int32_t id : ids) {
                RC_ASSERT(id >= 0);
                RC_ASSERT(id < 49408);
            }
        });
    } else {
        printf("(no gguf given -- tokenizer properties skipped)\n");
    }

    printf(failures ? "PROPERTIES FAIL (%d)\n" : "PROPERTIES PASS\n", failures);
    return failures ? 1 : 0;
}
