// Upstream-parity reader: extracts every layer from a PSD (name, bbox,
// RGBA expanded to full canvas) as PNGs, for comparing against our own
// --png-dir output by tag name (docs/decisions/0008-open-work.md's
// upstream-parity item). Read-only; no PSD writing anywhere in this repo.
//   psd_layers <file.psd> <out_dir>
#include "Psd/Psd.h"
#include "Psd/PsdPlatform.h"
#include "Psd/PsdMallocAllocator.h"
#include "Psd/PsdNativeFile.h"
#include "Psd/PsdDocument.h"
#include "Psd/PsdColorMode.h"
#include "Psd/PsdLayer.h"
#include "Psd/PsdChannel.h"
#include "Psd/PsdChannelType.h"
#include "Psd/PsdLayerMaskSection.h"
#include "Psd/PsdParseDocument.h"
#include "Psd/PsdParseLayerMaskSection.h"
#include "Psd/PsdLayerCanvasCopy.h"

#include "image_utils.h"

#include <cstdio>
#include <filesystem>
#include <string>

PSD_USING_NAMESPACE;

namespace {

unsigned int find_channel(Layer * layer, int16_t type) {
    for (unsigned int i = 0; i < layer->channelCount; i++) {
        if (layer->channels[i].data && layer->channels[i].type == type) { return i; }
    }
    return UINT_MAX;
}

// expands one 8-bit channel's layer-local data to full canvas size, host malloc
uint8_t * expand8(Allocator & alloc, Document * doc, Layer * layer, Channel * ch) {
    auto * canvas = static_cast<uint8_t *>(alloc.Allocate((size_t) doc->width * doc->height, 16u));
    memset(canvas, 0, (size_t) doc->width * doc->height);
    imageUtil::CopyLayerData(static_cast<const uint8_t *>(ch->data), canvas,
                             layer->left, layer->top, layer->right, layer->bottom,
                             doc->width, doc->height);
    return canvas;
}

std::string safe_name(const std::string & s) {
    std::string r = s;
    for (char & c : r) {
        if (!isalnum((unsigned char) c) && c != '-' && c != '_') { c = '_'; }
    }
    return r;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <file.psd> <out_dir>\n", argv[0]); return 1; }
    std::filesystem::create_directories(argv[2]);

    MallocAllocator allocator;
    NativeFile file(&allocator);
    std::wstring wpath(argv[1], argv[1] + strlen(argv[1]));
    if (!file.OpenRead(wpath.c_str())) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

    Document * document = CreateDocument(&file, &allocator);
    if (!document) { fprintf(stderr, "cannot parse PSD document\n"); file.Close(); return 1; }
    if (document->colorMode != colorMode::RGB) {
        fprintf(stderr, "not RGB color mode (mode=%d), unsupported\n", (int) document->colorMode);
        DestroyDocument(document, &allocator);
        file.Close();
        return 1;
    }
    if (document->bitsPerChannel != 8) {
        fprintf(stderr, "only 8-bit/channel supported (got %d)\n", document->bitsPerChannel);
        DestroyDocument(document, &allocator);
        file.Close();
        return 1;
    }

    printf("document: %ux%u, %u bits/channel\n", document->width, document->height, document->bitsPerChannel);

    LayerMaskSection * lms = ParseLayerMaskSection(document, &file, &allocator);
    if (!lms) { fprintf(stderr, "no layer/mask section\n"); DestroyDocument(document, &allocator); file.Close(); return 1; }

    for (unsigned int i = 0; i < lms->layerCount; i++) {
        Layer * layer = &lms->layers[i];
        ExtractLayer(document, &file, &allocator, layer);

        unsigned int ir = find_channel(layer, channelType::R);
        unsigned int ig = find_channel(layer, channelType::G);
        unsigned int ib = find_channel(layer, channelType::B);
        unsigned int ia = find_channel(layer, channelType::TRANSPARENCY_MASK);
        printf("layer %2u: \"%s\" bbox=[%d,%d,%d,%d] rgb=%d a=%d\n", i, layer->name.c_str(),
               layer->left, layer->top, layer->right, layer->bottom,
               ir != UINT_MAX && ig != UINT_MAX && ib != UINT_MAX, ia != UINT_MAX);

        Image img;
        img.w = document->width; img.h = document->height; img.c = 4;
        img.data.assign((size_t) img.w * img.h * 4, 0.0f);

        uint8_t * cr = ir != UINT_MAX ? expand8(allocator, document, layer, &layer->channels[ir]) : nullptr;
        uint8_t * cg = ig != UINT_MAX ? expand8(allocator, document, layer, &layer->channels[ig]) : nullptr;
        uint8_t * cb = ib != UINT_MAX ? expand8(allocator, document, layer, &layer->channels[ib]) : nullptr;
        uint8_t * ca = ia != UINT_MAX ? expand8(allocator, document, layer, &layer->channels[ia]) : nullptr;

        for (size_t p = 0; p < (size_t) img.w * img.h; p++) {
            img.data[p * 4 + 0] = cr ? cr[p] / 255.0f : 0.0f;
            img.data[p * 4 + 1] = cg ? cg[p] / 255.0f : 0.0f;
            img.data[p * 4 + 2] = cb ? cb[p] / 255.0f : 0.0f;
            img.data[p * 4 + 3] = ca ? ca[p] / 255.0f : 1.0f;
        }
        for (uint8_t * p : { cr, cg, cb, ca }) {
            if (p) { allocator.Free(p); }
        }

        char prefix[16];
        snprintf(prefix, sizeof(prefix), "%03u_", i);
        std::string path = std::string(argv[2]) + "/" + prefix + safe_name(layer->name.c_str()) + ".png";
        if (!save_image(path, img)) { fprintf(stderr, "failed to write %s\n", path.c_str()); }
    }

    unsigned int layerCount = lms->layerCount;
    DestroyLayerMaskSection(lms, &allocator);
    DestroyDocument(document, &allocator);
    file.Close();
    printf("wrote %u layers to %s\n", layerCount, argv[2]);
    return 0;
}
