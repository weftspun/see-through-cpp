#include "psd_write.h"

#include "Psd/Psd.h"
#include "Psd/PsdPlatform.h"
#include "Psd/PsdMallocAllocator.h"
#include "Psd/PsdNativeFile.h"
#include "Psd/PsdExport.h"
#include "Psd/PsdExportDocument.h"
#include "Psd/PsdExportColorMode.h"
#include "Psd/PsdExportChannel.h"
#include "Psd/PsdCompressionType.h"

#include "image_utils.h"

PSD_USING_NAMESPACE;

namespace {

// planar uint8 for one channel of img's [x0,y0)-[x1,y1) crop; img.c-strided
// float [0,1] (4 for RGBA, 1 for grayscale), ch in {0=R,1=G,2=B,3=A} or {0=GRAY}
std::vector<uint8_t> planar8(const Image & img, int x0, int y0, int x1, int y1, int ch) {
    std::vector<uint8_t> out((size_t) (x1 - x0) * (y1 - y0));
    size_t o = 0;
    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            float v = img.data[((size_t) y * img.w + x) * img.c + ch];
            out[o++] = (uint8_t) std::lround(std::min(1.0f, std::max(0.0f, v)) * 255.0f);
        }
    }
    return out;
}

} // namespace

// Note: layers here carry transparency only via the ALPHA channel
// (channelType::TRANSPARENCY_MASK, id -1), not a real Photoshop layer mask
// (id -2, shown as a separate mask thumbnail in the Layers panel). Upstream
// PSD output does write a real mask per layer, but the vendored Psd_SDK's
// export path has no support for it: ExportLayer::MAX_CHANNEL_COUNT is
// hardcoded to 4 (third_party/psd_sdk/src/Psd/PsdExportLayer.h) and
// GetExtraDataLength() always emits a zero-length mask-data block
// (third_party/psd_sdk/src/Psd/PsdExport.cpp) -- there's no channel ID -2
// case anywhere in the writer. Compositing is unaffected (alpha already
// encodes the same transparency); the only loss is Photoshop's separate
// mask-editing handle. Adding real mask support would mean patching the
// vendored writer to carry forward across future vendor updates -- not
// done here.
bool write_psd(const std::string & path, int canvas_w, int canvas_h,
               const std::vector<std::pair<std::string, std::vector<uint8_t>>> & layers,
               const std::vector<std::array<int, 4>> & xyxy) {
    if (layers.size() != xyxy.size()) { return false; }

    MallocAllocator allocator;
    NativeFile file(&allocator);
    std::wstring wpath(path.begin(), path.end());
    if (!file.OpenWrite(wpath.c_str())) { return false; }

    ExportDocument * document = CreateExportDocument(&allocator, (unsigned) canvas_w, (unsigned) canvas_h,
                                                      8u, exportColorMode::RGB);

    for (size_t i = 0; i < layers.size(); i++) {
        const std::string & name = layers[i].first;
        const std::vector<uint8_t> & png = layers[i].second;
        const std::array<int, 4> & bb = xyxy[i];
        int x0 = bb[0], y0 = bb[1], x1 = bb[2], y1 = bb[3];
        if (x1 <= x0 || y1 <= y0) { continue; }

        Image img;
        if (!load_image_from_memory(png.data(), png.size(), img, 4)) { continue; }

        unsigned int layer = AddLayer(document, &allocator, name.c_str());
        UpdateLayer(document, &allocator, layer, exportChannel::RED,   x0, y0, x1, y1, planar8(img, 0, 0, img.w, img.h, 0).data(), compressionType::RLE);
        UpdateLayer(document, &allocator, layer, exportChannel::GREEN, x0, y0, x1, y1, planar8(img, 0, 0, img.w, img.h, 1).data(), compressionType::RLE);
        UpdateLayer(document, &allocator, layer, exportChannel::BLUE,  x0, y0, x1, y1, planar8(img, 0, 0, img.w, img.h, 2).data(), compressionType::RLE);
        UpdateLayer(document, &allocator, layer, exportChannel::ALPHA, x0, y0, x1, y1, planar8(img, 0, 0, img.w, img.h, 3).data(), compressionType::RLE);
    }

    WriteDocument(document, &allocator, &file);
    DestroyExportDocument(document, &allocator);
    file.Close();
    return true;
}

bool write_psd_gray(const std::string & path, int canvas_w, int canvas_h,
                    const std::vector<std::pair<std::string, std::vector<uint8_t>>> & layers,
                    const std::vector<std::array<int, 4>> & xyxy) {
    if (layers.size() != xyxy.size()) { return false; }

    MallocAllocator allocator;
    NativeFile file(&allocator);
    std::wstring wpath(path.begin(), path.end());
    if (!file.OpenWrite(wpath.c_str())) { return false; }

    ExportDocument * document = CreateExportDocument(&allocator, (unsigned) canvas_w, (unsigned) canvas_h,
                                                      8u, exportColorMode::GRAYSCALE);

    for (size_t i = 0; i < layers.size(); i++) {
        const std::string & name = layers[i].first;
        const std::vector<uint8_t> & png = layers[i].second;
        const std::array<int, 4> & bb = xyxy[i];
        int x0 = bb[0], y0 = bb[1], x1 = bb[2], y1 = bb[3];
        if (x1 <= x0 || y1 <= y0) { continue; }

        Image img;
        if (!load_image_from_memory(png.data(), png.size(), img, 1)) { continue; }

        unsigned int layer = AddLayer(document, &allocator, name.c_str());
        UpdateLayer(document, &allocator, layer, exportChannel::GRAY, x0, y0, x1, y1,
                    planar8(img, 0, 0, img.w, img.h, 0).data(), compressionType::RLE);
    }

    WriteDocument(document, &allocator, &file);
    DestroyExportDocument(document, &allocator);
    file.Close();
    return true;
}
