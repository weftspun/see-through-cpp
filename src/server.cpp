// Minimal HTTP server wrapping the see-through pipeline (MADR 0008's
// post-MVP C ABI + server item, trellis2cpp-style). One endpoint:
//
//   POST /render?steps=8&res=1280&depth_res=768&seed=42&device=auto
//   body: raw image bytes (PNG/JPEG/whatever stb_image decodes)
//   response: multipart-free JSON { "svg": "<...>", "layers": [{"tag":...,
//             "png_base64":...}, ...] } -- z-ordered back-to-front.
//
//   see-through-server -m <model-dir> [--port 8080]
#include "see_through_capi.h"

#define CPPHTTPLIB_NO_EXCEPTIONS
#include "httplib.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

std::string b64_encode(const uint8_t * d, size_t n) {
    static const char * T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o;
    o.reserve((n + 2) / 3 * 4);
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = d[i] << 16 | (i + 1 < n ? d[i + 1] << 8 : 0) | (i + 2 < n ? d[i + 2] : 0);
        o += T[v >> 18]; o += T[(v >> 12) & 63];
        o += i + 1 < n ? T[(v >> 6) & 63] : '=';
        o += i + 2 < n ? T[v & 63] : '=';
    }
    return o;
}

std::string json_escape(const std::string & s) {
    std::string o;
    o.reserve(s.size());
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') { o += "\\n"; }
        else { o += c; }
    }
    return o;
}

int get_int(const httplib::Request & req, const char * name, int def) {
    if (!req.has_param(name)) { return def; }
    return std::atoi(req.get_param_value(name).c_str());
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_dir = "models";
    int port = 8080;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "-m") { model_dir = next(); }
        else if (a == "--port") { port = std::atoi(next().c_str()); }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    httplib::Server svr;
    svr.Post("/render", [&](const httplib::Request & req, httplib::Response & res) {
        if (req.body.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"empty request body (expected raw image bytes)\"}", "application/json");
            return;
        }
        st_render_result r;
        int rc = st_render(model_dir.c_str(),
                           reinterpret_cast<const uint8_t *>(req.body.data()), req.body.size(),
                           get_int(req, "steps", 30), get_int(req, "res", 1280),
                           get_int(req, "depth_res", 768),
                           (uint64_t) get_int(req, "seed", 42),
                           req.has_param("device") ? req.get_param_value("device").c_str() : "auto",
                           &r);
        if (rc != 0) {
            res.status = 500;
            res.set_content("{\"error\":\"pipeline failed (code " + std::to_string(rc) + ")\"}", "application/json");
            return;
        }
        std::string json = "{\"svg\":\"" + json_escape(std::string(r.svg, r.svg_len)) + "\",\"layers\":[";
        for (size_t i = 0; i < r.num_layers; i++) {
            if (i) { json += ","; }
            json += "{\"tag\":\"" + json_escape(r.layers[i].tag) + "\",\"png_base64\":\""
                  + b64_encode(r.layers[i].png, r.layers[i].png_len) + "\"}";
        }
        json += "]}";
        st_free_result(&r);
        res.set_content(json, "application/json");
    });

    printf("see-through-server: model_dir=%s, listening on :%d\n", model_dir.c_str(), port);
    svr.listen("0.0.0.0", port);
    return 0;
}
