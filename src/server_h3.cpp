// HTTP/3 (QUIC) server for the see-through pipeline, using the vendored
// picoquic + picotls + mbedtls stack (see docs/decisions/0008-open-work.md,
// "post-MVP: C ABI + server"). One endpoint:
//
//   POST /render  (body: raw image bytes; JSON response, same shape as the
//                  httplib-based server: {"layers":[{"tag":...,
//                  "png_base64":...}, ...]})
//
//   see-through-server-h3 -m <model-dir> [--port 4433] [--cert p.pem] [--key k.pem]
//
// picoquic owns real sockets here (picoquic_packet_loop, from sockloop.c/
// winsockloop.c) -- unlike a Flow-integrated embedding, this is a
// standalone process with no other event loop to cooperate with.
#include "see_through_capi.h"

#include "picoquic.h"
#include "picoquic_packet_loop.h"
#include "picoquic_utils.h"
#include "h3zero.h"
#include "h3zero_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <unordered_map>

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

std::string render_to_json(const std::string & model_dir, const std::string & image_bytes) {
    st_render_result r;
    int rc = st_render(model_dir.c_str(),
                       reinterpret_cast<const uint8_t *>(image_bytes.data()), image_bytes.size(),
                       30, 768, 768, 42, "auto", &r);
    if (rc != 0) {
        return "{\"error\":\"pipeline failed (code " + std::to_string(rc) + ")\"}";
    }
    std::string json = "{\"layers\":[";
    for (size_t i = 0; i < r.num_layers; i++) {
        if (i) { json += ","; }
        json += "{\"tag\":\"" + json_escape(r.layers[i].tag) + "\",\"png_base64\":\""
              + b64_encode(r.layers[i].png, r.layers[i].png_len) + "\"}";
    }
    json += "]}";
    st_free_result(&r);
    return json;
}

// per-stream request state: accumulated POST body -> rendered JSON response,
// streamed back over possibly-many picohttp_callback_provide_data pulls.
// path_callback has no per-stream storage of its own (path_app_ctx is
// shared across every stream to this path), so this map keyed by stream_ctx
// is ours to own.
struct RenderStreamState {
    std::string post_body;
    std::string response;
    size_t sent = 0;
};

struct RenderPathCtx {
    std::string model_dir;
    std::unordered_map<h3zero_stream_ctx_t *, RenderStreamState *> streams;
};

// h3zero_common.c only ever sets stream_ctx->path_callback_ctx for
// WebTransport paths (webtransport.c/wt_baton.c/picomask.c) -- for a plain
// POST path_table entry like ours, it's left at whatever the stream_ctx was
// initialized to (effectively NULL), unlike the legacy HTTP/0.9 code path
// in demoserver.c, which does pass path_table[i].path_app_ctx through
// correctly. Confirmed by a server crash (SIGSEGV, no trace output even at
// the first line of this function) the moment a POST arrived. Sidestepping
// entirely: this server only ever has one render context, so a global
// avoids depending on a value the framework doesn't reliably pass through.
RenderPathCtx * g_render_ctx = nullptr;

int render_path_callback(picoquic_cnx_t * cnx, uint8_t * bytes, size_t length,
                         picohttp_call_back_event_t event,
                         h3zero_stream_ctx_t * stream_ctx, void * /*path_app_ctx*/) {
    auto * pctx = g_render_ctx;
    switch (event) {
    case picohttp_callback_post: {
        pctx->streams[stream_ctx] = new RenderStreamState();
        return 0;
    }
    case picohttp_callback_post_data: {
        auto it = pctx->streams.find(stream_ctx);
        if (it == pctx->streams.end()) { return -1; }
        it->second->post_body.append(reinterpret_cast<char *>(bytes), length);
        return 0;
    }
    case picohttp_callback_post_fin: {
        auto it = pctx->streams.find(stream_ctx);
        if (it == pctx->streams.end()) { return -1; }
        RenderStreamState * st = it->second;
        picoquic_log_app_message(cnx, "rendering, %zu image bytes received", st->post_body.size());
        st->response = render_to_json(pctx->model_dir, st->post_body);
        // large responses (always true here) are pulled via provide_data,
        // not concatenated into the `bytes` buffer the framework passed us
        return (int) st->response.size();
    }
    case picohttp_callback_provide_data: {
        // NOTE: for this event, `bytes`/`length` are actually the opaque
        // `context`/`space` the framework's h3zero_callback_prepare_to_send
        // passes through unchanged -- see h3zero_common.c's dispatch.
        auto it = pctx->streams.find(stream_ctx);
        if (it == pctx->streams.end()) { return -1; }
        RenderStreamState * st = it->second;
        void * context = bytes;
        size_t space = length;
        size_t remaining = st->response.size() - st->sent;
        size_t n = remaining < space ? remaining : space;
        int is_fin = (st->sent + n) >= st->response.size();
        uint8_t * buf = picoquic_provide_stream_data_buffer(context, n, is_fin, !is_fin);
        if (buf == NULL) { return -1; }
        memcpy(buf, st->response.data() + st->sent, n);
        st->sent += n;
        if (is_fin) {
            delete st;
            pctx->streams.erase(it);
        }
        return 0;
    }
    default:
        return 0;
    }
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_dir = "models";
    int port = 4433;
    std::string cert_path = "third_party/picoquic/certs/secp256r1/cert.pem";
    std::string key_path = "third_party/picoquic/certs/secp256r1/key.pem";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "-m") { model_dir = next(); }
        else if (a == "--port") { port = std::atoi(next().c_str()); }
        else if (a == "--cert") { cert_path = next(); }
        else if (a == "--key") { key_path = next(); }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    picohttp_server_path_item_t path_table[1];
    RenderPathCtx render_ctx;
    render_ctx.model_dir = model_dir;
    g_render_ctx = &render_ctx;
    path_table[0].path = "/render";
    path_table[0].path_length = strlen("/render");
    path_table[0].path_callback = render_path_callback;
    path_table[0].path_app_ctx = &render_ctx;

    picohttp_server_parameters_t server_params;
    memset(&server_params, 0, sizeof(server_params));
    server_params.web_folder = NULL;
    server_params.path_table = path_table;
    server_params.path_table_nb = 1;

    // h3zero_callback (picohttp/h3zero_common.c) checks, on a connection's
    // FIRST callback invocation, whether callback_ctx == the default ctx
    // registered at picoquic_create time -- if so, it treats that pointer
    // as a picohttp_server_parameters_t* and builds a fresh, per-connection
    // h3zero_callback_ctx_t from it via h3zero_callback_create_context().
    // Passing an ALREADY-BUILT h3zero_callback_ctx_t* here (as an earlier
    // version of this file did) is a type-confusion bug: the framework
    // reinterprets that memory as a picohttp_server_parameters_t, reading
    // garbage path_table/path_table_nb -- harmless for GET (fails closed
    // to 404-ish), but an out-of-bounds path_table[] access (and crash) for
    // POST, which is exactly the SIGSEGV this was causing. Fixed by passing
    // &server_params directly, matching picoquicdemo.c's own quic_server().
    uint8_t reset_seed[PICOQUIC_RESET_SECRET_SIZE];
    memset(reset_seed, 0x55, sizeof(reset_seed));
    picoquic_quic_t * quic = picoquic_create(64,
        cert_path.c_str(), key_path.c_str(), NULL, "h3",
        h3zero_callback, &server_params, NULL, NULL,
        reset_seed, picoquic_current_time(), NULL, NULL, NULL, 0);
    if (quic == NULL) {
        fprintf(stderr, "picoquic_create failed (check cert/key paths: %s / %s)\n",
                cert_path.c_str(), key_path.c_str());
        return 1;
    }

    printf("see-through-server-h3: model_dir=%s, listening on udp/%d (h3, POST /render)\n",
           model_dir.c_str(), port);
    int ret = picoquic_packet_loop(quic, port, 0, 0, 0, 0, NULL, NULL);
    picoquic_free(quic);
    return ret;
}
