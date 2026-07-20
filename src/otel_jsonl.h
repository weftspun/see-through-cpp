// Minimal, header-only JSONL (newline-delimited JSON) writer for the
// profiling data lake: OpenTelemetry-style spans (trace_id/span_id/
// parent_span_id/name/start+end_time_unix_nano/attributes/status), matching
// the schema used by v-sekai-multiplayer-fabric/fabric-godot-core's
// open_telemetry module (modules/open_telemetry/structures/otel_span.h):
// trace_id = 32 hex chars (128-bit), span_id = 16 hex chars (64-bit),
// randomly generated (not sequential integers), per OTel's own ID
// generation convention.
//
// A "data lake" here means one append-only file, profiling/spans.jsonl --
// one JSON object per line, one line per span. Any JSONL/line-delimited-JSON
// reader (DuckDB `read_json_auto`, Polars `scan_ndjson`, `jq`, plain
// line-by-line parsing) can query it directly; appending a run's spans is
// just an fopen(..., "a") + write, no read-modify-write needed.
#pragma once

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// OTel-style random hex IDs (trace_id: 128-bit/32 hex, span_id: 64-bit/16 hex)
// ---------------------------------------------------------------------------

inline std::string generate_hex_id(int nibbles) {
    static thread_local std::mt19937_64 gen{ std::random_device{}() };
    std::uniform_int_distribution<uint64_t> dist;
    std::string s(nibbles, '0');
    static const char * hex = "0123456789abcdef";
    for (int i = 0; i < nibbles; i += 16) {
        uint64_t v = dist(gen);
        for (int j = 0; j < 16 && i + j < nibbles; j++) {
            s[i + j] = hex[(v >> (60 - 4 * j)) & 0xF];
        }
    }
    return s;
}
inline std::string generate_trace_id() { return generate_hex_id(32); }
inline std::string generate_span_id() { return generate_hex_id(16); }

// ---------------------------------------------------------------------------
// OTel-style span schema (mirrors otel_span.h's core fields)
// ---------------------------------------------------------------------------

struct Span {
    std::string trace_id;         // 32 hex chars, shared by every span of one run
    std::string span_id;          // 16 hex chars, unique per span
    std::string parent_span_id;   // empty for the root span
    std::string name;
    uint64_t start_time_unix_nano = 0;
    uint64_t end_time_unix_nano = 0;
    int64_t status_code = 1;      // OTelSpan::STATUS_CODE_OK (2 = ERROR)
    std::vector<std::pair<std::string, std::string>> attributes;
};

inline std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char) c < 0x20) { char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", c); out += buf; }
                else { out += c; }
        }
    }
    return out;
}

inline std::string span_to_json(const Span & s) {
    std::string j = "{";
    j += "\"trace_id\":\"" + s.trace_id + "\",";
    j += "\"span_id\":\"" + s.span_id + "\",";
    j += "\"parent_span_id\":\"" + s.parent_span_id + "\",";
    j += "\"name\":\"" + json_escape(s.name) + "\",";
    j += "\"start_time_unix_nano\":" + std::to_string(s.start_time_unix_nano) + ",";
    j += "\"end_time_unix_nano\":" + std::to_string(s.end_time_unix_nano) + ",";
    j += "\"duration_ms\":" + std::to_string((s.end_time_unix_nano - s.start_time_unix_nano) / 1e6) + ",";
    j += "\"status_code\":" + std::to_string(s.status_code) + ",";
    j += "\"attributes\":{";
    for (size_t i = 0; i < s.attributes.size(); i++) {
        if (i) j += ",";
        j += "\"" + json_escape(s.attributes[i].first) + "\":\"" + json_escape(s.attributes[i].second) + "\"";
    }
    j += "}}";
    return j;
}

// appends and flushes exactly one span line to `path` (created if missing).
// Called from SpanScope's destructor as each span closes, so a run's spans
// hit disk incrementally -- a crash or hang mid-run still leaves every span
// completed so far on disk, instead of losing them all to an end-of-process
// batch write.
inline bool write_span_jsonl(const std::string & path, const Span & s) {
    FILE * f = std::fopen(path.c_str(), "a");
    if (!f) return false;
    std::string line = span_to_json(s) + "\n";
    bool ok = std::fwrite(line.data(), 1, line.size(), f) == line.size();
    std::fflush(f);
    std::fclose(f);
    return ok;
}

// appends one JSON line per span to `path` (created if missing) -- batch
// form, kept for callers (e.g. library/API consumers) that collect spans in
// memory and want one write; the CLI uses the incremental write_span_jsonl
// above instead.
inline bool write_spans_jsonl(const std::string & path, const std::vector<Span> & spans) {
    FILE * f = std::fopen(path.c_str(), "a");
    if (!f) return false;
    for (const Span & s : spans) {
        std::string line = span_to_json(s) + "\n";
        if (std::fwrite(line.data(), 1, line.size(), f) != line.size()) { std::fclose(f); return false; }
    }
    std::fflush(f);
    std::fclose(f);
    return true;
}
