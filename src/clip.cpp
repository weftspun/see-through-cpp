#include "clip.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// tokenizer
// ---------------------------------------------------------------------------

// GPT-2 bytes-to-unicode: printable latin-1 bytes map to themselves, the rest
// to 256+n; returned as UTF-8 strings (the alphabet vocab.json is written in)
static std::string cp_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char) cp;
    } else if (cp < 0x800) {
        s += (char) (0xC0 | (cp >> 6));
        s += (char) (0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char) (0xE0 | (cp >> 12));
        s += (char) (0x80 | ((cp >> 6) & 0x3F));
        s += (char) (0x80 | (cp & 0x3F));
    } else {
        s += (char) (0xF0 | (cp >> 18));
        s += (char) (0x80 | ((cp >> 12) & 0x3F));
        s += (char) (0x80 | ((cp >> 6) & 0x3F));
        s += (char) (0x80 | (cp & 0x3F));
    }
    return s;
}

static const std::string * byte_to_unicode(uint8_t b) {
    static std::string tab[256];
    static bool init = false;
    if (!init) {
        int n = 0;
        for (int i = 0; i < 256; i++) {
            bool printable = (i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255);
            tab[i] = cp_to_utf8(printable ? (uint32_t) i : (uint32_t) (256 + n++));
        }
        init = true;
    }
    return &tab[b];
}

// minimal parser for vocab.json ({"token": id, ...}); handles \uXXXX incl.
// surrogate pairs and the standard short escapes
static bool parse_vocab_json(const std::string & js, std::map<std::string, int32_t> & vocab) {
    size_t i = 0, n = js.size();
    auto skip_ws = [&] { while (i < n && isspace((unsigned char) js[i])) i++; };
    skip_ws();
    if (i >= n || js[i] != '{') return false;
    i++;
    auto hex4 = [&](size_t p) -> uint32_t {
        uint32_t v = 0;
        for (int k = 0; k < 4; k++) {
            char c = js[p + k];
            v = v * 16 + (c <= '9' ? c - '0' : (c | 32) - 'a' + 10);
        }
        return v;
    };
    for (;;) {
        skip_ws();
        if (i < n && js[i] == '}') return true;
        if (i >= n || js[i] != '"') return false;
        i++;
        std::string key;
        while (i < n && js[i] != '"') {
            char c = js[i];
            if (c == '\\') {
                char e = js[++i];
                switch (e) {
                    case 'n': key += '\n'; break;
                    case 't': key += '\t'; break;
                    case 'r': key += '\r'; break;
                    case 'b': key += '\b'; break;
                    case 'f': key += '\f'; break;
                    case 'u': {
                        uint32_t cp = hex4(i + 1);
                        i += 4;
                        if (cp >= 0xD800 && cp < 0xDC00 && i + 6 < n && js[i + 1] == '\\' && js[i + 2] == 'u') {
                            uint32_t lo = hex4(i + 3);
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            i += 6;
                        }
                        key += cp_to_utf8(cp);
                        break;
                    }
                    default: key += e; break;   // \" \\ \/
                }
            } else {
                key += c;
            }
            i++;
        }
        i++;   // closing quote
        skip_ws();
        if (i >= n || js[i] != ':') return false;
        i++;
        skip_ws();
        char * end = nullptr;
        long id = strtol(js.c_str() + i, &end, 10);
        if (end == js.c_str() + i) return false;
        i = end - js.c_str();
        vocab[key] = (int32_t) id;
        skip_ws();
        if (i < n && js[i] == ',') i++;
    }
}

bool ClipTokenizer::load(const std::string & vocab_json, const std::string & merges_txt) {
    if (!parse_vocab_json(vocab_json, vocab)) return false;
    // merges.txt: optional "#version" header, then "left right" per line
    size_t pos = 0;
    int rank = 0;
    while (pos < merges_txt.size()) {
        size_t eol = merges_txt.find('\n', pos);
        if (eol == std::string::npos) eol = merges_txt.size();
        std::string line = merges_txt.substr(pos, eol - pos);
        pos = eol + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        merge_rank[{line.substr(0, sp), line.substr(sp + 1)}] = rank++;
    }
    return !vocab.empty() && !merge_rank.empty();
}

// BPE over one regex word: bytes -> unicode symbols, "</w>" on the last, then
// repeatedly merge the lowest-rank adjacent pair
static void bpe_word(const ClipTokenizer & t, const std::string & word, std::vector<int32_t> & out) {
    std::vector<std::string> sym;
    for (unsigned char c : word) sym.push_back(*byte_to_unicode(c));
    if (sym.empty()) return;
    sym.back() += "</w>";
    while (sym.size() >= 2) {
        int best = -1;
        size_t bi = 0;
        for (size_t i = 0; i + 1 < sym.size(); i++) {
            auto it = t.merge_rank.find({sym[i], sym[i + 1]});
            if (it != t.merge_rank.end() && (best < 0 || it->second < best)) { best = it->second; bi = i; }
        }
        if (best < 0) break;
        const std::string first = sym[bi], second = sym[bi + 1];
        std::vector<std::string> merged;
        for (size_t i = 0; i < sym.size(); i++) {
            if (i + 1 < sym.size() && sym[i] == first && sym[i + 1] == second) {
                merged.push_back(first + second);
                i++;
            } else {
                merged.push_back(sym[i]);
            }
        }
        sym = std::move(merged);
    }
    for (const std::string & s : sym) {
        auto it = t.vocab.find(s);
        if (it != t.vocab.end()) out.push_back(it->second);
        else fprintf(stderr, "clip tokenizer: no vocab entry for '%s'\n", s.c_str());
    }
}

std::vector<int32_t> ClipTokenizer::encode(const std::string & text) const {
    // lowercase + the CLIP word pattern: contractions | letter runs | single
    // digit | punctuation runs (bytes >= 0x80 are treated as letters)
    std::string lo = text;
    for (char & c : lo) c = (char) tolower((unsigned char) c);

    std::vector<int32_t> ids;
    auto is_letter = [](unsigned char c) { return isalpha(c) || c >= 0x80; };
    size_t i = 0, n = lo.size();
    while (i < n) {
        unsigned char c = lo[i];
        if (isspace(c)) { i++; continue; }
        size_t beg = i;
        if (c == '\'') {
            static const char * contr[] = { "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };
            size_t len = 0;
            for (const char * p : contr) {
                size_t l = strlen(p);
                if (lo.compare(i, l, p) == 0 && l > len) len = l;
            }
            if (len) { i += len; bpe_word(*this, lo.substr(beg, len), ids); continue; }
        }
        if (is_letter(c)) {
            while (i < n && is_letter((unsigned char) lo[i])) i++;
        } else if (isdigit(c)) {
            i++;
        } else {
            while (i < n) {
                unsigned char d = lo[i];
                if (isspace(d) || is_letter(d) || isdigit(d)) break;
                i++;
            }
        }
        bpe_word(*this, lo.substr(beg, i - beg), ids);
    }
    return ids;
}

std::vector<int32_t> ClipTokenizer::encode_padded(const std::string & text, int n_ctx,
                                                  int32_t pad_id, int * eos_pos) const {
    std::vector<int32_t> ids = encode(text);
    if ((int) ids.size() > n_ctx - 2) ids.resize(n_ctx - 2);
    std::vector<int32_t> out;
    out.push_back(bos_id);
    out.insert(out.end(), ids.begin(), ids.end());
    out.push_back(eos_id);
    if (eos_pos) *eos_pos = (int) out.size() - 1;
    out.resize(n_ctx, pad_id);
    return out;
}

// ---------------------------------------------------------------------------
// text transformer graph
// ---------------------------------------------------------------------------

static int cfg_int(const std::string & js, const char * key, int fallback) {
    std::string pat = std::string("\"") + key + "\"";
    size_t p = js.find(pat);
    if (p == std::string::npos) return fallback;
    p = js.find(':', p);
    if (p == std::string::npos) return fallback;
    return atoi(js.c_str() + p + 1);
}

ClipParams clip_params_from_config(const std::string & config_json) {
    ClipParams p;
    p.n_layer    = cfg_int(config_json, "num_hidden_layers", 12);
    p.n_head     = cfg_int(config_json, "num_attention_heads", 12);
    p.d_model    = cfg_int(config_json, "hidden_size", 768);
    p.quick_gelu = config_json.find("\"quick_gelu\"") != std::string::npos;
    return p;
}

void clip_text_graph(Model & m, ggml_tensor * ids, const ClipParams & p,
                     ggml_tensor ** penultimate, ggml_tensor ** final_out) {
    ggml_context * ctx = m.ctx_g;
    const int64_t n  = ids->ne[0];
    const int64_t d  = p.d_model;
    const int64_t hd = d / p.n_head;

    ggml_tensor * x = ggml_get_rows(ctx, m.get("text_model.embeddings.token_embedding.weight"), ids);
    ggml_tensor * pos_w = m.get("text_model.embeddings.position_embedding.weight");
    ggml_tensor * pos = ggml_view_2d(ctx, pos_w, d, n, pos_w->nb[1], 0);
    x = ggml_add(ctx, x, ggml_cast(ctx, pos, GGML_TYPE_F32));

    if (penultimate) *penultimate = nullptr;
    for (int l = 0; l < p.n_layer; l++) {
        if (penultimate && l == p.n_layer - 1) *penultimate = x;
        const std::string pre = "text_model.encoder.layers." + std::to_string(l) + ".";

        // pre-LN causal multi-head self-attention
        ggml_tensor * h = layer_norm_affine(m, x, pre + "layer_norm1");
        ggml_tensor * q = linear(m, h, pre + "self_attn.q_proj");
        ggml_tensor * k = linear(m, h, pre + "self_attn.k_proj");
        ggml_tensor * v = linear(m, h, pre + "self_attn.v_proj");
        q = ggml_scale(ctx, q, 1.0f / sqrtf((float) hd));
        q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, q, hd, p.n_head, n), 0, 2, 1, 3)); // (hd,n,H)
        k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, k, hd, p.n_head, n), 0, 2, 1, 3));
        v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, hd, p.n_head, n), 1, 2, 0, 3)); // (n,hd,H)
        ggml_tensor * kq = ggml_mul_mat(ctx, k, q);                    // (n,n,H)
        kq = ggml_soft_max(ctx, ggml_diag_mask_inf(ctx, kq, 0));
        ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);                  // (hd,n,H)
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));      // (hd,H,n)
        kqv = ggml_reshape_2d(ctx, kqv, d, n);
        x = ggml_add(ctx, x, linear(m, kqv, pre + "self_attn.out_proj"));

        // MLP
        h = layer_norm_affine(m, x, pre + "layer_norm2");
        h = linear(m, h, pre + "mlp.fc1");
        h = p.quick_gelu ? ggml_gelu_quick(ctx, h) : ggml_gelu_erf(ctx, h);
        h = linear(m, h, pre + "mlp.fc2");
        x = ggml_add(ctx, x, h);
    }
    if (final_out) *final_out = layer_norm_affine(m, x, "text_model.final_layer_norm");
}

ggml_tensor * clip_pooled_projection(Model & m, ggml_tensor * final_out, int eos_pos) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * row = ggml_view_2d(ctx, final_out, final_out->ne[0], 1,
                                     final_out->nb[1], (size_t) eos_pos * final_out->nb[1]);
    return ggml_mul_mat(ctx, m.get("text_projection.weight"), ggml_cont(ctx, row));
}
