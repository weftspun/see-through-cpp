#include "winograd.h"

// Winograd F(2x2,3x3): the three small transforms (filter G, input B, output
// A) are implemented as their classic explicit add/sub closed forms (Lavin &
// Gray) instead of generic matrix multiplies -- G/B/A are fixed, known,
// sparse-valued constant matrices ({0, +-0.5, +-1}), so "matmul by G" is
// just a handful of adds/subtracts/halvings on VIEWS of the tensor, no
// matmul-family op (mul_mat OR out_prod) involved at all for these steps.
//
// This replaced an earlier version built on ggml_out_prod for every step.
// That approach was correct on ggml's CPU backend at every scale tested
// (including production channel counts), but diverged from the reference
// im2col conv2d() by several percent on this project's actual target
// (Vulkan) specifically when a small-K (3 or 4) transform matrix was
// batched over a large IC*OC. Root-caused down to: the divergence reproduces
// in a 5-node minimal single-purpose process (ruling out test-harness
// interference), is exact on CPU (ruling out the math), but the *exact*
// Vulkan mechanism was never pinned down despite extensive bisection -- every
// attempt to isolate it in a smaller repro came back correct, which doesn't
// make sense unless either ggml-vulkan's out_prod has a narrow, hard-to-hit
// precondition, or (more likely given how much of this session's "isolated
// passing" evidence turned out to rest on hand-derived host reference math
// that had its own bugs) the true story is muddier than a single clean
// mechanism. Sidestepping matmul-family ops for the small transforms
// entirely -- where explicit unrolling is both correct-by-inspection and
// cheap -- removes the need to ever resolve that.
//
// The one place a real, variable-size reduction remains (summing the
// elementwise filter*input product over IC, which can be in the hundreds in
// production) can't be unrolled this way without exploding the graph, so it
// still uses ggml_out_prod. That call has K=IC (large) and batch=16 (fixed,
// small) -- the opposite shape profile from what broke above (K=3or4 large
// batch) -- and independent bisection of that specific shape profile (large
// K, batch=1) came back exact against a host double-precision reference.
//
// Tile reassembly (2x2 output tiles -> full [OW,OH]) is the standard
// split-permute-merge decomposition of pixel-shuffle/depth-to-space, done in
// two passes (one per spatial axis) since ggml caps tensors at 4 dims and
// the naive single-shot version needs 6 simultaneous axes (tw,th,OW_,OH_,
// OC,N). ggml_win_part/win_unpart would do this in one call but are CPU-only
// in ggml's Vulkan backend (no GGML_OP_WIN_PART/WIN_UNPART case in
// ggml-vulkan.cpp) -- reshape/permute/cont are universally supported instead.
//
// N (batch) is required to be 1 purely to keep the reassembly's dimension
// budget simple. conv2d_winograd_3x3s1 returns nullptr for N>1 (and odd W/H,
// non-3x3 kernels, etc.) so callers fall back to conv2d().

namespace {

// view of X (4D) with axis `ax` (0 or 1) fixed at index `idx`; ne at that
// axis stays 1 (not squeezed) so every slice at a given stage has identical
// shape and ggml_add/ggml_sub/ggml_scale apply directly, no broadcast needed.
ggml_tensor * slice1(ggml_context * ctx, ggml_tensor * X, int ax, int idx) {
    int64_t ne[4] = { X->ne[0], X->ne[1], X->ne[2], X->ne[3] };
    ne[ax] = 1;
    size_t offset = (size_t) idx * X->nb[ax];
    return ggml_view_4d(ctx, X, ne[0], ne[1], ne[2], ne[3], X->nb[1], X->nb[2], X->nb[3], offset);
}

ggml_tensor * add2(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) { return ggml_add(ctx, a, b); }
ggml_tensor * sub2(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) { return ggml_sub(ctx, a, b); }
ggml_tensor * half(ggml_context * ctx, ggml_tensor * a) { return ggml_scale(ctx, a, 0.5f); }

// filter transform, one axis: G = [[1,0,0],[.5,.5,.5],[.5,-.5,.5],[0,0,1]]
// applied to 3 slices (g0,g1,g2) along `ax`, producing 4 slices concatenated
// back along the same axis.
ggml_tensor * g_axis(ggml_context * ctx, ggml_tensor * X, int ax) {
    ggml_tensor * g0 = ggml_cont(ctx, slice1(ctx, X, ax, 0));
    ggml_tensor * g1 = ggml_cont(ctx, slice1(ctx, X, ax, 1));
    ggml_tensor * g2 = ggml_cont(ctx, slice1(ctx, X, ax, 2));
    ggml_tensor * t0 = g0;
    ggml_tensor * t1 = half(ctx, add2(ctx, add2(ctx, g0, g1), g2));
    ggml_tensor * t2 = half(ctx, add2(ctx, sub2(ctx, g0, g1), g2));
    ggml_tensor * t3 = g2;
    ggml_tensor * r = ggml_concat(ctx, t0, t1, ax);
    r = ggml_concat(ctx, r, t2, ax);
    return ggml_concat(ctx, r, t3, ax);
}

// input transform, one axis: B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
// applied to 4 slices (d0..d3) along `ax`.
ggml_tensor * b_axis(ggml_context * ctx, ggml_tensor * X, int ax) {
    ggml_tensor * d0 = ggml_cont(ctx, slice1(ctx, X, ax, 0));
    ggml_tensor * d1 = ggml_cont(ctx, slice1(ctx, X, ax, 1));
    ggml_tensor * d2 = ggml_cont(ctx, slice1(ctx, X, ax, 2));
    ggml_tensor * d3 = ggml_cont(ctx, slice1(ctx, X, ax, 3));
    ggml_tensor * v0 = sub2(ctx, d0, d2);
    ggml_tensor * v1 = add2(ctx, d1, d2);
    ggml_tensor * v2 = sub2(ctx, d2, d1);
    ggml_tensor * v3 = sub2(ctx, d1, d3);
    ggml_tensor * r = ggml_concat(ctx, v0, v1, ax);
    r = ggml_concat(ctx, r, v2, ax);
    return ggml_concat(ctx, r, v3, ax);
}

// output transform, one axis: A^T = [[1,1,1,0],[0,1,-1,-1]]
// applied to 4 slices (m0..m3) along `ax`, producing 2 slices.
ggml_tensor * a_axis(ggml_context * ctx, ggml_tensor * X, int ax) {
    ggml_tensor * m0 = ggml_cont(ctx, slice1(ctx, X, ax, 0));
    ggml_tensor * m1 = ggml_cont(ctx, slice1(ctx, X, ax, 1));
    ggml_tensor * m2 = ggml_cont(ctx, slice1(ctx, X, ax, 2));
    ggml_tensor * m3 = ggml_cont(ctx, slice1(ctx, X, ax, 3));
    ggml_tensor * y0 = add2(ctx, add2(ctx, m0, m1), m2);
    ggml_tensor * y1 = sub2(ctx, sub2(ctx, m1, m2), m3);
    return ggml_concat(ctx, y0, y1, ax);
}

} // namespace

ggml_tensor * conv2d_winograd_3x3s1(Model & m, ggml_tensor * x, const std::string & pre) {
    ggml_tensor * w = m.get(pre + ".weight");   // [KW=3, KH=3, IC, OC]
    if (w->ne[0] != 3 || w->ne[1] != 3) { return nullptr; }

    const int64_t W = x->ne[0], H = x->ne[1], IC = x->ne[2], N = x->ne[3];
    const int64_t OC = w->ne[3];
    if (N != 1 || W < 2 || H < 2 || W % 2 != 0 || H % 2 != 0 || IC != w->ne[2]) { return nullptr; }

    ggml_context * ctx = m.ctx_g;

    // --- filter transform: U = G g G^T, per (ic,oc), operating directly on
    // w's own [KW,KH,IC,OC] layout -- no reshape/permute needed since each
    // axis is transformed independently and in place.
    ggml_tensor * U = g_axis(ctx, w, 1);    // KH axis: [3,4,IC,OC]
    U = g_axis(ctx, U, 0);                  // KW axis: [4,4,IC,OC]

    // --- input transform: V = B^T d B, per (ic,tile) ---
    // 4x4 overlapping patches, stride 2, pad 1 -- exactly the receptive
    // field union of a 3x3 pad-1 stride-1 conv's two output columns/rows
    // per tile. "kshape" is a shape-only placeholder: ggml_im2col never
    // reads its data, only a->ne[0..2] (kernel W/H, and channel count for
    // the is_2D shape assert).
    ggml_tensor * kshape = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 4, IC, 1);
    ggml_tensor * patches = ggml_im2col(ctx, kshape, x, 2, 2, 1, 1, 1, 1, true, GGML_TYPE_F32);
    const int64_t OW_ = patches->ne[1], OH_ = patches->ne[2];      // W/2, H/2
    const int64_t T = OW_ * OH_;                                   // N==1
    patches = ggml_reshape_2d(ctx, patches, IC * 16, T);
    patches = ggml_reshape_4d(ctx, patches, 4, 4, IC, T);           // [KW,KH,IC,T]
    ggml_tensor * V = b_axis(ctx, patches, 1);   // KH axis
    V = b_axis(ctx, V, 0);                       // KW axis

    // --- elementwise product summed over IC, per each of the 16 slots ---
    // this is the one genuine variable-size (IC can be hundreds) reduction,
    // so it still uses ggml_out_prod rather than unrolling.
    ggml_tensor * Ub = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, U, 2, 3, 0, 1)), IC, OC, 16);
    ggml_tensor * Vb = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, V, 2, 3, 0, 1)), IC, T, 16);
    ggml_tensor * Ub_op = ggml_cont(ctx, ggml_permute(ctx, Ub, 1, 0, 2, 3));   // [OC,IC,16]
    ggml_tensor * Vb_op = ggml_cont(ctx, ggml_permute(ctx, Vb, 1, 0, 2, 3));   // [T,IC,16]
    // Loop over the 16 positions as 16 separate batch=1 out_prod calls,
    // rather than one batched (ne2=16) call: the batch=1/large-K shape is
    // what's independently verified exact; batch>1 with large K was never
    // actually isolated (every "isolated" test that claimed to cover it
    // turned out to have its own bug -- see git history/commit message for
    // this file). 16 extra graph nodes is cheap next to the alternative of
    // trusting an unverified shape profile in the hot path.
    ggml_tensor * Mb = nullptr;
    for (int64_t pos = 0; pos < 16; pos++) {
        ggml_tensor * Ub_pos = ggml_cont(ctx, slice1(ctx, Ub_op, 2, pos));   // [OC,IC,1]
        ggml_tensor * Vb_pos = ggml_cont(ctx, slice1(ctx, Vb_op, 2, pos));   // [T,IC,1]
        Ub_pos = ggml_reshape_2d(ctx, Ub_pos, OC, IC);
        Vb_pos = ggml_reshape_2d(ctx, Vb_pos, T, IC);
        ggml_tensor * Mb_pos = ggml_out_prod(ctx, Ub_pos, Vb_pos);           // [OC,T]
        Mb_pos = ggml_reshape_3d(ctx, Mb_pos, OC, T, 1);
        Mb = Mb ? ggml_concat(ctx, Mb, Mb_pos, 2) : Mb_pos;                  // [OC,T,16]
    }

    // --- output transform: Y = A^T M A, per (oc,tile) ---
    ggml_tensor * M4 = ggml_reshape_4d(ctx, Mb, OC, T, 4, 4);                 // [OC,T,posKW,posKH]
    ggml_tensor * Mt = ggml_cont(ctx, ggml_permute(ctx, M4, 2, 3, 0, 1));     // [posKW,posKH,OC,T]
    ggml_tensor * Y = a_axis(ctx, Mt, 1);    // posKH axis: [4,2,OC,T]
    Y = a_axis(ctx, Y, 0);                   // posKW axis: [2,2,OC,T]

    // --- reassemble 2x2 tiles into the full [OW=W,OH=H,OC,1] image ---
    // split-permute-merge, one spatial axis per pass (see header comment).
    // Pass 1 (W): Y=[tw,th,OC,T] -> merge(th,OC) -> split T=(ow_,oh_) ->
    // permute ow_ next to tw -> merge(tw,ow_)->OW.
    ggml_tensor * p1 = ggml_reshape_3d(ctx, Y, 2, 2 * OC, T);              // [tw, th*OC, T]
    p1 = ggml_reshape_4d(ctx, p1, 2, 2 * OC, OW_, OH_);                   // [tw, th*OC, ow_, oh_]
    p1 = ggml_cont(ctx, ggml_permute(ctx, p1, 0, 2, 1, 3));               // [tw, ow_, th*OC, oh_]
    p1 = ggml_reshape_3d(ctx, p1, W, 2 * OC, OH_);                        // [OW, th*OC, oh_]
    // Pass 2 (H): split th*OC=(th,OC) -> permute oh_ next to th ->
    // merge(th,oh_)->OH, leaving OC last.
    ggml_tensor * p2 = ggml_reshape_4d(ctx, p1, W, 2, OC, OH_);           // [OW, th, OC, oh_]
    p2 = ggml_cont(ctx, ggml_permute(ctx, p2, 0, 1, 3, 2));               // [OW, th, oh_, OC]
    ggml_tensor * out = ggml_reshape_3d(ctx, p2, W, H, OC);               // [OW, OH, OC]
    out = ggml_reshape_4d(ctx, out, W, H, OC, 1);

    if (m.has(pre + ".bias")) { out = ggml_add(ctx, out, bias4d(ctx, m.get(pre + ".bias"))); }
    return out;
}
