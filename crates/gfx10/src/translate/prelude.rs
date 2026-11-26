//! Static MSL prelude helpers included in generated shader libraries.

/// The compute loop guard (see `Tx::loop_guard`). A thread checks the
/// dispatch's budget at its first backward branch and adds to it every unit.
pub(super) const GUARD_PRELUDE: &str = r#"
static inline bool guard_stop(device atomic_uint* g, uint back) {
    bool stop = back > GUARD_LIMIT;
    if (back % GUARD_UNIT == 0u) {
        stop = atomic_fetch_add_explicit(&g[0], 1u, memory_order_relaxed) >= GUARD_BUDGET || stop;
    } else if (back == 1u) {
        stop = atomic_load_explicit(&g[0], memory_order_relaxed) >= GUARD_BUDGET || stop;
    }
    if (stop) atomic_fetch_add_explicit(&g[1], 1u, memory_order_relaxed);
    return stop;
}

static inline bool guard_leave(device atomic_uint* g, uint back, thread uint& bail_at, threadgroup bool& tg_bail) {
    if (bail_at == 0u && guard_stop(g, back)) {
        bail_at = back;
        tg_bail = true;
    }
    return bail_at != 0u && back - bail_at > GUARD_UNIT;
}
"#;

/// The loop guard of lockstep programs (after `GUARD_PRELUDE` and `WAVE_PRELUDE`): a wave
/// leaves together once any of its threads' budget is spent, and the rest of the group
/// leaves at its next `s_barrier`.
pub(super) const LOCKSTEP_GUARD_PRELUDE: &str = r#"
static inline bool lockstep_leave(device atomic_uint* g, uint back, thread Wave& w, threadgroup bool& tg_bail) {
    bool leave = wave_any(guard_stop(g, back), w);
    if (leave) tg_bail = true;
    return leave;
}
"#;

pub(super) const PRELUDE: &str = r#"#include <metal_stdlib>
using namespace metal;

#define F(x) as_type<float>(uint(x))
#define U(x) as_type<uint>(float(x))

static inline uint bfe_u32(uint v, uint o, uint w) {
    w = min(w, 32u);
    if (w == 0u) return 0u;
    uint m = w >= 32u ? 0xffffffffu : ((1u << w) - 1u);
    return (v >> (o & 31u)) & m;
}

static inline uint bfe_i32(uint v, uint o, uint w) {
    if (w == 0u) return 0u;
    uint f = uint(int(v) >> (o & 31u));
    if (w >= 32u) return f;
    return uint(int(f << (32u - w)) >> (32u - w));
}

static inline ulong bfe_u64(ulong v, uint ctrl) {
    uint o = ctrl & 63u;
    uint w = min((ctrl >> 16) & 0x7fu, 64u);
    if (w == 0u) return 0ul;
    ulong m = w >= 64u ? ~0ul : ((1ul << w) - 1ul);
    return (v >> o) & m;
}

static inline ulong bfe_i64(ulong v, uint ctrl) {
    uint w = (ctrl >> 16) & 0x7fu;
    if (w == 0u) return 0ul;
    ulong f = ulong(long(v) >> (ctrl & 63u));
    if (w >= 64u) return f;
    return ulong(long(f << (64u - w)) >> (64u - w));
}

// Leading bits equal to the sign bit, or -1 when all are.
static inline uint ffbh_i32(uint v) {
    uint x = int(v) < 0 ? ~v : v;
    return x == 0u ? 0xffffffffu : clz(x);
}

static inline uint ffbh_i64(ulong v) {
    ulong x = long(v) < 0 ? ~v : v;
    return x == 0ul ? 0xffffffffu : uint(clz(x));
}

// Sum of the bytes' absolute differences; masked, bytes where `b` is zero count nothing.
static inline uint sad_u8(uint a, uint b, bool masked) {
    uint r = 0u;
    for (uint i = 0u; i < 32u; i += 8u) {
        uint x = (a >> i) & 0xffu, y = (b >> i) & 0xffu;
        if (!masked || y != 0u) r += absdiff(x, y);
    }
    return r;
}

static inline uint lerp_u8(uint a, uint b, uint c) {
    uint r = 0u;
    for (uint i = 0u; i < 32u; i += 8u) {
        r |= ((((a >> i) & 0xffu) + ((b >> i) & 0xffu) + ((c >> i) & 1u)) >> 1) << i;
    }
    return r;
}

// Bytes of {a, b} (b low) by selectors: 8 to 11 replicate a sign bit, 12 is
// zero, 13 and up are 0xff.
static inline uint perm_b32(uint a, uint b, uint sel) {
    ulong w = ulong(a) << 32 | ulong(b);
    uint r = 0u;
    for (uint i = 0u; i < 32u; i += 8u) {
        uint s = (sel >> i) & 0xffu;
        uint byte;
        if (s >= 13u) byte = 0xffu;
        else if (s == 12u) byte = 0u;
        else if (s >= 8u) byte = uint((w >> (16u * (s - 8u) + 15u)) & 1ul) * 0xffu;
        else byte = uint((w >> (8u * s)) & 0xfful);
        r |= byte << i;
    }
    return r;
}

static inline ulong brev64(ulong v) {
    return ulong(reverse_bits(uint(v))) << 32 | ulong(reverse_bits(uint(v >> 32)));
}

static inline ulong wqm64(ulong v) {
    ulong r = 0ul;
    for (uint q = 0u; q < 16u; q++) {
        if ((v >> (4u * q)) & 0xful) r |= 0xful << (4u * q);
    }
    return r;
}

static inline uint quadmask64(ulong v) {
    uint r = 0u;
    for (uint q = 0u; q < 16u; q++) {
        if ((v >> (4u * q)) & 0xful) r |= 1u << q;
    }
    return r;
}

static inline int med3_i(int a, int b, int c) {
    return max(min(a, b), min(max(a, b), c));
}


// A half's bits as a float, in integer steps: the compiler may keep a
// value converted to half at float precision, an export may not.
static inline float f16_float(uint h) {
    uint s = (h & 0x8000u) << 16;
    uint e = (h >> 10) & 0x1Fu;
    uint m = h & 0x3FFu;
    if (e == 0u) return as_type<float>(s | as_type<uint>(float(m) * 0x1p-24f));
    if (e == 31u) return as_type<float>(s | 0x7F800000u | m << 13);
    return as_type<float>(s | (e + 112u) << 23 | m << 13);
}

// A float's bits as a half rounded toward zero, in integer steps: past the
// largest half it becomes +-65504; infinity stays, NaN keeps its sign and top
// payload bits and becomes quiet.
static inline uint f16_rtz_bits(float x) {
    uint b = as_type<uint>(x);
    uint s = (b >> 16) & 0x8000u;
    uint a = b & 0x7FFFFFFFu;
    if (a > 0x7F800000u) return s | 0x7E00u | (a >> 13 & 0x3FFu);
    if (a == 0x7F800000u) return s | 0x7C00u;
    int e = int(a >> 23) - 112;
    if (e >= 31) return s | 0x7BFFu;
    if (e <= 0) {
        if (e < -10) return s;
        return s | ((a & 0x7FFFFFu) | 0x800000u) >> uint(14 - e);
    }
    return s | uint(e) << 10 | (a & 0x7FFFFFu) >> 13;
}

static inline half half_rtz(float x) {
    return as_type<half>(ushort(f16_rtz_bits(x)));
}

static inline float f16_trunc(float x) {
    return f16_float(f16_rtz_bits(x));
}

// The CB's blend, measured on RDNA2: 32-bit float arithmetic, a ZERO factor
// drops its term (NaN or infinity times it too), min and max take the factors
// and ignore a NaN, and BOTH_SRC_ALPHA and BOTH_INV_SRC_ALPHA (11, 12) are 0.
static inline float cb_factor(uint f, uint c, float4 s, float4 d, float4 s1, float4 k) {
    switch (f) {
    case 1u: return 1.0f;
    case 2u: return s[c];
    case 3u: return 1.0f - s[c];
    case 4u: return s.w;
    case 5u: return 1.0f - s.w;
    case 6u: return d.w;
    case 7u: return 1.0f - d.w;
    case 8u: return d[c];
    case 9u: return 1.0f - d[c];
    case 10u: return c == 3u ? 1.0f : fmin(s.w, 1.0f - d.w);
    case 13u: return k[c];
    case 14u: return 1.0f - k[c];
    case 15u: return s1[c];
    case 16u: return 1.0f - s1[c];
    case 17u: return s1.w;
    case 18u: return 1.0f - s1.w;
    case 19u: return k.w;
    case 20u: return 1.0f - k.w;
    default: return 0.0f;
    }
}

static inline float4 cb_blend(float4 s, float4 d, float4 s1, float4 k, uint w) {
    if ((w >> 15) & 1u) {
        uint4 u = as_type<uint4>(d);
        d = as_type<float4>((u + 0x7FFu + ((u >> 12) & 1u)) & 0xFFFFF000u);
    }
    uint clamp_kind = (w >> 13) & 3u;
    if (clamp_kind != 0u) {
        float lo = clamp_kind == 1u ? 0.0f : -1.0f;
        s = clamp(s, lo, 1.0f);
        s1 = clamp(s1, lo, 1.0f);
        k = clamp(k, lo, 1.0f);
    }
    float4 r;
    for (uint c = 0u; c < 4u; c++) {
        uint fields = c == 3u ? w >> 16 : w;
        uint sf = fields & 0x1Fu, op = (fields >> 5) & 7u, df = (fields >> 8) & 0x1Fu;
        float a = sf == 0u ? 0.0f : s[c] * cb_factor(sf, c, s, d, s1, k);
        float b = df == 0u ? 0.0f : d[c] * cb_factor(df, c, s, d, s1, k);
        switch (op) {
        case 1u: r[c] = a - b; break;
        case 2u: r[c] = fmin(a, b); break;
        case 3u: r[c] = fmax(a, b); break;
        case 4u: r[c] = b - a; break;
        default: r[c] = a + b; break;
        }
    }
    return r;
}

// A blend result as a 16-bit float target keeps it: rounded toward zero, and
// every NaN the negative quiet one (0xFE00), as measured on RDNA2. Its bits go
// out as a half, since Metal makes NaN 0x7E00 when it converts a float.
static inline half4 cb_half(float4 x) {
    half4 r;
    for (uint c = 0u; c < 4u; c++) {
        r[c] = as_type<half>(ushort(isnan(x[c]) ? 0xFE00u : f16_rtz_bits(x[c])));
    }
    return r;
}

// Targets Metal has no format for (Request::packed_targets), packed as
// RDNA2's CB writes them (measured on the Deck): channel widths from bit 0
// up; unorm and snorm round to nearest even, uint and sint saturate; sRGB
// stores each colour channel's 8-bit sRGB value in the low bits of its field
// and alpha as unorm; float fields of 10 and 11 bits are unsigned floats with
// a 5-bit exponent, rounded toward zero, 16-bit ones halves rounded toward
// zero. What a 2-bit float field holds for values of 1 and up is not known
// (below 1 it is 0).
static inline uint4 cb_widths(uint layout) {
    switch (layout) {
    case 1u: return uint4(8u, 0u, 0u, 0u);
    case 2u: return uint4(16u, 0u, 0u, 0u);
    case 3u: return uint4(8u, 8u, 0u, 0u);
    case 4u: return uint4(32u, 0u, 0u, 0u);
    case 5u: return uint4(16u, 16u, 0u, 0u);
    case 6u: return uint4(11u, 11u, 10u, 0u);
    case 7u: return uint4(10u, 11u, 11u, 0u);
    case 8u: return uint4(2u, 10u, 10u, 10u);
    case 9u: return uint4(10u, 10u, 10u, 2u);
    case 10u: return uint4(8u, 8u, 8u, 8u);
    case 11u: return uint4(32u, 32u, 0u, 0u);
    case 12u: return uint4(16u, 16u, 16u, 16u);
    case 16u: return uint4(5u, 6u, 5u, 0u);
    case 17u: return uint4(5u, 5u, 5u, 1u);
    case 18u: return uint4(1u, 5u, 5u, 5u);
    case 19u: return uint4(4u, 4u, 4u, 4u);
    case 24u: return uint4(9u, 9u, 9u, 5u);
    default: return uint4(32u, 32u, 32u, 32u);
    }
}

static inline uint cb_mask(uint n) {
    return n >= 32u ? 0xFFFFFFFFu : (1u << n) - 1u;
}

static inline uint cb_ufloat(float x, uint b) {
    uint m = b - 5u;
    if (isnan(x)) return (1u << b) - 1u;
    if (!(x > 0.0f)) return 0u;
    if (isinf(x)) return 31u << m;
    uint u = as_type<uint>(x);
    int e = int(u >> 23) - 112;
    if (e >= 31) return (30u << m) | ((1u << m) - 1u);
    if (e <= 0) return uint(x * float(1u << (14u + m)));
    return uint(e) << m | (u & 0x7FFFFFu) >> (23u - m);
}

static inline float cb_ufloat_value(uint f, uint b) {
    uint m = b - 5u, e = f >> m, man = f & ((1u << m) - 1u);
    if (e == 31u) return man == 0u ? INFINITY : NAN;
    if (e == 0u) return float(man) / float(1u << (14u + m));
    return as_type<float>((e + 112u) << 23 | man << (23u - m));
}

static inline uint cb_srgb8(float x) {
    x = clamp(x, 0.0f, 1.0f);
    float y = x <= 0.0031308f ? 12.92f * x : 1.055f * powr(x, 1.0f / 2.4f) - 0.055f;
    return uint(rint(y * 255.0f));
}

static inline uint cb_unorm(float x, uint n) {
    x = clamp(x, 0.0f, 1.0f);
    return n >= 32u ? (x >= 1.0f ? 0xFFFFFFFFu : uint(rint(x * 4294967296.0f))) : uint(rint(x * float(cb_mask(n))));
}

static inline uint cb_field(float x, uint n, uint number, uint c) {
    switch (number) {
    case 1u: {
        float h = n >= 32u ? 2147483647.0f : float(cb_mask(n) >> 1);
        return uint(int(rint(clamp(x, -1.0f, 1.0f) * h))) & cb_mask(n);
    }
    case 6u: return c < 3u ? cb_srgb8(x) & cb_mask(n) : cb_unorm(x, n);
    case 7u:
        if (n == 32u) return as_type<uint>(x);
        if (n == 16u) return f16_rtz_bits(x);
        if (n >= 10u) return cb_ufloat(x, n);
        return uint(clamp(x, 0.0f, float(cb_mask(n))));
    default: return cb_unorm(x, n);
    }
}

static inline float cb_field_value(uint f, uint n, uint number, uint c) {
    switch (number) {
    case 1u: {
        int s = n >= 32u ? int(f) : int(f << (32u - n)) >> (32u - n);
        return max(float(s) / (n >= 32u ? 2147483647.0f : float(cb_mask(n) >> 1)), -1.0f);
    }
    case 6u: {
        if (c == 3u) return float(f) / float(cb_mask(n));
        float y = float(f & 0xFFu) / 255.0f;
        return y <= 0.04045f ? y / 12.92f : powr((y + 0.055f) / 1.055f, 2.4f);
    }
    case 7u:
        if (n == 32u) return as_type<float>(f);
        if (n == 16u) return f16_float(f);
        if (n >= 10u) return cb_ufloat_value(f, n);
        return float(f);
    default: return float(f) / float(cb_mask(n));
    }
}

// RGB9E5: three 9-bit mantissas sharing a 5-bit exponent (bias 15), rounded
// half up (measured on the Deck: halves exported at a tie round up).
static inline uint4 cb_pack_9e5(float4 v) {
    float3 c = clamp(v.xyz, 0.0f, 65408.0f);
    c = select(c, float3(0.0f), isnan(v.xyz));
    float m = max(max(c.x, c.y), c.z);
    int e = max(-16, int(floor(log2(max(m, 1e-30f))))) + 16;
    float scale = exp2(float(e - 24));
    if (uint(floor(m / scale + 0.5f)) == 512u) { e += 1; scale *= 2.0f; }
    uint3 q = uint3(floor(c / scale + 0.5f));
    return uint4(q.x | q.y << 9 | q.z << 18 | uint(e) << 27, 0u, 0u, 0u);
}

static inline float4 cb_unpack_9e5(uint w) {
    float scale = exp2(float(int(w >> 27) - 24));
    return float4(float(w & 0x1FFu) * scale, float((w >> 9) & 0x1FFu) * scale, float((w >> 18) & 0x1FFu) * scale, 1.0f);
}

static inline uint4 cb_pack(float4 v, uint layout, uint number) {
    if (layout == 24u && number == 7u) return cb_pack_9e5(v);
    uint4 b = cb_widths(layout), w = uint4(0u);
    uint at = 0u;
    for (uint c = 0u; c < 4u; c++) {
        if (b[c] == 0u) continue;
        w[at >> 5] |= cb_field(v[c], b[c], number, c) << (at & 31u);
        at += b[c];
    }
    return w;
}

static inline uint4 cb_pack_u(uint4 v, uint layout) {
    uint4 b = cb_widths(layout), w = uint4(0u);
    uint at = 0u;
    for (uint c = 0u; c < 4u; c++) {
        if (b[c] == 0u) continue;
        w[at >> 5] |= min(v[c], cb_mask(b[c])) << (at & 31u);
        at += b[c];
    }
    return w;
}

static inline uint4 cb_pack_s(int4 v, uint layout) {
    uint4 b = cb_widths(layout), w = uint4(0u);
    uint at = 0u;
    for (uint c = 0u; c < 4u; c++) {
        if (b[c] == 0u) continue;
        int hi = int(cb_mask(b[c]) >> 1);
        w[at >> 5] |= (uint(clamp(v[c], -hi - 1, hi)) & cb_mask(b[c])) << (at & 31u);
        at += b[c];
    }
    return w;
}

// What a blend reads of a packed target: the inverse of cb_pack.
static inline float4 cb_unpack(uint4 w, uint layout, uint number) {
    if (layout == 24u && number == 7u) return cb_unpack_9e5(w.x);
    uint4 b = cb_widths(layout);
    float4 r = float4(0.0f, 0.0f, 0.0f, 1.0f);
    uint at = 0u;
    for (uint c = 0u; c < 4u; c++) {
        if (b[c] == 0u) continue;
        r[c] = cb_field_value((w[at >> 5] >> (at & 31u)) & cb_mask(b[c]), b[c], number, c);
        at += b[c];
    }
    return r;
}

// `mem` covers [base, end); windows over [base2, end2) and [base3, end3) lie `delta`
// and `delta3` bytes from `mem` in the GPU's address space. Stores skip [ro, ro_end).
struct Win { ulong base; ulong end; ulong base2; ulong end2; long delta; ulong ro; ulong ro_end; ulong base3; ulong end3; long delta3; };
#define OOB_OFF (-0x7fffffffffffffffl - 1l)

// Byte offset from `mem` of guest address `a`, or OOB_OFF outside both windows.
static inline long goff(Win w, ulong a) {
    if (a >= w.base && a < w.end) return long(a - w.base);
    if (a >= w.base2 && a < w.end2) return long(a - w.base2) + w.delta;
    if (a >= w.base3 && a < w.end3) return long(a - w.base3) + w.delta3;
    return OOB_OFF;
}

static inline uint ld32(device const uchar* mem, Win w, ulong addr) {
    long o = goff(w, addr);
    return o == OOB_OFF ? 0u : *(device const uint*)(mem + o);
}

// Like goff, for `len` bytes at `a` that all lie in one window.
static inline long goff_span(Win w, ulong a, ulong len) {
    if (a >= w.base && a + len <= w.end) return long(a - w.base);
    if (a >= w.base2 && a + len <= w.end2) return long(a - w.base2) + w.delta;
    if (a >= w.base3 && a + len <= w.end3) return long(a - w.base3) + w.delta3;
    return OOB_OFF;
}

// Like goff, and in `run` the bytes from `a` that stay in its window, or outside
// every window, the bytes before the next window begins.
static inline long goff_run(Win w, ulong a, thread ulong& run) {
    if (a >= w.base && a < w.end) { run = w.end - a; return long(a - w.base); }
    if (a >= w.base2 && a < w.end2) { run = w.end2 - a; return long(a - w.base2) + w.delta; }
    if (a >= w.base3 && a < w.end3) { run = w.end3 - a; return long(a - w.base3) + w.delta3; }
    run = ~0ul;
    if (w.base > a) run = min(run, w.base - a);
    if (w.base2 > a) run = min(run, w.base2 - a);
    if (w.base3 > a) run = min(run, w.base3 - a);
    return OOB_OFF;
}

// The dword `b` bytes into two runs from goff_run. Scalar loads span at most 64
// bytes, so two runs cover them unless a window or a gap is shorter than that.
static inline uint ld_runs(device const uchar* mem, long o1, ulong r1, long o2, ulong r2, ulong b) {
    if (b + 4ul <= r1) return o1 == OOB_OFF ? 0u : *(device const uint*)(mem + o1 + long(b));
    b -= r1;
    if (b + 4ul <= r2) return o2 == OOB_OFF ? 0u : *(device const uint*)(mem + o2 + long(b));
    return 0u;
}

// image_bvh_intersect_ray as Mesa radv's software emulation states it. Box nodes
// (types 4, 5) give the children the ray enters, nearest first; triangles (t, det, i, j).
static inline uint4 bvh_intersect(device const uchar* mem, Win w, uint4 d, ulong node, float tmax, float3 o, float3 dir, float3 inv) {
    ulong a = ((ulong(d.x) | (ulong(d.y & 0xffu) << 32)) << 8) + ((node & ~7ul) << 3);
    uint type = uint(node) & 7u;
    if (type == 4u || type == 5u) {
        float4 dist = float4(INFINITY);
        uint4 child = uint4(0xffffffffu);
        inv = clamp(inv, float3(-FLT_MAX), float3(FLT_MAX));
        for (uint i = 0u; i < 4u; i++) {
            float3 lo, hi;
            if (type == 5u) {
                ulong b = a + 16ul + 24ul * i;
                lo = float3(F(ld32(mem, w, b)), F(ld32(mem, w, b + 4ul)), F(ld32(mem, w, b + 8ul)));
                hi = float3(F(ld32(mem, w, b + 12ul)), F(ld32(mem, w, b + 16ul)), F(ld32(mem, w, b + 20ul)));
            } else {
                ulong b = a + 16ul + 12ul * i;
                half2 h0 = as_type<half2>(ld32(mem, w, b));
                half2 h1 = as_type<half2>(ld32(mem, w, b + 4ul));
                half2 h2 = as_type<half2>(ld32(mem, w, b + 8ul));
                lo = float3(float(h0.x), float(h0.y), float(h1.x));
                hi = float3(float(h1.y), float(h2.x), float(h2.y));
            }
            float3 b0 = (lo - o) * inv, b1 = (hi - o) * inv;
            float3 near = min(b0, b1), far = max(b0, b1);
            float t0 = max(max(near.x, near.y), near.z);
            float t1 = min(min(far.x, far.y), far.z);
            if (!isnan(lo.x) && t1 >= max(0.0f, t0) && t0 < tmax) {
                dist[i] = t0;
                child[i] = ld32(mem, w, a + 4ul * i);
            }
        }
        if ((d.y >> 31) != 0u) {
            const uint2 pairs[5] = { uint2(0, 1), uint2(2, 3), uint2(0, 2), uint2(1, 3), uint2(1, 2) };
            for (uint p = 0u; p < 5u; p++) {
                uint x = pairs[p].x, y = pairs[p].y;
                if (dist[y] < dist[x]) {
                    float td = dist[x]; dist[x] = dist[y]; dist[y] = td;
                    uint tc = child[x]; child[x] = child[y]; child[y] = tc;
                }
            }
        }
        return child;
    }
    if (type > 3u) return uint4(0xffffffffu);
    float3 v[3];
    for (uint i = 0u; i < 3u; i++) {
        ulong b = a + 12ul * i;
        v[i] = float3(F(ld32(mem, w, b)), F(ld32(mem, w, b + 4ul)), F(ld32(mem, w, b + 8ul))) - o;
    }
    // Watertight test (Woop, Benthin and Wald, JCGT 2013), as radv does it.
    float3 ad = abs(dir);
    uint kz = ad.x >= ad.y ? (ad.x >= ad.z ? 0u : 2u) : (ad.y >= ad.z ? 1u : 2u);
    uint kx = (kz + 1u) % 3u, ky = (kx + 1u) % 3u;
    if (dir[kz] < 0.0f) { uint t = kx; kx = ky; ky = t; }
    float sz = 1.0f / dir[kz], sx = dir[kx] * sz, sy = dir[ky] * sz;
    float ax = v[0][kx] - sx * v[0][kz], ay = v[0][ky] - sy * v[0][kz];
    float bx = v[1][kx] - sx * v[1][kz], by = v[1][ky] - sy * v[1][kz];
    float cx = v[2][kx] - sx * v[2][kz], cy = v[2][ky] - sy * v[2][kz];
    float u = cx * by - cy * bx, vv = ax * cy - ay * cx, ww = bx * ay - by * ax;
    bool back = u < 0.0f || vv < 0.0f || ww < 0.0f;
    bool front = u > 0.0f || vv > 0.0f || ww > 0.0f;
    if (!(back && front)) {
        float det = u + vv + ww;
        float t = u * sz * v[0][kz] + vv * sz * v[1][kz] + ww * sz * v[2][kz];
        if (!(sign(det) * t < 0.0f)) return uint4(U(t), U(det), U(vv), U(ww));
    }
    return uint4(U(INFINITY), U(1.0f), 0u, 0u);
}

// Whether the range check rejects `bytes` bytes at `offset` in record `index` of
// buffer `vs` (RDNA2 ISA 8.1.5). The SGPR offset is not part of the record offset; in
// OOB_SELECT 3 it shrinks the buffer. With stride 0, modes 0 and 1 check the offset
// against the size.
static inline bool buffer_oob(uint4 vs, uint index, uint offset, uint soffset, uint idxen, uint bytes) {
    uint stride = (vs.y >> 16) & 0x3fffu;
    bool swizzle = (vs.y >> 31) != 0u;
    uint i = idxen != 0u ? index : 0u;
    switch ((vs.w >> 28) & 3u) {
    case 0u: return stride == 0u ? offset >= vs.z : i >= vs.z || offset >= stride;
    case 1u: return stride == 0u ? offset >= vs.z : i >= vs.z;
    case 2u: return vs.z == 0u;
    default:
        return swizzle && stride != 0u ? i >= vs.z || ulong(offset) + bytes > stride
            : ulong(offset) + bytes + soffset > ulong(vs.z);
    }
}

// Guest address of an access at `offset` in record `index`. Swizzled buffers
// interleave the dwords of `index_stride` records; what an access moves lies
// contiguous from there, and accesses without an index stay linear (both measured
// on gfx1033).
static inline ulong buffer_addr(uint4 vs, uint index, uint offset, uint soffset, uint idxen) {
    ulong base = ulong(vs.x) | (ulong(vs.y & 0xffffu) << 32);
    uint stride = (vs.y >> 16) & 0x3fffu;
    uint i = idxen != 0u ? index : 0u;
    if ((vs.y >> 31) == 0u || idxen == 0u) return base + soffset + ulong(i) * stride + offset;
    uint is = 8u << ((vs.w >> 21) & 3u);
    return base + soffset + (ulong(i / is) * stride + (offset & ~3u)) * is + (i % is) * 4u + (offset & 3u);
}

// Byte offset from `mem` of an access of `bytes` bytes, or OOB_OFF when out of range.
static inline long buffer_offset(Win w, uint4 vs, uint index, uint offset, uint soffset, uint idxen, uint bytes) {
    if (buffer_oob(vs, index, offset, soffset, idxen, bytes)) return OOB_OFF;
    return goff(w, buffer_addr(vs, index, offset, soffset, idxen));
}

// As `buffer_offset`, for stores and atomics: OOB_OFF inside [ro, ro_end) too.
static inline long buffer_store_offset(Win w, uint4 vs, uint index, uint offset, uint soffset, uint idxen, uint bytes) {
    if (buffer_oob(vs, index, offset, soffset, idxen, bytes)) return OOB_OFF;
    ulong a = buffer_addr(vs, index, offset, soffset, idxen);
    if (a >= w.ro && a < w.ro_end) return OOB_OFF;
    return goff(w, a);
}

// Offset of dword `k` of a dword access starting at `o0` (OOB_OFF if the access
// is outside every window), or OOB_OFF when that dword fails the range check.
static inline long buffer_dword(long o0, uint4 vs, uint index, uint offset, uint soffset, uint idxen, uint k) {
    if (o0 == OOB_OFF || buffer_oob(vs, index, offset + 4u * k, soffset, idxen, 4u)) return OOB_OFF;
    return o0 + 4 * long(k);
}

static inline long buffer_dwords(Win w, uint4 vs, uint index, uint offset, uint soffset, uint idxen) {
    return goff(w, buffer_addr(vs, index, offset, soffset, idxen));
}

static inline long buffer_store_dwords(Win w, uint4 vs, uint index, uint offset, uint soffset, uint idxen) {
    ulong a = buffer_addr(vs, index, offset, soffset, idxen);
    if (a >= w.ro && a < w.ro_end) return OOB_OFF;
    return goff(w, a);
}

static inline uint4 buffer_load_raw(device const uchar* mem, Win w, uint4 vs, uint index, uint offset, uint soffset, uint idxen, uint n) {
    long o0 = buffer_dwords(w, vs, index, offset, soffset, idxen);
    uint4 r = uint4(0u);
    for (uint k = 0u; k < n; k++) {
        long o = buffer_dword(o0, vs, index, offset, soffset, idxen, k);
        if (o != OOB_OFF) r[k] = *(device const uint*)(mem + o);
    }
    return r;
}

// GLC loads and stores, which other groups may be polling or writing while
// this one runs: atomics, so the compiler keeps them in loops and in order.
static inline uint4 buffer_load_atomic(device uchar* mem, Win w, uint4 vs, uint index, uint offset, uint soffset, uint idxen, uint n) {
    long o0 = buffer_dwords(w, vs, index, offset, soffset, idxen);
    uint4 r = uint4(0u);
    for (uint k = 0u; k < n; k++) {
        long o = buffer_dword(o0, vs, index, offset, soffset, idxen, k);
        if (o == OOB_OFF) continue;
        r[k] = (o & 3) != 0 ? *(device const uint*)(mem + o)
            : atomic_load_explicit((device atomic_uint*)(mem + o), memory_order_relaxed);
    }
    return r;
}

static inline uint4 buffer_load_volatile(device const uchar* mem, Win w, uint4 vs, uint index, uint offset, uint soffset, uint idxen, uint n) {
    long o0 = buffer_dwords(w, vs, index, offset, soffset, idxen);
    uint4 r = uint4(0u);
    for (uint k = 0u; k < n; k++) {
        long o = buffer_dword(o0, vs, index, offset, soffset, idxen, k);
        if (o != OOB_OFF) r[k] = *(volatile device const uint*)(mem + o);
    }
    return r;
}

static inline void store_atomic(device uchar* mem, long o, uint v) {
    if ((o & 3) != 0) { *(device uint*)(mem + o) = v; return; }
    atomic_store_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed);
}

static inline uint4 buffer_load_ubyte(device const uchar* mem, Win w, uint4 vs, uint index, uint offset, uint soffset, uint idxen) {
    long o = buffer_offset(w, vs, index, offset, soffset, idxen, 1u);
    return uint4(o == OOB_OFF ? 0u : uint(mem[o]), 0u, 0u, 0u);
}

static inline uint4 buffer_load_sbyte(device const uchar* mem, Win w, uint4 vs, uint index, uint offset, uint soffset, uint idxen) {
    long o = buffer_offset(w, vs, index, offset, soffset, idxen, 1u);
    return uint4(o == OOB_OFF ? 0u : uint(int(char(mem[o]))), 0u, 0u, 0u);
}

static inline uint4 buffer_load_ushort(device const uchar* mem, Win w, uint4 vs, uint index, uint offset, uint soffset, uint idxen) {
    long o = buffer_offset(w, vs, index, offset, soffset, idxen, 2u);
    return uint4(o == OOB_OFF ? 0u : uint(*(device const ushort*)(mem + o)), 0u, 0u, 0u);
}

static inline uint4 buffer_load_sshort(device const uchar* mem, Win w, uint4 vs, uint index, uint offset, uint soffset, uint idxen) {
    long o = buffer_offset(w, vs, index, offset, soffset, idxen, 2u);
    return uint4(o == OOB_OFF ? 0u : uint(int(*(device const short*)(mem + o))), 0u, 0u, 0u);
}

// Stores skip the image of a loaded module, as buffer stores do.
static inline long flat_store_offset(Win w, ulong a) {
    if (a >= w.ro && a < w.ro_end) return OOB_OFF;
    return goff(w, a);
}

static inline uint4 flat_load_raw(device const uchar* mem, long o) {
    if (o == OOB_OFF) return uint4(0u);
    device const uint* p = (device const uint*)(mem + o);
    return uint4(p[0], p[1], p[2], p[3]);
}

static inline uint4 flat_load_ubyte(device const uchar* mem, long o) {
    return uint4(o == OOB_OFF ? 0u : uint(mem[o]), 0u, 0u, 0u);
}

static inline uint4 flat_load_sbyte(device const uchar* mem, long o) {
    return uint4(o == OOB_OFF ? 0u : uint(int(char(mem[o]))), 0u, 0u, 0u);
}

static inline uint4 flat_load_ushort(device const uchar* mem, long o) {
    return uint4(o == OOB_OFF ? 0u : uint(*(device const ushort*)(mem + o)), 0u, 0u, 0u);
}

static inline uint4 flat_load_sshort(device const uchar* mem, long o) {
    return uint4(o == OOB_OFF ? 0u : uint(int(*(device const short*)(mem + o))), 0u, 0u, 0u);
}

static inline uint atomic_rmw_add(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    return uint(atomic_fetch_add_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_rmw_sub(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    return uint(atomic_fetch_sub_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_rmw_and(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    return uint(atomic_fetch_and_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_rmw_or(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    return uint(atomic_fetch_or_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_rmw_xor(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    return uint(atomic_fetch_xor_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_rmw_umin(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    return uint(atomic_fetch_min_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_rmw_umax(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    return uint(atomic_fetch_max_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_rmw_smin(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    return uint(atomic_fetch_min_explicit((device atomic_int*)(mem + o), int(v), memory_order_relaxed));
}

static inline uint atomic_rmw_smax(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    return uint(atomic_fetch_max_explicit((device atomic_int*)(mem + o), int(v), memory_order_relaxed));
}

static inline uint atomic_rmw_swap(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    return uint(atomic_exchange_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed));
}

static inline uint atomic_cas(device uchar* mem, long o, uint v, uint cmp) {
    if (o == OOB_OFF) return 0u;
    device atomic_uint* p = (device atomic_uint*)(mem + o);
    uint e = cmp;
    while (!atomic_compare_exchange_weak_explicit(p, &e, v, memory_order_relaxed, memory_order_relaxed) && e == cmp) {}
    return e;
}

// Float min and max as the memory atomics do them: denormals flush to signed zero,
// a signalling NaN comes back quieted, a quiet NaN loses to a number, -0 < +0.
static inline uint fminmax_bits(uint a, uint b, bool mx) {
    if ((a & 0x7f800000u) == 0u) a &= 0x80000000u;
    if ((b & 0x7f800000u) == 0u) b &= 0x80000000u;
    bool an = (a & 0x7fffffffu) > 0x7f800000u, bn = (b & 0x7fffffffu) > 0x7f800000u;
    if (an && (a & 0x400000u) == 0u) return a | 0x400000u;
    if (bn && (b & 0x400000u) == 0u) return b | 0x400000u;
    if (an) return b;
    if (bn) return a;
    // Signed-magnitude order as two's complement keys.
    int ka = (a >> 31) != 0u ? int(~(a & 0x7fffffffu)) : int(a);
    int kb = (b >> 31) != 0u ? int(~(b & 0x7fffffffu)) : int(b);
    return (mx ? ka >= kb : ka <= kb) ? a : b;
}

static inline uint atomic_rmw_fmin(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    device atomic_uint* p = (device atomic_uint*)(mem + o);
    uint e = atomic_load_explicit(p, memory_order_relaxed);
    for (;;) {
        uint n = fminmax_bits(e, v, false);
        if (n == e || atomic_compare_exchange_weak_explicit(p, &e, n, memory_order_relaxed, memory_order_relaxed)) return e;
    }
}

static inline uint atomic_rmw_fmax(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    device atomic_uint* p = (device atomic_uint*)(mem + o);
    uint e = atomic_load_explicit(p, memory_order_relaxed);
    for (;;) {
        uint n = fminmax_bits(e, v, true);
        if (n == e || atomic_compare_exchange_weak_explicit(p, &e, n, memory_order_relaxed, memory_order_relaxed)) return e;
    }
}

// Wrapping increment and decrement: past `v` (or below 1) the value starts over.
static inline uint atomic_rmw_inc(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    device atomic_uint* p = (device atomic_uint*)(mem + o);
    uint e = atomic_load_explicit(p, memory_order_relaxed);
    for (;;) {
        uint n = e >= v ? 0u : e + 1u;
        if (n == e || atomic_compare_exchange_weak_explicit(p, &e, n, memory_order_relaxed, memory_order_relaxed)) return e;
    }
}

static inline uint atomic_rmw_dec(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    device atomic_uint* p = (device atomic_uint*)(mem + o);
    uint e = atomic_load_explicit(p, memory_order_relaxed);
    for (;;) {
        uint n = e == 0u || e > v ? v : e - 1u;
        if (n == e || atomic_compare_exchange_weak_explicit(p, &e, n, memory_order_relaxed, memory_order_relaxed)) return e;
    }
}

static inline bool feq_bits(uint a, uint b) {
    return F(a) == F(b);
}

// Float compare-and-swap: +0 and -0 are equal, NaN equals nothing.
static inline uint atomic_fcas(device uchar* mem, long o, uint v, uint cmp) {
    if (o == OOB_OFF) return 0u;
    device atomic_uint* p = (device atomic_uint*)(mem + o);
    uint e = atomic_load_explicit(p, memory_order_relaxed);
    while (feq_bits(e, cmp) && !atomic_compare_exchange_weak_explicit(p, &e, v, memory_order_relaxed, memory_order_relaxed)) {}
    return e;
}

// 64-bit atomics from 32-bit ones (Metal has none wider on every Apple GPU). Adds
// carry into the high half, so concurrent ones leave the exact sum; the others
// read and write the halves apart and can tear when lanes update one place at once.
static inline ulong atomic_add64(device uchar* mem, long o, ulong v) {
    if (o == OOB_OFF) return 0ul;
    device atomic_uint* p = (device atomic_uint*)(mem + o);
    uint lo = atomic_fetch_add_explicit(p, uint(v), memory_order_relaxed);
    uint carry = lo + uint(v) < lo ? 1u : 0u;
    uint hi = atomic_fetch_add_explicit(p + 1, uint(v >> 32) + carry, memory_order_relaxed);
    return ulong(hi) << 32 | lo;
}

static inline ulong atomic_sub64(device uchar* mem, long o, ulong v) {
    if (o == OOB_OFF) return 0ul;
    device atomic_uint* p = (device atomic_uint*)(mem + o);
    uint lo = atomic_fetch_sub_explicit(p, uint(v), memory_order_relaxed);
    uint borrow = lo < uint(v) ? 1u : 0u;
    uint hi = atomic_fetch_sub_explicit(p + 1, uint(v >> 32) + borrow, memory_order_relaxed);
    return ulong(hi) << 32 | lo;
}

static inline bool f64_nan(ulong a) {
    return (a & 0x7ffffffffffffffful) > 0x7ff0000000000000ul;
}

static inline ulong f64_minmax(ulong a, ulong b, bool mx) {
    bool an = f64_nan(a), bn = f64_nan(b);
    ulong quiet = 0x0008000000000000ul;
    if (an && (a & quiet) == 0ul) return a | quiet;
    if (bn && (b & quiet) == 0ul) return b | quiet;
    if (an) return b;
    if (bn) return a;
    long ka = (a >> 63) != 0ul ? long(~(a & 0x7ffffffffffffffful)) : long(a);
    long kb = (b >> 63) != 0ul ? long(~(b & 0x7ffffffffffffffful)) : long(b);
    return (mx ? ka >= kb : ka <= kb) ? a : b;
}

static inline bool f64_eq(ulong a, ulong b) {
    return !f64_nan(a) && !f64_nan(b) && (a == b || ((a | b) << 1) == 0ul);
}

// op: 0 umin, 1 umax, 2 smin, 3 smax, 4 inc, 5 dec, 6 fmin, 7 fmax, 8 cmpswap
// (`c` is the compared value), 9 fcmpswap.
static inline ulong atomic_rmw64(device uchar* mem, long o, ulong v, ulong c, uint op) {
    if (o == OOB_OFF) return 0ul;
    device atomic_uint* p = (device atomic_uint*)(mem + o);
    uint lo = atomic_load_explicit(p, memory_order_relaxed);
    uint hi = atomic_load_explicit(p + 1, memory_order_relaxed);
    ulong old = ulong(hi) << 32 | lo, n;
    switch (op) {
    case 0u: n = min(old, v); break;
    case 1u: n = max(old, v); break;
    case 2u: n = ulong(min(long(old), long(v))); break;
    case 3u: n = ulong(max(long(old), long(v))); break;
    case 4u: n = old >= v ? 0ul : old + 1ul; break;
    case 5u: n = old == 0ul || old > v ? v : old - 1ul; break;
    case 6u: n = f64_minmax(old, v, false); break;
    case 7u: n = f64_minmax(old, v, true); break;
    case 8u: n = old == c ? v : old; break;
    default: n = f64_eq(old, c) ? v : old; break;
    }
    if (n != old) {
        atomic_store_explicit(p, uint(n), memory_order_relaxed);
        atomic_store_explicit(p + 1, uint(n >> 32), memory_order_relaxed);
    }
    return old;
}

static inline float cube_ma(float x, float y, float z) {
    if (abs(z) >= abs(x) && abs(z) >= abs(y)) return 2.0f * z;
    if (abs(y) >= abs(x)) return 2.0f * y;
    return 2.0f * x;
}

static inline float cube_id(float x, float y, float z) {
    if (abs(z) >= abs(x) && abs(z) >= abs(y)) return z < 0.0f ? 5.0f : 4.0f;
    if (abs(y) >= abs(x)) return y < 0.0f ? 3.0f : 2.0f;
    return x < 0.0f ? 1.0f : 0.0f;
}

static inline float cube_sc(float x, float y, float z) {
    if (abs(z) >= abs(x) && abs(z) >= abs(y)) return z < 0.0f ? -x : x;
    if (abs(y) >= abs(x)) return x;
    return x < 0.0f ? z : -z;
}

static inline float cube_tc(float x, float y, float z) {
    if (abs(z) >= abs(x) && abs(z) >= abs(y)) return -y;
    if (abs(y) >= abs(x)) return y < 0.0f ? -z : z;
    return -y;
}

// The direction for cube coordinates as image_sample takes them: face
// coordinates in [1, 2] (v_cubesc / |ma| + 1.5) and the face id.
static inline float3 cube_dir(float3 c) {
    float s = (clamp(c.x, 1.0f, 2.0f) - 1.5f) * 2.0f;
    float t = (clamp(c.y, 1.0f, 2.0f) - 1.5f) * 2.0f;
    switch (uint(clamp(floor(c.z), 0.0f, 5.0f))) {
    case 0u: return float3(1.0f, -t, -s);
    case 1u: return float3(-1.0f, -t, s);
    case 2u: return float3(s, 1.0f, t);
    case 3u: return float3(s, -1.0f, -t);
    case 4u: return float3(s, -t, 1.0f);
    default: return float3(-s, -t, -1.0f);
    }
}

// How many channels GFX10 image format `f` stores.
static inline uint image_channels(uint f) {
    if ((f >= 1u && f <= 13u) || (f >= 20u && f <= 22u) || f == 128u || f == 131u
        || f == 140u || f == 175u || f == 176u || f == 285u) return 1u;
    if ((f >= 14u && f <= 19u) || (f >= 23u && f <= 29u) || (f >= 62u && f <= 64u)
        || f == 129u || f == 137u || (f >= 141u && f <= 146u) || f == 177u || f == 178u) return 2u;
    if ((f >= 30u && f <= 43u) || (f >= 72u && f <= 74u) || f == 132u || f == 133u
        || f == 179u || f == 180u) return 3u;
    return 4u;
}

// A T#'s DST_SEL (dwords 1 and 3) on a fetched texel. Selecting a channel
// the format lacks gives channel c mod n (measured on RDNA2).
template <typename T>
static inline vec<T, 4> image_sel(vec<T, 4> v, uint w1, uint w3) {
    uint n = image_channels((w1 >> 20) & 0x1ffu);
    vec<T, 4> r;
    for (uint i = 0u; i < 4u; i++) {
        uint s = (w3 >> (3u * i)) & 7u;
        r[i] = s >= 4u ? v[(s - 4u) % n] : T(s == 1u ? 1 : 0);
    }
    return r;
}

// A texel read raw from an SRGB image (format in T# dword `w1`) as the
// hardware returns it: colour channels decoded and rounded to bfloat16
// (measured on RDNA2).
static inline float4 image_srgb(float4 v, uint w1) {
    uint f = (w1 >> 20) & 0x1ffu;
    if (!(f >= 128u && f <= 130u) && f != 170u && f != 172u && f != 174u && f != 182u) return v;
    for (uint i = 0u; i < min(image_channels(f), 3u); i++) {
        float c = v[i];
        c = c <= 0.04045f ? c / 12.92f : precise::pow((c + 0.055f) / 1.055f, 2.4f);
        uint b = as_type<uint>(c);
        v[i] = as_type<float>((b + 0x7fffu + ((b >> 16) & 1u)) & 0xffff0000u);
    }
    return v;
}

// The channels a store writes into an image of `n` channels: a format of
// two takes the first and the last component (measured on RDNA2).
template <typename T>
static inline vec<T, 4> image_store_channels(vec<T, 4> v, uint w1) {
    if (image_channels((w1 >> 20) & 0x1ffu) == 2u) v.y = v.w;
    return v;
}

// How a GFX10 image format lays out its channels: their widths from the
// lowest bits up and their number type (0 UNORM, 1 SNORM, 2 USCALED,
// 3 SSCALED, 4 UINT, 5 SINT, 6 FLOAT, 7 SRGB, 8 shared exponent).
struct ImageLayout {
    uint4 width;
    uint type;
};

static inline ImageLayout image_layout(uint f) {
    ImageLayout l;
    l.type = 4u;
    l.width = uint4(0u);
    if (f >= 1u && f <= 6u) { l.width = uint4(8u, 0u, 0u, 0u); l.type = f - 1u; }
    else if (f >= 7u && f <= 13u) { l.width = uint4(16u, 0u, 0u, 0u); l.type = f - 7u; }
    else if (f >= 14u && f <= 19u) { l.width = uint4(8u, 8u, 0u, 0u); l.type = f - 14u; }
    else if (f >= 20u && f <= 22u) { l.width = uint4(32u, 0u, 0u, 0u); l.type = f - 16u; }
    else if (f >= 23u && f <= 29u) { l.width = uint4(16u, 16u, 0u, 0u); l.type = f - 23u; }
    else if (f >= 30u && f <= 36u) { l.width = uint4(11u, 11u, 10u, 0u); l.type = f - 30u; }
    else if (f >= 37u && f <= 43u) { l.width = uint4(10u, 11u, 11u, 0u); l.type = f - 37u; }
    else if (f >= 44u && f <= 49u) { l.width = uint4(2u, 10u, 10u, 10u); l.type = f - 44u; }
    else if (f >= 50u && f <= 55u) { l.width = uint4(10u, 10u, 10u, 2u); l.type = f - 50u; }
    else if (f >= 56u && f <= 61u) { l.width = uint4(8u); l.type = f - 56u; }
    else if (f >= 62u && f <= 64u) { l.width = uint4(32u, 32u, 0u, 0u); l.type = f - 58u; }
    else if (f >= 65u && f <= 71u) { l.width = uint4(16u); l.type = f - 65u; }
    else if (f >= 75u && f <= 77u) { l.width = uint4(32u); l.type = f - 71u; }
    else if (f >= 128u && f <= 130u) { l.width = f == 128u ? uint4(8u, 0u, 0u, 0u) : f == 129u ? uint4(8u, 8u, 0u, 0u) : uint4(8u); l.type = 7u; }
    else if (f == 132u) { l.width = uint4(9u, 9u, 9u, 5u); l.type = 8u; }
    else if (f == 133u) { l.width = uint4(5u, 6u, 5u, 0u); l.type = 0u; }
    else if (f == 135u) { l.width = uint4(1u, 5u, 5u, 5u); l.type = 0u; }
    else if (f == 136u) { l.width = uint4(4u); l.type = 0u; }
    else if (f == 137u) { l.width = uint4(4u, 4u, 0u, 0u); l.type = 0u; }
    else if (f == 140u) { l.width = uint4(32u, 0u, 0u, 0u); l.type = 6u; }
    else if (f == 141u || f == 142u) { l.width = uint4(24u, 8u, 0u, 0u); l.type = f == 141u ? 0u : 4u; }
    else if (f == 143u || f == 144u) { l.width = uint4(8u, 24u, 0u, 0u); l.type = f == 143u ? 0u : 4u; }
    else if (f == 145u || f == 146u) { l.width = uint4(32u, 8u, 0u, 0u); l.type = f == 145u ? 4u : 6u; }
    return l;
}

static inline uint image_mask(uint w) {
    return w >= 32u ? 0xffffffffu : (1u << w) - 1u;
}

static inline int image_sext(uint u, uint w) {
    return w >= 32u ? int(u) : int(u << (32u - w)) >> (32u - w);
}

// Half float bits as float bits, NaN payloads kept.
static inline uint image_from_half(uint h) {
    uint s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    if (e == 31u) return s | 0x7f800000u | m << 13;
    if (e != 0u) return s | (e + 112u) << 23 | m << 13;
    if (m == 0u) return s;
    uint k = 31u - clz(m);
    return s | (k + 103u) << 23 | ((m << (23u - k)) & 0x7fffffu);
}

// Float bits as half float bits, rounded toward zero as image stores do:
// finite values past the largest half become 65504. A NaN keeps its sign
// and top payload bits, not quieted; one whose kept payload is 0 becomes
// quiet (assumed). Measured on RDNA2.
static inline uint image_to_half(uint b) {
    uint s = (b >> 16) & 0x8000u, e = (b >> 23) & 0xffu, m = b & 0x7fffffu;
    if (e == 255u) return s | 0x7c00u | (m != 0u && (m >> 13) == 0u ? 0x200u : m >> 13);
    if (e >= 143u) return s | 0x7bffu;
    if (e >= 113u) return s | (e - 112u) << 10 | m >> 13;
    if (e >= 103u) return s | (m | 0x800000u) >> (126u - e);
    return s;
}

// An unsigned float of `m` mantissa bits and a 5-bit exponent (the 10- and
// 11-bit channels), as float bits.
static inline uint image_small_float(uint u, uint m) {
    uint e = u >> m, f = u & ((1u << m) - 1u);
    if (e == 31u) return 0x7f800000u | f << (23u - m);
    if (e == 0u) return as_type<uint>(float(f) * exp2(-14.0f - float(m)));
    return (e + 112u) << 23 | f << (23u - m);
}

// A texel read raw from an image of a format Metal lacks (T# dwords `w1`
// and `w3`), converted and its channels selected as the hardware does.
static inline uint4 image_decode(uint4 raw, uint w1, uint w3) {
    uint f = (w1 >> 20) & 0x1ffu;
    ImageLayout l = image_layout(f);
    uint4 c = uint4(0u);
    uint at = 0u, n = 0u;
    for (uint i = 0u; i < 4u; i++) {
        uint w = l.width[i];
        if (w == 0u) break;
        uint u = (raw[at / 32u] >> (at % 32u)) & image_mask(w);
        at += w;
        n++;
        switch (l.type) {
        case 0u: c[i] = as_type<uint>(precise::divide(float(u), float(image_mask(w)))); break;
        case 1u: c[i] = as_type<uint>(max(precise::divide(float(image_sext(u, w)), float(image_mask(w - 1u))), -1.0f)); break;
        case 2u: c[i] = as_type<uint>(float(u)); break;
        case 3u: c[i] = as_type<uint>(float(image_sext(u, w))); break;
        case 5u: c[i] = uint(image_sext(u, w)); break;
        case 6u:
            c[i] = w == 32u ? u : w == 16u ? image_from_half(u) : image_small_float(u, w - 5u);
            break;
        case 7u: {
            float x = precise::divide(float(u), float(image_mask(w)));
            if (i < 3u) {
                x = x <= 0.04045f ? x / 12.92f : precise::pow((x + 0.055f) / 1.055f, 2.4f);
                uint b = as_type<uint>(x);
                x = as_type<float>((b + 0x7fffu + ((b >> 16) & 1u)) & 0xffff0000u);
            }
            c[i] = as_type<uint>(x);
            break;
        }
        default: c[i] = u; break;
        }
    }
    if (l.type == 8u) {
        float scale = exp2(float(int(c[3]) - 24));
        c = uint4(as_type<uint3>(float3(c.xyz) * scale), 0u);
        n = 3u;
    }
    if (n == 0u) n = 1u;
    uint one = l.type == 4u || l.type == 5u ? 1u : 0x3f800000u;
    uint4 r;
    for (uint i = 0u; i < 4u; i++) {
        uint s = (w3 >> (3u * i)) & 7u;
        r[i] = s >= 4u ? c[(s - 4u) % n] : (s == 1u ? one : 0u);
    }
    return r;
}

// An unsigned float of `m` mantissa bits and a 5-bit exponent from float
// bits `b`.
static inline uint image_to_small_float(uint b, uint m) {
    float x = as_type<float>(b);
    if (isnan(x)) return 31u << m | (b >> (23u - m) & ((1u << m) - 1u)) | 1u;
    if (x <= 0.0f) return 0u;
    if (isinf(x)) return 31u << m;
    uint e = (b >> 23) & 0xffu;
    if (e < 113u) return uint(rint(x * exp2(14.0f + float(m))));
    uint v = ((e - 112u) << m) + ((b & 0x7fffffu) >> (23u - m));
    uint rest = b & ((1u << (23u - m)) - 1u), half_way = 1u << (22u - m);
    if (rest > half_way || (rest == half_way && (v & 1u) != 0u)) v++;
    return min(v, (31u << m) - 1u);
}

// The bits an element of format `f` takes.
static inline uint image_element_bits(uint f) {
    ImageLayout l = image_layout(f);
    return l.width.x + l.width.y + l.width.z + l.width.w;
}

// An element read as bits (`image_load_pck`): its dwords, an element of 8
// or 16 bits zero- or (`sgn`) sign-extended (measured on RDNA2).
static inline uint4 image_unpack(uint4 raw, uint w1, bool sgn) {
    uint bits = image_element_bits((w1 >> 20) & 0x1ffu);
    if (bits < 32u) raw.x = sgn ? uint(image_sext(raw.x, bits)) : raw.x & image_mask(bits);
    for (uint i = (bits + 31u) / 32u; i < 4u; i++) raw[i] = 0u;
    return raw;
}

// An element written as bits (`image_store_pck`) from store data dwords.
static inline uint4 image_pack(uint4 v, uint w1) {
    uint bits = image_element_bits((w1 >> 20) & 0x1ffu);
    if (bits < 32u) v.x &= image_mask(bits);
    return v;
}

// Whether image stores write an image of format `f`: not the depth and
// stencil, 4_4 and 96-bit formats (measured on RDNA2).
static inline bool image_storable(uint f) {
    return !(f >= 72u && f <= 74u) && f != 137u && !(f >= 141u && f <= 146u);
}

// Store data (VGPR bits) for an image of a format Metal lacks, converted
// into the bits of one element. SCALED formats are written as NORM
// (measured on RDNA2).
static inline uint4 image_encode(uint4 v, uint w1) {
    uint f = (w1 >> 20) & 0x1ffu;
    ImageLayout l = image_layout(f);
    uint n = 0u;
    for (uint i = 0u; i < 4u; i++) n += l.width[i] != 0u ? 1u : 0u;
    if (n == 2u) v.y = v.w;
    uint4 raw = uint4(0u);
    uint at = 0u;
    if (l.type == 8u) {
        float3 x = clamp(as_type<float3>(v.xyz), 0.0f, 65408.0f);
        x = select(x, float3(0.0f), isnan(x));
        float top = max(max(x.x, x.y), x.z);
        int e = max(-16, int(floor(log2(top)))) + 16;
        float scale = exp2(float(e - 24));
        if (uint(rint(top / scale)) == 512u) { e++; scale *= 2.0f; }
        uint3 m = uint3(rint(x / scale));
        raw.x = m.x | m.y << 9 | m.z << 18 | uint(e) << 27;
        return raw;
    }
    for (uint i = 0u; i < n; i++) {
        uint w = l.width[i], u;
        float x = as_type<float>(v[i]);
        float top = float(image_mask(w));
        switch (l.type) {
        case 0u: case 2u: case 7u: u = isnan(x) ? 0u : uint(rint(clamp(x, 0.0f, 1.0f) * top)); break;
        case 1u: case 3u: {
            float h = float(image_mask(w - 1u));
            u = isnan(x) ? 0u : uint(int(rint(clamp(x, -1.0f, 1.0f) * h)));
            break;
        }
        case 4u: u = min(v[i], image_mask(w)); break;
        case 5u: {
            int h = int(image_mask(w - 1u));
            u = uint(clamp(int(v[i]), -h - 1, h));
            break;
        }
        default:
            u = w == 32u ? v[i] : w == 16u ? image_to_half(v[i]) : image_to_small_float(v[i], w - 5u);
            break;
        }
        raw[at / 32u] |= (u & image_mask(w)) << (at % 32u);
        at += w;
    }
    return raw;
}

// An S#'s LOD_BIAS (signed 6.8 fixed point in dword 2).
static inline float lod_bias(uint w2) {
    return float(int(w2 << 18) >> 18) / 256.0f;
}

// An explicit LOD as the S# biases and clamps it (MIN_LOD and MAX_LOD,
// unsigned 4.8 in dword 1).
static inline float sampler_lod(uint w1, uint w2, float l) {
    return clamp(l + lod_bias(w2), float(w1 & 0xFFFu) / 256.0f, float((w1 >> 12) & 0xFFFu) / 256.0f);
}

// The factor that makes 2D coordinates normalized: 1 / size of the base
// level when the S# sets FORCE_UNNORMALIZED.
template <typename T>
static inline float2 coord_scale(uint w0, T t) {
    return (w0 & 0x8000u) != 0u ? 1.0f / float2(t.get_width(0), t.get_height(0)) : float2(1.0f);
}

// The LOD RDNA2 takes from derivatives in texels (measured with
// image_get_lod): log2 of the Jacobian's larger singular value in 1D and
// 2D, of the longer gradient in 3D, in steps of 1/256; zero derivatives
// give -120.50390625.
static inline float hw_lod(float squared) {
    return max(rint(0.5f * log2(squared) * 256.0f) / 256.0f, -120.50390625f);
}

static inline float lod_2d(float2 dx, float2 dy) {
    float s = dot(dx, dx) + dot(dy, dy);
    float det = dx.x * dy.y - dy.x * dx.y;
    return hw_lod(0.5f * (s + sqrt(max(s * s - 4.0f * det * det, 0.0f))));
}

static inline float lod_3d(float3 dx, float3 dy) {
    return hw_lod(max(dot(dx, dx), dot(dy, dy)));
}

// The level a gather reads at `lod`: the nearest one, as a sample with a
// point mip filter picks it (assumed).
static inline uint gather_level(float lod, uint levels) {
    return uint(clamp(ceil(lod + 0.5f) - 1.0f, 0.0f, float(levels - 1u)));
}

// Texel `i` of a row of `n` under SQ_TEX_CLAMP mode `m`, -1 for the border.
static inline int tex_wrap(int i, int n, uint m) {
    switch (m) {
    case 0u: return ((i % n) + n) % n;
    case 1u: {
        int p = ((i % (2 * n)) + 2 * n) % (2 * n);
        return p < n ? p : 2 * n - 1 - p;
    }
    case 3u: case 5u: case 7u: return clamp(i < 0 ? -1 - i : i, 0, n - 1);
    case 6u: return i < 0 || i >= n ? -1 : i;
    default: return clamp(i, 0, n - 1);
    }
}

// The texels a gather reads at normalized `uv` on a level of `size`, in the
// ISA's order: the bilinear footprint for image_gather4 (positions keep 8
// fraction bits), and for image_gather4h (`row`) the four texels of the row
// around it, clamped to the edge under any address mode (measured).
static inline void gather_texels(float2 uv, int2 size, int2 off, uint w0, bool row,
                                 thread int4& xs, thread int4& ys) {
    float2 p = floor(rint((uv * float2(size) - float2(0.5f, row ? 0.0f : 0.5f)) * 256.0f) / 256.0f);
    int2 b = int2(p) + off;
    if (row) {
        xs = clamp(int4(b.x - 1, b.x, b.x + 1, b.x + 2), 0, size.x - 1);
        ys = int4(clamp(b.y, 0, size.y - 1));
        return;
    }
    xs = int4(b.x, b.x + 1, b.x + 1, b.x);
    ys = int4(b.y + 1, b.y + 1, b.y, b.y);
    for (int i = 0; i < 4; i++) {
        xs[i] = tex_wrap(xs[i], size.x, w0 & 7u);
        ys[i] = tex_wrap(ys[i], size.y, (w0 >> 3) & 7u);
    }
}

// Component `c` of an S#'s BORDER_COLOR_TYPE (register colours read as
// transparent black).
static inline float border_component(uint w3, uint c) {
    uint b = w3 >> 30;
    return b == 2u || (b == 1u && c == 3u) ? 1.0f : 0.0f;
}

template <typename E>
static inline vec<E, 4> gather_read(texture2d<E> t, uint l, int4 xs, int4 ys, uint w3, uint c) {
    vec<E, 4> r;
    for (int i = 0; i < 4; i++) {
        r[i] = xs[i] < 0 || ys[i] < 0 ? E(border_component(w3, c)) : t.read(uint2(xs[i], ys[i]), l)[c];
    }
    return r;
}

template <typename E>
static inline vec<E, 4> gather_read(texture2d_array<E> t, uint slice, uint l, int4 xs, int4 ys, uint w3, uint c) {
    vec<E, 4> r;
    for (int i = 0; i < 4; i++) {
        r[i] = xs[i] < 0 || ys[i] < 0 ? E(border_component(w3, c)) : t.read(uint2(xs[i], ys[i]), slice, l)[c];
    }
    return r;
}

// SQ_TEX_DEPTH_COMPARE: the reference against the fetched value.
static inline bool depth_compare(uint f, float ref, float v) {
    switch (f) {
    case 1u: return ref < v;
    case 2u: return ref == v;
    case 3u: return ref <= v;
    case 4u: return ref > v;
    case 5u: return ref != v;
    case 6u: return ref >= v;
    case 7u: return true;
    default: return false;
    }
}

static inline uint fmt_sel(uint sel, uint4 v, uint one) {
    switch (sel) {
    case 1u: return one;
    case 4u: return v.x;
    case 5u: return v.y;
    case 6u: return v.z;
    case 7u: return v.w;
    default: return 0u;
    }
}

// A buffer format (GFX10 unified encoding, BUF_FMT_*): its component count (0 for
// formats not handled), each component's bits from bit 0 up, and its number kind:
// 0 UNORM, 1 SNORM, 2 USCALED, 3 SSCALED, 4 UINT, 5 SINT, 6 FLOAT. Format names list
// components from the top bit down: 2_10_10_10 has x in its low 10 bits.
struct BufFmt { uint n; uint4 bits; uint kind; };

static inline BufFmt buffer_fmt(uint fmt) {
    if (fmt >= 1u && fmt <= 6u) return { 1u, uint4(8u, 0u, 0u, 0u), fmt - 1u };
    if (fmt >= 7u && fmt <= 13u) return { 1u, uint4(16u, 0u, 0u, 0u), fmt - 7u };
    if (fmt >= 14u && fmt <= 19u) return { 2u, uint4(8u, 8u, 0u, 0u), fmt - 14u };
    if (fmt >= 20u && fmt <= 22u) return { 1u, uint4(32u, 0u, 0u, 0u), fmt - 16u };
    if (fmt >= 23u && fmt <= 29u) return { 2u, uint4(16u, 16u, 0u, 0u), fmt - 23u };
    if (fmt >= 30u && fmt <= 36u) return { 3u, uint4(11u, 11u, 10u, 0u), fmt - 30u };
    if (fmt >= 37u && fmt <= 43u) return { 3u, uint4(10u, 11u, 11u, 0u), fmt - 37u };
    if (fmt >= 44u && fmt <= 49u) return { 4u, uint4(2u, 10u, 10u, 10u), fmt - 44u };
    if (fmt >= 50u && fmt <= 55u) return { 4u, uint4(10u, 10u, 10u, 2u), fmt - 50u };
    if (fmt >= 56u && fmt <= 61u) return { 4u, uint4(8u), fmt - 56u };
    if (fmt >= 62u && fmt <= 64u) return { 2u, uint4(32u, 32u, 0u, 0u), fmt - 58u };
    if (fmt >= 65u && fmt <= 71u) return { 4u, uint4(16u), fmt - 65u };
    if (fmt >= 72u && fmt <= 74u) return { 3u, uint4(32u, 32u, 32u, 0u), fmt - 68u };
    if (fmt >= 75u && fmt <= 77u) return { 4u, uint4(32u), fmt - 71u };
    return { 0u, uint4(0u), 0u };
}

static inline uint fmt_bytes(BufFmt f) {
    return (f.bits.x + f.bits.y + f.bits.z + f.bits.w) / 8u;
}

// An unsigned float of 5 exponent bits and `mb` mantissa bits as an f32.
static inline uint small_float(uint v, uint mb) {
    uint e = v >> mb, m = v & ((1u << mb) - 1u);
    if (e == 31u) return 0x7f800000u | (m << (23u - mb));
    if (e == 0u) return U(float(m) * exp2(-14.0f - float(mb)));
    return ((e + 112u) << 23) | (m << (23u - mb));
}

// An f32 as an unsigned float of 5 exponent bits and `mb` mantissa bits, rounded
// toward zero like the 16-bit stores: negatives go to zero, overflow to the largest
// finite value, NaN keeps the top of its payload.
static inline uint to_small_float(float f, uint mb) {
    uint mask = (1u << mb) - 1u;
    uint b = as_type<uint>(f);
    if (isnan(f)) return (31u << mb) | ((b & 0x7fffffu) >> (23u - mb));
    if (!(f > 0.0f)) return 0u;
    if (isinf(f)) return 31u << mb;
    int e = int((b >> 23) & 0xffu) - 112;
    uint m = b & 0x7fffffu;
    if (e >= 31) return (30u << mb) | mask;
    if (e <= 0) return 1 - e > 24 ? 0u : ((m | 0x800000u) >> uint(1 - e)) >> (23u - mb);
    return (uint(e) << mb) | (m >> (23u - mb));
}

// One component of `bits` bits converted to its result dword as `kind` says.
// Halves to and from f32 keeping NaN payloads, which Metal's conversions drop;
// stores round toward zero.
static inline uint half_to_f32(uint h) {
    if ((h & 0x7c00u) == 0x7c00u && (h & 0x3ffu) != 0u) return (h & 0x8000u) << 16 | 0x7f800000u | (h & 0x3ffu) << 13;
    return as_type<uint>(float(as_type<half>(ushort(h))));
}

static inline uint f32_to_half_rtz(uint v) {
    if ((v & 0x7fffffffu) > 0x7f800000u) return (v >> 16 & 0x8000u) | 0x7c00u | (v & 0x7fffffu) >> 13;
    return uint(as_type<ushort>(half_rtz(as_type<float>(v))));
}

static inline uint fmt_conv(uint v, uint bits, uint kind) {
    uint m = bits == 32u ? 0xffffffffu : (1u << bits) - 1u;
    v &= m;
    int s = bits == 32u ? int(v) : int(v << (32u - bits)) >> (32u - bits);
    switch (kind) {
    case 0u: return as_type<uint>(float(v) / float(m));
    case 1u: return as_type<uint>(max(float(s) / float(m >> 1), -1.0f));
    case 2u: return as_type<uint>(float(v));
    case 3u: return as_type<uint>(float(s));
    case 4u: return v;
    case 5u: return uint(s);
    default:
        if (bits == 16u) return half_to_f32(v);
        if (bits == 11u) return small_float(v, 6u);
        if (bits == 10u) return small_float(v, 5u);
        return v;
    }
}

// A result dword packed into a `bits`-bit component of `kind`: NaN stores zero in
// the fixed-point kinds, integers saturate, floats round toward zero.
static inline uint fmt_pack(uint v, uint bits, uint kind) {
    uint m = bits == 32u ? 0xffffffffu : (1u << bits) - 1u;
    float f = as_type<float>(v);
    int hi = int(m >> 1), lo = -hi - 1;
    switch (kind) {
    case 0u: return isnan(f) ? 0u : uint(rint(clamp(f, 0.0f, 1.0f) * float(m))) & m;
    case 1u: return isnan(f) ? 0u : uint(int(rint(clamp(f, -1.0f, 1.0f) * float(hi)))) & m;
    case 2u: return isnan(f) ? 0u : uint(clamp(f, 0.0f, float(m))) & m;
    case 3u: return isnan(f) ? 0u : uint(int(clamp(f, float(lo), float(hi)))) & m;
    case 4u: return min(v, m);
    case 5u: return bits == 32u ? v : uint(clamp(int(v), lo, hi)) & m;
    default:
        if (bits == 16u) return f32_to_half_rtz(v);
        if (bits == 11u) return to_small_float(f, 6u);
        if (bits == 10u) return to_small_float(f, 5u);
        return v;
    }
}

static inline uint ld_le(device const uchar* p, uint bytes) {
    uint v = 0u;
    for (uint i = 0u; i < bytes; i++) v |= uint(p[i]) << (8u * i);
    return v;
}

static inline void st_le(device uchar* p, uint bytes, uint v) {
    for (uint i = 0u; i < bytes; i++) p[i] = uchar(v >> (8u * i));
}

// Typed loads. MUBUF ones fill the components a format lacks with the ones it has
// (x for one, x and y for two, x for the fourth of three), and then DST_SEL picks the
// results. MTBUF ones (`sel` 0) ignore DST_SEL and read zero past the format's
// components. Both measured on gfx1033.
static inline uint4 buffer_load_format(device const uchar* mem, Win w, uint4 vs, uint index, uint offset, uint soffset, uint idxen, uint sel) {
    BufFmt f = buffer_fmt((vs.w >> 12) & 0x7fu);
    if (f.n == 0u) return uint4(0u);
    uint bytes = fmt_bytes(f);
    long o = buffer_offset(w, vs, index, offset, soffset, idxen, bytes);
    uint one = f.kind == 4u || f.kind == 5u ? 1u : 0x3f800000u;
    uint4 r = uint4(0u);
    if (o != OOB_OFF) {
        uint4 raw;
        for (uint k = 0u; k < 4u; k++) raw[k] = 4u * k < bytes ? ld_le(mem + o + 4 * k, min(4u, bytes - 4u * k)) : 0u;
        uint4 c;
        uint pos = 0u;
        for (uint i = 0u; i < f.n; i++) {
            uint b = f.bits[i];
            c[i] = fmt_conv(raw[pos / 32u] >> (pos % 32u), b, f.kind);
            pos += b;
        }
        for (uint i = 0u; i < 4u; i++) r[i] = sel != 0u ? c[i % f.n] : i < f.n ? c[i] : 0u;
    }
    if (sel == 0u) return r;
    return uint4(fmt_sel(vs.w & 7u, r, one), fmt_sel((vs.w >> 3) & 7u, r, one),
                 fmt_sel((vs.w >> 6) & 7u, r, one), fmt_sel((vs.w >> 9) & 7u, r, one));
}

// Typed stores write the whole element. Data components past `n` are zero. In
// MUBUF ones memory component c takes the data DST_SEL c picks, except that the
// second component of a two-component format takes DST_SEL_W's; a constant select
// stores one for SEL_0 and zero for SEL_1. MTBUF ones (`sel` 0) store the data in
// order. Measured on gfx1033 (constants on one 32_32_32_32_FLOAT descriptor only).
static inline void buffer_store_format(device uchar* mem, Win w, uint4 vs, uint index, uint offset, uint soffset, uint idxen, uint4 d, uint n, uint sel) {
    BufFmt f = buffer_fmt((vs.w >> 12) & 0x7fu);
    if (f.n == 0u) return;
    uint bytes = fmt_bytes(f);
    long o = buffer_store_offset(w, vs, index, offset, soffset, idxen, bytes);
    if (o == OOB_OFF) return;
    for (uint i = n; i < 4u; i++) d[i] = 0u;
    uint one = f.kind == 4u || f.kind == 5u ? 1u : 0x3f800000u;
    uint4 raw = uint4(0u);
    uint pos = 0u;
    for (uint i = 0u; i < f.n; i++) {
        uint slot = f.n == 2u && i == 1u ? 3u : i;
        uint s = sel != 0u ? (vs.w >> (3u * slot)) & 7u : 4u + i;
        uint v = s >= 4u ? d[s - 4u] : s == 0u ? one : 0u;
        uint b = f.bits[i];
        raw[pos / 32u] |= fmt_pack(v, b, f.kind) << (pos % 32u);
        pos += b;
    }
    for (uint k = 0u; 4u * k < bytes; k++) st_le(mem + o + 4 * k, min(4u, bytes - 4u * k), raw[k]);
}

// D16 typed loads: results as 16 bits. Normalized and scaled values round to the
// nearest half, float ones toward zero; integers saturate (measured on gfx1033).
static inline uint4 buffer_load_format_d16(device const uchar* mem, Win w, uint4 vs, uint index, uint offset, uint soffset, uint idxen, uint sel) {
    uint4 r = buffer_load_format(mem, w, vs, index, offset, soffset, idxen, sel);
    uint kind = buffer_fmt((vs.w >> 12) & 0x7fu).kind;
    for (uint i = 0u; i < 4u; i++) {
        r[i] = kind == 4u ? min(r[i], 0xffffu)
            : kind == 5u ? uint(clamp(int(r[i]), -32768, 32767)) & 0xffffu
            : kind == 6u ? f32_to_half_rtz(r[i])
            : uint(as_type<ushort>(half(F(r[i]))));
    }
    return r;
}

// D16 typed stores: 16-bit data widened as the format's kind reads it.
static inline void buffer_store_format_d16(device uchar* mem, Win w, uint4 vs, uint index, uint offset, uint soffset, uint idxen, uint4 h, uint n, uint sel) {
    uint kind = buffer_fmt((vs.w >> 12) & 0x7fu).kind;
    uint4 d;
    for (uint i = 0u; i < 4u; i++) {
        d[i] = kind == 4u ? h[i] : kind == 5u ? uint(int(short(ushort(h[i])))) : half_to_f32(h[i]);
    }
    buffer_store_format(mem, w, vs, index, offset, soffset, idxen, d, n, sel);
}

"#;

// Lockstep programs: a wave64 group run as two SIMD groups that meet at every
// call below, so lane masks and lane reads span the whole wave.
pub(super) const WAVE_PRELUDE: &str = r#"
// A wave64 is two Metal SIMD groups, which meet to exchange values. In a group of one wave
// a barrier does; in a bigger one each SIMD group counts its meetings in `seq` and waits
// for its partner to get as far, so waves stay independent between `s_barrier`s.
struct Wave {
    threadgroup uint* vote;
    threadgroup atomic_uint* seq;
    ushort sgi;
    uint k;
};

static inline void wave_meet(thread Wave& w) {
    w.k++;
#if LOCKSTEP_WAVES == 1
    threadgroup_barrier(mem_flags::mem_threadgroup);
#else
    atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
    if (simd_is_first()) atomic_store_explicit(&w.seq[w.sgi], w.k, memory_order_relaxed);
    for (uint spin = 0u; spin < MEET_SPINS && atomic_load_explicit(&w.seq[w.sgi ^ 1u], memory_order_relaxed) < w.k; spin++) {}
    atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
#endif
}

// Each exchange uses the other half of the wave's `vote` slots, so one meeting per exchange is enough.
static inline threadgroup uint* wave_slots(thread Wave& w) {
    return w.vote + 4u * (w.sgi >> 1) + 2u * (w.k & 1u);
}

static inline bool wave_any(bool x, thread Wave& w) {
    threadgroup uint* v = wave_slots(w);
    bool a = simd_any(x);
    if (simd_is_first()) v[w.sgi & 1] = a ? 1u : 0u;
    wave_meet(w);
    return (v[0] | v[1]) != 0u;
}

static inline ulong wave_ballot(bool x, thread Wave& w) {
    threadgroup uint* v = wave_slots(w);
    uint m = uint(static_cast<simd_vote::vote_t>(simd_ballot(x)));
    if (simd_is_first()) v[w.sgi & 1] = m;
    wave_meet(w);
    return ulong(v[1]) << 32 | ulong(v[0]);
}

static inline uint wave_readlane(uint x, uint lane, thread Wave& w) {
    threadgroup uint* v = wave_slots(w);
    uint r = simd_shuffle(x, ushort(lane & 31u));
    if ((w.sgi & 1) == ushort(lane >> 5 & 1u) && simd_is_first()) v[0] = r;
    wave_meet(w);
    return v[0];
}

static inline bool lane_on(uint lo, uint hi, uint wl) {
    return ((wl < 32u ? lo : hi) >> (wl & 31u) & 1u) != 0u;
}
"#;

pub(super) const LDS_PRELUDE: &str = r#"
static inline uint lds_rmw_add(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    return uint(atomic_fetch_add_explicit((threadgroup atomic_uint*)&lds[a >> 2], v, memory_order_relaxed));
}

static inline uint lds_rmw_sub(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    return uint(atomic_fetch_sub_explicit((threadgroup atomic_uint*)&lds[a >> 2], v, memory_order_relaxed));
}

static inline uint lds_rmw_and(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    return uint(atomic_fetch_and_explicit((threadgroup atomic_uint*)&lds[a >> 2], v, memory_order_relaxed));
}

static inline uint lds_rmw_or(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    return uint(atomic_fetch_or_explicit((threadgroup atomic_uint*)&lds[a >> 2], v, memory_order_relaxed));
}

static inline uint lds_rmw_xor(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    return uint(atomic_fetch_xor_explicit((threadgroup atomic_uint*)&lds[a >> 2], v, memory_order_relaxed));
}

static inline uint lds_rmw_umin(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    return uint(atomic_fetch_min_explicit((threadgroup atomic_uint*)&lds[a >> 2], v, memory_order_relaxed));
}

static inline uint lds_rmw_umax(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    return uint(atomic_fetch_max_explicit((threadgroup atomic_uint*)&lds[a >> 2], v, memory_order_relaxed));
}

static inline uint lds_rmw_smin(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    return uint(atomic_fetch_min_explicit((threadgroup atomic_int*)&lds[a >> 2], int(v), memory_order_relaxed));
}

static inline uint lds_rmw_smax(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    return uint(atomic_fetch_max_explicit((threadgroup atomic_int*)&lds[a >> 2], int(v), memory_order_relaxed));
}

static inline uint lds_rmw_swap(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    return uint(atomic_exchange_explicit((threadgroup atomic_uint*)&lds[a >> 2], v, memory_order_relaxed));
}

static inline uint lds_rd(threadgroup uint* lds, uint a) {
    return a < LDS_BYTES ? lds[a >> 2] : 0u;
}

static inline void lds_wr(threadgroup uint* lds, uint a, uint v) {
    if (a < LDS_BYTES) lds[a >> 2] = v;
}

static inline uint lds_cas(threadgroup uint* lds, uint a, uint v, uint cmp) {
    if (a >= LDS_BYTES) return 0u;
    threadgroup atomic_uint* p = (threadgroup atomic_uint*)&lds[a >> 2];
    uint e = cmp;
    while (!atomic_compare_exchange_weak_explicit(p, &e, v, memory_order_relaxed, memory_order_relaxed) && e == cmp) {}
    return e;
}

// LDS atomics Metal lacks, as compare-and-swap loops; `op` picks the update:
// 0 fmin, 1 fmax, 2 rsub, 3 inc, 4 dec, 5 wrap (with `w`), 6 mskor (`v` the
// mask, `w` the bits), 7 fadd (a NaN operand comes back quieted), 8 float
// compare-and-store (`v` compared, `w` stored).
static inline uint lds_rmw_op(threadgroup uint* lds, uint a, uint v, uint w, uint op) {
    if (a >= LDS_BYTES) return 0u;
    threadgroup atomic_uint* p = (threadgroup atomic_uint*)&lds[a >> 2];
    uint e = atomic_load_explicit(p, memory_order_relaxed);
    for (;;) {
        uint n;
        switch (op) {
        case 0u: n = fminmax_bits(e, v, false); break;
        case 1u: n = fminmax_bits(e, v, true); break;
        case 2u: n = v - e; break;
        case 3u: n = e >= v ? 0u : e + 1u; break;
        case 4u: n = e == 0u || e > v ? v : e - 1u; break;
        case 5u: n = e >= v ? e - v : e + w; break;
        case 6u: n = (e & ~v) | w; break;
        case 7u: n = isnan(F(e)) ? e | 0x400000u : isnan(F(v)) ? v | 0x400000u : U(F(e) + F(v)); break;
        default: n = F(e) == F(v) ? w : e; break;
        }
        if (n == e || atomic_compare_exchange_weak_explicit(p, &e, n, memory_order_relaxed, memory_order_relaxed)) return e;
    }
}

// 64-bit LDS atomics, a read and a write of two dwords: exact when lanes that
// hit one place run one after another, as lockstep programs run them. `op`: 0 add,
// 1 sub, 2 rsub, 3 inc, 4 dec, 5 smin, 6 smax, 7 umin, 8 umax, 9 and, 10 or,
// 11 xor, 12 mskor (`v` the mask, `w` the bits), 13 exchange, 14 compare-and-store
// (`v` compared, `w` stored), 15 the same on doubles, 16 fmin, 17 fmax.
static inline ulong lds_rmw64(threadgroup uint* lds, uint a, ulong v, ulong w, uint op) {
    if (a + 4u >= LDS_BYTES) return 0ul;
    ulong e = ulong(lds[(a >> 2) + 1u]) << 32 | lds[a >> 2], n;
    switch (op) {
    case 0u: n = e + v; break;
    case 1u: n = e - v; break;
    case 2u: n = v - e; break;
    case 3u: n = e >= v ? 0ul : e + 1ul; break;
    case 4u: n = e == 0ul || e > v ? v : e - 1ul; break;
    case 5u: n = ulong(min(long(e), long(v))); break;
    case 6u: n = ulong(max(long(e), long(v))); break;
    case 7u: n = min(e, v); break;
    case 8u: n = max(e, v); break;
    case 9u: n = e & v; break;
    case 10u: n = e | v; break;
    case 11u: n = e ^ v; break;
    case 12u: n = (e & ~v) | w; break;
    case 13u: n = v; break;
    case 14u: n = e == v ? w : e; break;
    case 15u: n = f64_eq(e, v) ? w : e; break;
    case 16u: n = f64_minmax(e, v, false); break;
    default: n = f64_minmax(e, v, true); break;
    }
    lds[a >> 2] = uint(n);
    lds[(a >> 2) + 1u] = uint(n >> 32);
    return e;
}
"#;

/// A lane's scratch memory, `SCRATCH_BYTES` of it as a private array, read and
/// written little-endian at any byte. Accesses past its end read 0 and are dropped.
pub(super) const SCRATCH_PRELUDE: &str = r#"
static inline uint scratch_ld(thread uint* s, uint a, uint bytes) {
    if (a >= SCRATCH_BYTES || SCRATCH_BYTES - a < bytes) return 0u;
    if ((a & 3u) == 0u && bytes == 4u) return s[a >> 2];
    uint v = 0u;
    for (uint i = 0u; i < bytes; i++) v |= (s[(a + i) >> 2] >> (8u * ((a + i) & 3u)) & 0xffu) << (8u * i);
    return v;
}

static inline void scratch_st(thread uint* s, uint a, uint bytes, uint v) {
    if (a >= SCRATCH_BYTES || SCRATCH_BYTES - a < bytes) return;
    if ((a & 3u) == 0u && bytes == 4u) { s[a >> 2] = v; return; }
    for (uint i = 0u; i < bytes; i++) {
        uint k = (a + i) >> 2, sh = 8u * ((a + i) & 3u);
        s[k] = (s[k] & ~(0xffu << sh)) | ((v >> (8u * i) & 0xffu) << sh);
    }
}
"#;
