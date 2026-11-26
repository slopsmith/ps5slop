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

// v_cmp_class_f32's mask: signalling NaN, quiet NaN, -inf, -normal,
// -denormal, -0, +0, +denormal, +normal, +inf.
static inline bool fclass(float x, uint mask) {
    uint b = as_type<uint>(x);
    bool neg = (b >> 31) != 0u;
    uint e = (b >> 23) & 0xffu, f = b & 0x7fffffu;
    uint k;
    if (e == 0xffu) k = f != 0u ? ((f >> 22) != 0u ? 1u : 0u) : (neg ? 2u : 9u);
    else if (e == 0u) k = f != 0u ? (neg ? 4u : 7u) : (neg ? 5u : 6u);
    else k = neg ? 3u : 8u;
    return (mask >> k & 1u) != 0u;
}

static inline uint bfe_u32(uint v, uint o, uint w) {
    w = min(w, 32u);
    if (w == 0u) return 0u;
    uint m = w >= 32u ? 0xffffffffu : ((1u << w) - 1u);
    return (v >> (o & 31u)) & m;
}

static inline uint bfe_i32(uint v, uint o, uint w) {
    w = min(w, 32u);
    if (w == 0u) return 0u;
    int s = int(v << (32u - min(32u, (o & 31u) + w))) >> (32u - w);
    return uint(s);
}

static inline ulong bfe_u64(ulong v, uint ctrl) {
    uint o = ctrl & 63u;
    uint w = min((ctrl >> 16) & 0x7fu, 64u);
    if (w == 0u) return 0ul;
    ulong m = w >= 64u ? ~0ul : ((1ul << w) - 1ul);
    return (v >> o) & m;
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

static inline float med3(float a, float b, float c) {
    return fmax(fmin(a, b), fmin(fmax(a, b), c));
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

// Byte offset into the memory buffer of a buffer element, or OOB_OFF when out of range.
static inline long buffer_offset(Win mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    ulong base = ulong(vs.x) | (ulong(vs.y & 0xffffu) << 32);
    uint stride = (vs.y >> 16) & 0x3fffu;
    // OOB_SELECT (word 3 bits 28-29) for strided buffers, as RDNA2 ISA 8.1.5 table 35 states it.
    uint i = idxen != 0u ? index : 0u;
    uint oob = (vs.w >> 28) & 3u;
    bool out = stride == 0u ? offset >= vs.z
        : oob == 0u ? i >= vs.z || offset >= stride
        : oob == 1u ? i >= vs.z
        : oob == 2u ? vs.z == 0u
        : idxen != 0u ? i >= vs.z : offset >= vs.z;
    if (out) return OOB_OFF;
    return goff(mem_base, base + ulong(index) * ulong(stride) * ulong(idxen) + ulong(offset));
}

// As `buffer_offset`, for stores and atomics: OOB_OFF inside [ro, ro_end) too.
static inline long buffer_store_offset(Win w, uint4 vs, uint index, uint offset, uint idxen) {
    ulong a = (ulong(vs.x) | (ulong(vs.y & 0xffffu) << 32))
        + ulong(index) * ulong((vs.y >> 16) & 0x3fffu) * ulong(idxen) + ulong(offset);
    if (a >= w.ro && a < w.ro_end) return OOB_OFF;
    return buffer_offset(w, vs, index, offset, idxen);
}

static inline uint4 buffer_load_raw(device const uchar* mem, Win mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    long o = buffer_offset(mem_base, vs, index, offset, idxen);
    if (o == OOB_OFF) return uint4(0u);
    device const uint* p = (device const uint*)(mem + o);
    return uint4(p[0], p[1], p[2], p[3]);
}

// GLC loads and stores, which other groups may be polling or writing while
// this one runs: atomics, so the compiler keeps them in loops and in order.
static inline uint4 buffer_load_atomic(device uchar* mem, Win mem_base, uint4 vs, uint index, uint offset, uint idxen, uint n) {
    long o = buffer_offset(mem_base, vs, index, offset, idxen);
    uint4 r = uint4(0u);
    if (o == OOB_OFF) return r;
    if ((o & 3) != 0) return buffer_load_raw(mem, mem_base, vs, index, offset, idxen);
    for (uint i = 0u; i < n; i++) r[i] = atomic_load_explicit((device atomic_uint*)(mem + o) + i, memory_order_relaxed);
    return r;
}

static inline uint4 buffer_load_volatile(device const uchar* mem, Win mem_base, uint4 vs, uint index, uint offset, uint idxen, uint n) {
    long o = buffer_offset(mem_base, vs, index, offset, idxen);
    uint4 r = uint4(0u);
    if (o == OOB_OFF) return r;
    for (uint i = 0u; i < n; i++) r[i] = ((volatile device const uint*)(mem + o))[i];
    return r;
}

static inline void store_atomic(device uchar* mem, long o, uint v) {
    if ((o & 3) != 0) { *(device uint*)(mem + o) = v; return; }
    atomic_store_explicit((device atomic_uint*)(mem + o), v, memory_order_relaxed);
}

static inline uint4 buffer_load_ubyte(device const uchar* mem, Win mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    long o = buffer_offset(mem_base, vs, index, offset, idxen);
    return uint4(o == OOB_OFF ? 0u : uint(mem[o]), 0u, 0u, 0u);
}

static inline uint4 buffer_load_sbyte(device const uchar* mem, Win mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    long o = buffer_offset(mem_base, vs, index, offset, idxen);
    return uint4(o == OOB_OFF ? 0u : uint(int(char(mem[o]))), 0u, 0u, 0u);
}

static inline uint4 buffer_load_ushort(device const uchar* mem, Win mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    long o = buffer_offset(mem_base, vs, index, offset, idxen);
    return uint4(o == OOB_OFF ? 0u : uint(*(device const ushort*)(mem + o)), 0u, 0u, 0u);
}

static inline uint4 buffer_load_sshort(device const uchar* mem, Win mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    long o = buffer_offset(mem_base, vs, index, offset, idxen);
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

static inline uint atomic_rmw_fmin(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    device atomic_uint* p = (device atomic_uint*)(mem + o);
    uint e = atomic_load_explicit(p, memory_order_relaxed);
    for (;;) {
        uint n = as_type<uint>(fmin(as_type<float>(e), as_type<float>(v)));
        if (n == e || atomic_compare_exchange_weak_explicit(p, &e, n, memory_order_relaxed, memory_order_relaxed)) return e;
    }
}

static inline uint atomic_rmw_fmax(device uchar* mem, long o, uint v) {
    if (o == OOB_OFF) return 0u;
    device atomic_uint* p = (device atomic_uint*)(mem + o);
    uint e = atomic_load_explicit(p, memory_order_relaxed);
    for (;;) {
        uint n = as_type<uint>(fmax(as_type<float>(e), as_type<float>(v)));
        if (n == e || atomic_compare_exchange_weak_explicit(p, &e, n, memory_order_relaxed, memory_order_relaxed)) return e;
    }
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
    float s = (c.x - 1.5f) * 2.0f;
    float t = (c.y - 1.5f) * 2.0f;
    switch (uint(rint(c.z))) {
    case 0u: return float3(1.0f, -t, -s);
    case 1u: return float3(-1.0f, -t, s);
    case 2u: return float3(s, 1.0f, t);
    case 3u: return float3(s, -1.0f, -t);
    case 4u: return float3(s, -t, 1.0f);
    default: return float3(-s, -t, -1.0f);
    }
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

// One component of `bits` bits converted as `kind`: 0 UNORM, 1 SNORM, 2 USCALED,
// 3 SSCALED, 4 UINT, 5 SINT, 6 FLOAT.
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
    default: return bits == 16u ? as_type<uint>(float(as_type<half>(ushort(v)))) : v;
    }
}

static inline uint ld_le(device const uchar* p, uint bytes) {
    uint v = 0u;
    for (uint i = 0u; i < bytes; i++) v |= uint(p[i]) << (8u * i);
    return v;
}

// Components, bits per component and conversion of a buffer format
// (GFX10 unified encoding, BUF_FMT_*); n is 0 for formats not handled.
static inline uint3 buffer_fmt(uint fmt) {
    uint n = 0u, bits = 0u, kind = 0u;
    if (fmt >= 1u && fmt <= 6u) { n = 1u; bits = 8u; kind = fmt - 1u; }
    else if (fmt >= 7u && fmt <= 13u) { n = 1u; bits = 16u; kind = fmt - 7u; }
    else if (fmt >= 14u && fmt <= 19u) { n = 2u; bits = 8u; kind = fmt - 14u; }
    else if (fmt >= 20u && fmt <= 22u) { n = 1u; bits = 32u; kind = fmt - 16u; }
    else if (fmt >= 23u && fmt <= 29u) { n = 2u; bits = 16u; kind = fmt - 23u; }
    else if (fmt >= 50u && fmt <= 55u) { n = 4u; bits = 10u; kind = fmt - 50u; }
    else if (fmt >= 56u && fmt <= 61u) { n = 4u; bits = 8u; kind = fmt - 56u; }
    else if (fmt >= 62u && fmt <= 64u) { n = 2u; bits = 32u; kind = fmt - 58u; }
    else if (fmt >= 65u && fmt <= 71u) { n = 4u; bits = 16u; kind = fmt - 65u; }
    else if (fmt >= 72u && fmt <= 74u) { n = 3u; bits = 32u; kind = fmt - 68u; }
    else if (fmt >= 75u && fmt <= 77u) { n = 4u; bits = 32u; kind = fmt - 71u; }
    return uint3(n, bits, kind);
}

// Buffer formats as raw result dwords.
static inline uint4 buffer_load_format(device const uchar* mem, Win mem_base, uint4 vs, uint index, uint offset, uint idxen) {
    long o = buffer_offset(mem_base, vs, index, offset, idxen);
    if (o == OOB_OFF) return uint4(0u);
    device const uchar* p = mem + o;
    uint3 f = buffer_fmt((vs.w >> 12) & 0x7fu);
    uint n = f.x, bits = f.y, kind = f.z;
    uint4 r = uint4(0u);
    if (bits == 10u) {
        uint w = ld_le(p, 4u);
        r = uint4(fmt_conv(w, 10u, kind), fmt_conv(w >> 10, 10u, kind), fmt_conv(w >> 20, 10u, kind), fmt_conv(w >> 30, 2u, kind));
    } else {
        uint bytes = bits / 8u;
        for (uint i = 0u; i < n; i++) r[i] = fmt_conv(ld_le(p + i * bytes, bytes), bits, kind);
    }
    uint one = kind == 4u || kind == 5u ? 1u : 0x3f800000u;
    return uint4(fmt_sel(vs.w & 7u, r, one), fmt_sel((vs.w >> 3) & 7u, r, one),
                 fmt_sel((vs.w >> 6) & 7u, r, one), fmt_sel((vs.w >> 9) & 7u, r, one));
}

// The inverse of fmt_conv: a result dword packed into a `bits`-bit component.
static inline uint fmt_pack(uint v, uint bits, uint kind) {
    uint m = bits == 32u ? 0xffffffffu : (1u << bits) - 1u;
    float f = as_type<float>(v);
    switch (kind) {
    case 0u: return uint(rint(clamp(f, 0.0f, 1.0f) * float(m))) & m;
    case 1u: return uint(int(rint(clamp(f, -1.0f, 1.0f) * float(m >> 1)))) & m;
    case 2u: return uint(clamp(f, 0.0f, float(m))) & m;
    case 3u: return uint(int(clamp(f, -float(m >> 1) - 1.0f, float(m >> 1)))) & m;
    case 4u: case 5u: return v & m;
    default: return bits == 16u ? uint(as_type<ushort>(half(f))) : v;
    }
}

static inline void st_le(device uchar* p, uint bytes, uint v) {
    for (uint i = 0u; i < bytes; i++) p[i] = uchar(v >> (8u * i));
}

// Typed store: the first n data components go to the format's components in
// order; components past n are left as they are.
static inline void buffer_store_format(device uchar* mem, Win mem_base, uint4 vs, uint index, uint offset, uint idxen, uint4 d, uint n) {
    long o = buffer_store_offset(mem_base, vs, index, offset, idxen);
    if (o == OOB_OFF) return;
    device uchar* p = mem + o;
    uint3 f = buffer_fmt((vs.w >> 12) & 0x7fu);
    uint bits = f.y, kind = f.z;
    if (f.x == 0u) return;
    if (bits == 10u) {
        uint w = ld_le(p, 4u);
        uint shift[4] = {0u, 10u, 20u, 30u};
        for (uint i = 0u; i < min(n, 4u); i++) {
            uint b = i == 3u ? 2u : 10u;
            uint m = ((1u << b) - 1u) << shift[i];
            w = (w & ~m) | ((fmt_pack(d[i], b, kind) << shift[i]) & m);
        }
        st_le(p, 4u, w);
        return;
    }
    uint bytes = bits / 8u;
    for (uint i = 0u; i < min(n, f.x); i++) st_le(p + i * bytes, bytes, fmt_pack(d[i], bits, kind));
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

static inline uint lds_rmw_fmin(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    threadgroup atomic_uint* p = (threadgroup atomic_uint*)&lds[a >> 2];
    uint e = atomic_load_explicit(p, memory_order_relaxed);
    for (;;) {
        uint n = as_type<uint>(fmin(as_type<float>(e), as_type<float>(v)));
        if (n == e || atomic_compare_exchange_weak_explicit(p, &e, n, memory_order_relaxed, memory_order_relaxed)) return e;
    }
}

static inline uint lds_rmw_fmax(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    threadgroup atomic_uint* p = (threadgroup atomic_uint*)&lds[a >> 2];
    uint e = atomic_load_explicit(p, memory_order_relaxed);
    for (;;) {
        uint n = as_type<uint>(fmax(as_type<float>(e), as_type<float>(v)));
        if (n == e || atomic_compare_exchange_weak_explicit(p, &e, n, memory_order_relaxed, memory_order_relaxed)) return e;
    }
}

static inline uint lds_rmw_rsub(threadgroup uint* lds, uint a, uint v) {
    if (a >= LDS_BYTES) return 0u;
    threadgroup atomic_uint* p = (threadgroup atomic_uint*)&lds[a >> 2];
    uint e = atomic_load_explicit(p, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(p, &e, v - e, memory_order_relaxed, memory_order_relaxed)) {}
    return e;
}
"#;
