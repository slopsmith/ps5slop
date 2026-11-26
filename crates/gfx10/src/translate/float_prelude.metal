// Float VALU semantics under the program's float mode. Every helper takes
// and returns bit patterns, so Metal never flushes or canonicalizes a value
// on its own. The mode is fixed per program:
//   FM_RND32, FM_RND16: round mode (0 nearest even, 1 +inf, 2 -inf, 3 zero);
//   FM_DN32, FM_DN16: denormals (bit 0 inputs kept, bit 1 results kept);
//   FM_DX10: a clamped NaN becomes 0; FM_IEEE: signalling NaNs are quieted,
//   output modifiers are ignored.
// FX_NAN, FX_DN, FX_RND and FX_MODS are 0 for parts left to Metal's own
// arithmetic (Request::plain_float), for measuring what they cost.
// Metal float arithmetic rounds to nearest even and flushes f32 denormals,
// so other round modes come from the rounding error, which TwoSum and fma
// give exactly.

static inline bool nan32(uint x) { return (x & 0x7fffffffu) > 0x7f800000u; }
static inline bool snan32(uint x) { return nan32(x) && (x & 0x400000u) == 0u; }
static inline bool den32(uint x) { return (x & 0x7f800000u) == 0u && (x & 0x7fffffu) != 0u; }
static inline bool zero32(uint x) { return (x & 0x7fffffffu) == 0u; }
static inline uint flush32(uint x) { return FX_DN && den32(x) ? x & 0x80000000u : x; }
static inline uint in32(uint x) { return (FM_DN32 & 1) == 0 ? flush32(x) : x; }
static inline uint out32(uint x) { return (FM_DN32 & 2) == 0 ? flush32(x) : x; }
static inline uint quiet32(uint x) { return FX_NAN && FM_IEEE ? x | 0x400000u : x; }

// The NaN an instruction returns: its first NaN source, or the default NaN.
static inline uint nan_of32(uint a, uint b, uint c) {
    return nan32(a) ? quiet32(a) : nan32(b) ? quiet32(b) : nan32(c) ? quiet32(c) : 0xffc00000u;
}

static inline uint fin32(float r, uint a, uint b, uint c) {
    uint u = as_type<uint>(r);
    return FX_NAN && nan32(u) ? nan_of32(a, b, c) : out32(u);
}

// An order-preserving integer for a non-NaN float; both zeros map to 0.
static inline int key32(uint x) { return (x >> 31) != 0u ? -int(x & 0x7fffffffu) : int(x); }

// The error of s = a + b rounded, exactly (TwoSum).
static inline float sum_err(float a, float b, float s) {
    float bb = s - a;
    return (a - (s - bb)) + (b - bb);
}

// The sign of x + y + z, exactly (three passes of VecSum).
static inline float sign3(float x, float y, float z) {
    for (int i = 0; i < 3; i++) {
        float s = x + y; float e = sum_err(x, y, s);
        float t = s + z; float f = sum_err(s, z, t);
        x = e; y = f; z = t;
    }
    return z != 0.0f ? sign(z) : y != 0.0f ? sign(y) : sign(x);
}

// Error terms below 2^-126 would be flushed, so tiny operands are scaled
// up first; the scale is exact.

static inline float add_err(float x, float y, float r) {
    if (max(abs(x), abs(y)) >= 0x1p-60f) return sum_err(x, y, r);
    float xs = x * 0x1p64f, ys = y * 0x1p64f;
    return sum_err(xs, ys, xs + ys);
}

static inline float mul_err(float x, float y, float r) {
    if (abs(r) >= 0x1p-60f || isinf(r)) return fma(x, y, -r);
    if (abs(x) > abs(y)) { float t = x; x = y; y = t; }
    return fma(x * 0x1p64f, y, -(r * 0x1p64f));
}

static inline float fma_err_scaled(float a, float b, float c, float r) {
    float p = a * b;
    float pe = fma(a, b, -p);
    float s = p + c;
    float se = sum_err(p, c, s);
    return sign3(s - r, se, pe);
}

// The sign of a * b + c - r, for r = fma(a, b, c) rounded.
static inline float fma_err(float a, float b, float c, float r) {
    if (abs(a * b) >= 0x1p-60f || isinf(a * b)) return fma_err_scaled(a, b, c, r);
    if (abs(a) > abs(b)) { float t = a; a = b; b = t; }
    float ps = a * 0x1p64f * b;
    if (abs(c) >= 0x1p-60f || abs(ps) < abs(c) * 0x1p34f) {
        // The product is below c's last bit: only its sign counts.
        float d = c - r;
        if (d != 0.0f) return sign(d);
        return a == 0.0f || b == 0.0f ? 0.0f : sign(a) * sign(b);
    }
    return fma_err_scaled(a * 0x1p64f, b, c * 0x1p64f, r * 0x1p64f);
}

// A zero result r with e the sign of what rounding lost: a flushed tiny
// result keeps its sign; an exact zero is -0 when rounding toward -inf,
// unless both addends are +0 (pos).
static inline float zero_sign(float r, float e, bool pos, uint m) {
    if (r != 0.0f) return r;
    if (e != 0.0f) return copysign(0.0f, e);
    return m == 2u && !pos ? -0.0f : r;
}

static inline bool pos0(float x, float y) { return as_type<uint>(x) == 0u && as_type<uint>(y) == 0u; }

// The product of a and b as a zero of the right sign, or 1 if not zero.
static inline float prod0(uint a, uint b) {
    return zero32(a) || zero32(b) ? as_type<float>((a ^ b) & 0x80000000u) : 1.0f;
}

// One step from x toward +inf (up) or -inf.
static inline float step32(float x, bool up) {
    uint u = as_type<uint>(x);
    if (zero32(u)) return as_type<float>(up ? 1u : 0x80000001u);
    return as_type<float>(((u >> 31) == 0u) == up ? u + 1u : u - 1u);
}

// r, rounded to nearest even from a value e above it (e's sign is what
// matters), rounded under mode m instead.
static inline float round_dir(float r, float e, uint m) {
    if (m == 0u || e == 0.0f || isnan(e) || isinf(r)) return r;
    bool up = e > 0.0f;
    if (m == 1u) return up ? step32(r, true) : r;
    if (m == 2u) return up ? r : step32(r, false);
    return up == (r < 0.0f) ? step32(r, up) : r;
}

// A finite result that overflowed to infinity, under mode m.
static inline float round_ovf(float r, uint m) {
    bool toward = m == 3u || (m == 1u && r < 0.0f) || (m == 2u && r > 0.0f);
    return toward ? copysign(FLT_MAX, r) : r;
}

static inline float dir32(float r, float e, bool finite) {
    if (FM_RND32 == 0) return r;
    return isinf(r) && finite ? round_ovf(r, FM_RND32) : round_dir(r, e, FM_RND32);
}

// Exact f32 arithmetic on integers, for modes that keep denormals, which
// Metal flushes: a value is m * 2^e.
static inline void unpack32(uint x, thread bool& neg, thread int& e, thread ulong& m) {
    neg = (x >> 31) != 0u;
    uint ex = (x >> 23) & 0xffu;
    m = ulong(ex == 0u ? x & 0x7fffffu : (x & 0x7fffffu) | 0x800000u);
    e = (ex == 0u ? 1 : int(ex)) - 150;
}

// m * 2^e, plus something below m's last bit if sticky, as f32 bits
// rounded under mode md; results below 2^-126 before rounding are flushed
// unless keep.
static inline uint pack32(bool neg, int e, ulong m, bool sticky, uint md, bool keep) {
    uint s = neg ? 0x80000000u : 0u;
    if (m == 0ul) return s;
    int top = 63 - int(clz(m)) + e;
    if (top < -126 && !keep) return s;
    if (top > 127) return s | (md == 3u || (md == 1u && neg) || (md == 2u && !neg) ? 0x7f7fffffu : 0x7f800000u);
    int lsb = max(top - 23, -149);
    int sh = lsb - e;
    ulong q = 0ul, rem = 0ul, mid = 1ul;
    if (sh <= 0) {
        q = m << uint(-sh);
    } else {
        if (sh > 62) {
            int k = sh - 62;
            sticky = sticky || (k >= 64 ? m != 0ul : (m & ((1ul << uint(k)) - 1ul)) != 0ul);
            m = k >= 64 ? 0ul : m >> uint(k);
            sh = 62;
        }
        q = m >> uint(sh);
        rem = m & ((1ul << uint(sh)) - 1ul);
        mid = 1ul << uint(sh - 1);
    }
    bool away = (md == 1u && !neg) || (md == 2u && neg);
    bool toward = md == 3u || (md == 1u && neg) || (md == 2u && !neg);
    if (md == 0u) {
        if (rem > mid || (rem == mid && (sticky || (q & 1ul) != 0ul))) q += 1ul;
    } else if (away && (rem != 0ul || sticky)) {
        q += 1ul;
    }
    uint bits = (uint(lsb + 149) << 23) + uint(q);
    if (bits >= 0x7f800000u) bits = toward ? 0x7f7fffffu : 0x7f800000u;
    return s | bits;
}

// (-1)^na ma 2^ea + (-1)^nb mb 2^eb, rounded once.
static inline uint soft_sum(bool na, int ea, ulong ma, bool nb, int eb, ulong mb, uint md, bool keep) {
    if (ma == 0ul && mb == 0ul) return (na && nb) || (na != nb && md == 2u) ? 0x80000000u : 0u;
    if (ma == 0ul) return pack32(nb, eb, mb, false, md, keep);
    if (mb == 0ul) return pack32(na, ea, ma, false, md, keep);
    int sa = int(clz(ma)) - 2, sb = int(clz(mb)) - 2;
    ma <<= uint(sa); ea -= sa;
    mb <<= uint(sb); eb -= sb;
    if (ea < eb || (ea == eb && mb > ma)) {
        bool tn = na; na = nb; nb = tn;
        int te = ea; ea = eb; eb = te;
        ulong tm = ma; ma = mb; mb = tm;
    }
    int d = ea - eb;
    bool st = false;
    if (d >= 62) {
        st = true;
        mb = 0ul;
    } else if (d > 0) {
        st = (mb & ((1ul << uint(d)) - 1ul)) != 0ul;
        mb >>= uint(d);
    }
    ulong m = na == nb ? ma + mb : ma - mb - (st ? 1ul : 0ul);
    if (m == 0ul && !st) return md == 2u ? 0x80000000u : 0u;
    return pack32(na, ea, m, st, md, keep);
}

// Whether a result with these finite sources needs the exact path: a
// denormal source or a result near the denormal range, in a mode that
// keeps denormals.
static inline bool soft32(float r, uint a, uint b, uint c) {
    if (!FX_DN || FM_DN32 == 0) return false;
    bool fin = (a & 0x7f800000u) != 0x7f800000u && (b & 0x7f800000u) != 0x7f800000u && (c & 0x7f800000u) != 0x7f800000u;
    return fin && (den32(a) || den32(b) || den32(c) || abs(r) < 0x1p-125f);
}

static inline uint soft_add32(uint a, uint b) {
    bool na, nb; int ea, eb; ulong ma, mb;
    unpack32(a, na, ea, ma); unpack32(b, nb, eb, mb);
    return soft_sum(na, ea, ma, nb, eb, mb, FM_RND32, (FM_DN32 & 2) != 0);
}

static inline uint soft_fma32(uint a, uint b, uint c) {
    bool na, nb, nc; int ea, eb, ec; ulong ma, mb, mc;
    unpack32(a, na, ea, ma); unpack32(b, nb, eb, mb); unpack32(c, nc, ec, mc);
    return soft_sum(na != nb, ea + eb, ma * mb, nc, ec, mc, FM_RND32, (FM_DN32 & 2) != 0);
}

// A source as Metal sees it: a kept denormal, which Metal would flush,
// stands in as the smallest normal for results the exact path leaves to
// Metal (those with an infinite or NaN source).
static inline float val32(uint a) {
    return FX_DN && (FM_DN32 & 1) != 0 && den32(a) ? as_type<float>((a & 0x80000000u) | 0x800000u) : F(a);
}

static inline uint add32(uint a, uint b) {
    a = in32(a); b = in32(b);
    float x = val32(a), y = val32(b), r = x + y;
    if (soft32(r, a, b, 0u)) return soft_add32(a, b);
    if (FM_RND32 != 0) {
        float e = add_err(x, y, r);
        r = zero_sign(dir32(r, e, isfinite(x) && isfinite(y)), e, pos0(x, y), FM_RND32);
    }
    return fin32(r, a, b, 0u);
}

static inline uint sub32(uint a, uint b) { return add32(a, b ^ 0x80000000u); }

static inline uint mul32(uint a, uint b) {
    a = in32(a); b = in32(b);
    float x = val32(a), y = val32(b), r = x * y;
    if (soft32(r, a, b, 0u)) {
        bool na, nb; int ea, eb; ulong ma, mb;
        unpack32(a, na, ea, ma); unpack32(b, nb, eb, mb);
        return pack32(na != nb, ea + eb, ma * mb, false, FM_RND32, (FM_DN32 & 2) != 0);
    }
    r = dir32(r, mul_err(x, y, r), isfinite(x) && isfinite(y));
    return fin32(r, a, b, 0u);
}

// DX9 rules: zero times anything, even infinity or NaN, is zero.
static inline uint mul_legacy32(uint a, uint b) {
    a = in32(a); b = in32(b);
    return zero32(a) || zero32(b) ? 0u : mul32(a, b);
}

// The NaN of a * b + c: the product's NaN (an invalid product gives the
// default NaN) comes before c's.
static inline uint nan_fma32(uint a, uint b, uint c) {
    if (nan32(a) || nan32(b)) return nan_of32(a, b, 0u);
    if ((zero32(a) && (b & 0x7fffffffu) == 0x7f800000u) || (zero32(b) && (a & 0x7fffffffu) == 0x7f800000u)) return 0xffc00000u;
    return nan_of32(c, 0u, 0u);
}

static inline uint fma32(uint a, uint b, uint c) {
    a = in32(a); b = in32(b); c = in32(c);
    float x = val32(a), y = val32(b), z = val32(c), r = fma(x, y, z);
    if (soft32(r, a, b, c)) return soft_fma32(a, b, c);

    if (FM_RND32 != 0) {
        float e = fma_err(x, y, z, r);
        r = zero_sign(dir32(r, e, isfinite(x) && isfinite(y) && isfinite(z)), e, pos0(prod0(a, b), z), FM_RND32);
    }
    uint u = as_type<uint>(r);
    return FX_NAN && nan32(u) ? nan_fma32(a, b, c) : out32(u);
}

// A product of the legacy multiply rules plus c, fused.
static inline uint fma_legacy32(uint a, uint b, uint c) {
    a = in32(a); b = in32(b);
    return zero32(a) || zero32(b) ? add32(0u, c) : fma32(a, b, c);
}

// Multiply, then add, each rounded; denormals are flushed whatever the mode.
static inline uint mad32(uint a, uint b, uint c) {
    a = flush32(a); b = flush32(b); c = flush32(c);
    float x = F(a), y = F(b), z = F(c), p = x * y;
    p = F(flush32(as_type<uint>(dir32(p, mul_err(x, y, p), isfinite(x) && isfinite(y)))));
    float r = p + z;
    if (FM_RND32 != 0) {
        float e = add_err(p, z, r);
        r = zero_sign(dir32(r, e, isfinite(p) && isfinite(z)), e, pos0(p, z), FM_RND32);
    }
    uint u = as_type<uint>(r);
    return FX_NAN && nan32(u) ? nan_fma32(a, b, c) : flush32(u);
}

static inline uint mad_legacy32(uint a, uint b, uint c) {
    return zero32(flush32(a)) || zero32(flush32(b)) ? mad32(0u, 0u, c) : mad32(a, b, c);
}

static inline uint max32(uint a, uint b) {
    a = in32(a); b = in32(b);
    if (!FX_NAN) return out32(as_type<uint>(fmax(F(a), F(b))));
    if (FM_IEEE && snan32(a)) return quiet32(a);
    if (FM_IEEE && snan32(b)) return quiet32(b);
    if (nan32(a)) return nan32(b) ? a : out32(b);
    if (nan32(b)) return out32(a);
    int x = key32(a), y = key32(b);
    return out32(x > y ? a : y > x ? b : (a >> 31) != 0u ? b : a);
}

static inline uint min32(uint a, uint b) {
    a = in32(a); b = in32(b);
    if (!FX_NAN) return out32(as_type<uint>(fmin(F(a), F(b))));
    if (FM_IEEE && snan32(a)) return quiet32(a);
    if (FM_IEEE && snan32(b)) return quiet32(b);
    if (nan32(a)) return nan32(b) ? a : out32(b);
    if (nan32(b)) return out32(a);
    int x = key32(a), y = key32(b);
    return out32(x < y ? a : y < x ? b : (a >> 31) != 0u ? a : b);
}

static inline uint med3_32(uint a, uint b, uint c) {
    a = in32(a); b = in32(b); c = in32(c);
    if (!FX_NAN) return max32(min32(a, b), min32(max32(a, b), c));
    if (nan32(a) || nan32(b) || nan32(c)) return min32(min32(a, b), c);
    uint m = max32(max32(a, b), c);
    if (m == a) return max32(b, c);
    if (m == b) return max32(a, c);
    return max32(a, b);
}

// A one-source op whose Metal result is right but for NaNs and denormals.
static inline uint op1_32(float r, uint a) { return fin32(r, a, 0u, 0u); }

// A kept denormal source of a rounding op is a signed zero to Metal.
static inline float tiny32(uint a, float below, float above) {
    return (a >> 31) != 0u ? below : above;
}

static inline uint floor32(uint a) {
    a = in32(a);
    return fin32(den32(a) ? tiny32(a, -1.0f, 0.0f) : floor(F(a)), a, 0u, 0u);
}
static inline uint ceil32(uint a) {
    a = in32(a);
    return fin32(den32(a) ? tiny32(a, -0.0f, 1.0f) : ceil(F(a)), a, 0u, 0u);
}
static inline uint trunc32(uint a) { a = in32(a); return fin32(trunc(F(a)), a, 0u, 0u); }
static inline uint rndne32(uint a) { a = in32(a); return fin32(rint(F(a)), a, 0u, 0u); }


// Transcendentals flush denormals whatever the mode.
static inline uint tr32(float r, uint a) {
    uint u = as_type<uint>(r);
    if (!FX_NAN) return flush32(u);
    return nan32(a) ? quiet32(a) : nan32(u) ? 0xffc00000u : flush32(u);
}

static inline uint rcp32(uint a) { a = flush32(a); return tr32(1.0f / F(a), a); }
static inline uint rsq32(uint a) { a = flush32(a); return tr32(rsqrt(F(a)), a); }
static inline uint sqrt32(uint a) { a = flush32(a); return tr32(sqrt(F(a)), a); }
static inline uint exp32(uint a) { a = flush32(a); return tr32(exp2(F(a)), a); }
static inline uint log32(uint a) { a = flush32(a); return tr32(log2(F(a)), a); }

// sin and cos of x turns: x less its nearest integer is exact.
static inline uint sin32(uint a) {
    a = in32(a);
    float x = F(a);
    if (zero32(a)) return a;
    float r = isinf(x) ? NAN : sinpi(2.0f * (x - rint(x)));
    return fin32(r == 0.0f ? 0.0f : r, a, 0u, 0u);
}

static inline uint cos32(uint a) {
    a = in32(a);
    float x = F(a);
    return fin32(isinf(x) ? NAN : cospi(2.0f * (x - rint(x))), a, 0u, 0u);
}

static inline uint cubeid32(uint a, uint b, uint c) {
    a = in32(a); b = in32(b); c = in32(c);
    return fin32(cube_id(F(a), F(b), F(c)), a, b, c);
}
// cubesc and cubetc pick a source as it is, quieted in IEEE mode.
static inline uint cubesc32(uint a, uint b, uint c) {
    uint r = as_type<uint>(cube_sc(F(a), F(b), F(c)));
    return nan32(r) ? quiet32(r) : r;
}
static inline uint cubetc32(uint a, uint b, uint c) {
    uint r = as_type<uint>(cube_tc(F(a), F(b), F(c)));
    return nan32(r) ? quiet32(r) : r;
}

// cubema is 2 * major + 0, so a zero major axis gives +0; denormals are
// flushed whatever the mode.
static inline uint cubema32(uint a, uint b, uint c) {
    a = flush32(a); b = flush32(b); c = flush32(c);
    float x = F(a), y = F(b), z = F(c);
    float m = abs(z) >= abs(x) && abs(z) >= abs(y) ? z : abs(y) >= abs(x) ? y : x;
    float r = zero_sign(2.0f * m + 0.0f, 0.0f, pos0(m, 0.0f), FM_RND32);
    if (isinf(r) && isfinite(m)) r = round_ovf(r, FM_RND32);


    return fin32(r, a, b, c);
}


// x - floor(x), rounded under the mode, below 1.
static inline uint fract32(uint a) {
    a = in32(a);
    float x = F(a);
    if (nan32(a)) return quiet32(a);
    if (isinf(x)) return 0xffc00000u;
    if (den32(a)) return (a >> 31) != 0u ? 0x3f7fffffu : out32(a);
    float f = floor(x), r = x - f;
    r = dir32(r, add_err(x, -f, r), true);
    return out32(as_type<uint>(min(r, as_type<float>(0x3f7fffffu))));
}

static inline uint ldexp32(uint a, int n) {
    a = in32(a);
    float x = F(a);
    if (nan32(a)) return quiet32(a);
    if (isinf(x) || zero32(a)) return a;
    bool neg; int e; ulong m;
    unpack32(a, neg, e, m);
    return pack32(neg, e + clamp(n, -400, 400), m, false, FM_RND32, (FM_DN32 & 2) != 0);
}

static inline uint cvt_f32_i32(int i) {
    float r = float(i);
    long d = long(i) - long(r);
    return out32(as_type<uint>(dir32(r, float(d), true)));
}

static inline uint cvt_f32_u32(uint i) {
    float r = float(i);
    long d = long(i) - long(r);
    return out32(as_type<uint>(dir32(r, float(d), true)));
}

static inline uint cvt_i32_f32(uint a) {
    a = in32(a);
    float x = F(a);
    if (nan32(a)) return 0u;
    if (x >= 2147483648.0f) return 0x7fffffffu;
    if (x <= -2147483648.0f) return 0x80000000u;
    return uint(int(x));
}

static inline uint cvt_u32_f32(uint a) {
    a = in32(a);
    float x = F(a);
    if (nan32(a) || x <= 0.0f) return 0u;
    if (x >= 4294967296.0f) return 0xffffffffu;
    return uint(x);
}

// floor(x) to an integer: NaNs saturate by their sign.
static inline uint cvt_flr_i32_f32(uint a) {
    a = in32(a);
    if (nan32(a)) return (a >> 31) != 0u ? 0x80000000u : 0x7fffffffu;
    float f = den32(a) ? tiny32(a, -1.0f, 0.0f) : floor(F(a));

    if (f >= 2147483648.0f) return 0x7fffffffu;
    if (f <= -2147483648.0f) return 0x80000000u;
    return uint(int(f));
}

// floor(x + 0.5) to an integer, with x + 0.5 exact.
static inline uint cvt_rpi_i32_f32(uint a) {
    a = in32(a);
    if (nan32(a)) return (a >> 31) != 0u ? 0x80000000u : 0x7fffffffu;
    float x = F(a), f = floor(x);
    if (f >= 2147483648.0f) return 0x7fffffffu;
    if (f < -2147483648.0f) return 0x80000000u;
    return uint(int(f) + (x - f >= 0.5f ? 1 : 0));
}

static inline uint cvt_snorm16(uint a) {
    a = in32(a);
    return nan32(a) ? 0u : pack_float_to_snorm2x16(float2(F(a), 0.0f));
}

static inline uint cvt_unorm16(uint a) {
    a = in32(a);
    return nan32(a) ? 0u : pack_float_to_unorm2x16(float2(F(a), 0.0f));
}

// Output modifiers make -0 +0.
static inline uint omod32(uint r, float k) {
    if (!FX_MODS) return as_type<uint>(F(r) * k);
    if (FM_IEEE || (FM_DN32 & 2) != 0 || nan32(r)) return r;
    uint x = flush32(as_type<uint>(F(r) * k));
    return zero32(x) ? 0u : x;
}

static inline uint clamp32(uint r) {
    if (!FX_MODS) return as_type<uint>(saturate(F(r)));
    if (nan32(r)) return FM_DX10 ? 0u : r;
    if ((r >> 31) != 0u) return 0u;
    return min(r, 0x3f800000u);
}

// f16 values are held in the low half of a uint. Every f16 sum, product
// and fused multiply-add of f16 values is exact in f32 or has its error
// there, so f16 arithmetic runs in f32 and is rounded once, by h16.

static inline bool nan16(uint x) { return (x & 0x7fffu) > 0x7c00u; }
static inline bool snan16(uint x) { return nan16(x) && (x & 0x200u) == 0u; }
static inline bool den16(uint x) { return (x & 0x7c00u) == 0u && (x & 0x3ffu) != 0u; }
static inline bool zero16(uint x) { return (x & 0x7fffu) == 0u; }
static inline uint flush16(uint x) { return FX_DN && den16(x) ? x & 0x8000u : x; }
static inline uint in16(uint x) { x &= 0xffffu; return (FM_DN16 & 1) == 0 ? flush16(x) : x; }
static inline uint out16(uint x) { return (FM_DN16 & 2) == 0 ? flush16(x) : x; }
static inline uint quiet16(uint x) { return FX_NAN && FM_IEEE ? x | 0x200u : x; }
// f16 bits widened to f32 by integer operations: Metal was seen to keep a
// widened half at half precision in later arithmetic.
static inline float hf(uint x) {
    uint s = (x & 0x8000u) << 16, e = (x >> 10) & 0x1fu, m = x & 0x3ffu;
    if (e == 0x1fu) return as_type<float>(s | 0x7f800000u | m << 13);
    if (e == 0u) return as_type<float>(s | as_type<uint>(float(m) * 0x1p-24f));
    return as_type<float>(s | (e + 112u) << 23 | m << 13);
}
static inline int key16(uint x) { return (x & 0x8000u) != 0u ? -int(x & 0x7fffu) : int(x); }

static inline uint nan_of16(uint a, uint b, uint c) {
    return nan16(a) ? quiet16(a) : nan16(b) ? quiet16(b) : nan16(c) ? quiet16(c) : 0xfe00u;
}

// x + lo as f16 bits, rounded under mode m (lo is below x's last bit, or
// zero); results below the smallest normal are flushed unless keep.
static inline uint h16(float x, float lo, uint m, bool keep) {
    if (!FX_RND && m == 0u && (keep || !FX_DN)) return uint(as_type<ushort>(half(x)));
    uint u = as_type<uint>(x);
    uint s = (u >> 16) & 0x8000u;
    uint a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return 0xfe00u;
    if (a == 0x7f800000u) return s | 0x7c00u;
    bool neg = s != 0u;
    float lm = neg ? -lo : lo;
    bool away = (m == 1u && !neg) || (m == 2u && neg);
    bool toward = m == 3u || (m == 1u && neg) || (m == 2u && !neg);
    uint enc = 0u, rem = 0u, mid = 1u;
    if (a != 0u) {
        int e = int(a >> 23) - 127;
        uint mant = a & 0x7fffffu;
        if ((a >> 23) == 0u) e = -126; else mant |= 0x800000u;
        if (FX_DN && !keep && e < -14) return s;

        uint sh = e >= -14 ? 13u : uint(13 + (-14 - e));
        if (sh >= 25u) {
            rem = 1u; mid = 2u;
        } else {
            enc = e >= -14 ? uint(e + 15) << 10 | (mant >> 13 & 0x3ffu) : mant >> sh;
            rem = mant & ((1u << sh) - 1u);
            mid = 1u << (sh - 1u);
        }
    }
    if (m == 0u) {
        if (rem > mid || (rem == mid && (lm > 0.0f || (lm == 0.0f && (enc & 1u) != 0u)))) enc += 1u;
    } else if (away) {
        if (rem != 0u || lm > 0.0f) enc += 1u;
    } else if (rem == 0u && lm < 0.0f) {
        enc -= 1u;
    }
    if (enc >= 0x7c00u) enc = toward ? 0x7bffu : 0x7c00u;
    if (FX_DN && !keep && enc < 0x400u) enc = 0u;
    return s | enc;
}

static inline uint fin16(float r, float lo, uint a, uint b, uint c) {
    return FX_NAN && isnan(r) ? nan_of16(a, b, c) : h16(r, lo, FM_RND16, (FM_DN16 & 2) != 0);
}

// Metal's own half arithmetic, for FX_RND 0.
static inline half hh(uint x) { return as_type<half>(ushort(x)); }
static inline uint hu(half x) { return uint(as_type<ushort>(x)); }

static inline uint add16(uint a, uint b) {
    a = in16(a); b = in16(b);
    if (!FX_RND) return out16(hu(hh(a) + hh(b)));
    float x = hf(a), y = hf(b), r = zero_sign(x + y, 0.0f, pos0(x, y), FM_RND16);
    return fin16(r, sum_err(x, y, r), a, b, 0u);
}

static inline uint sub16(uint a, uint b) { return add16(a, b ^ 0x8000u); }

static inline uint mul16(uint a, uint b) {
    a = in16(a); b = in16(b);
    if (!FX_RND) return out16(hu(hh(a) * hh(b)));
    return fin16(hf(a) * hf(b), 0.0f, a, b, 0u);
}

static inline uint nan_fma16(uint a, uint b, uint c) {
    if (nan16(a) || nan16(b)) return nan_of16(a, b, 0u);
    if ((zero16(a) && (b & 0x7fffu) == 0x7c00u) || (zero16(b) && (a & 0x7fffu) == 0x7c00u)) return 0xfe00u;
    return nan_of16(c, 0u, 0u);
}

static inline uint fma16(uint a, uint b, uint c) {
    a = in16(a); b = in16(b); c = in16(c);
    if (!FX_RND) return out16(hu(fma(hh(a), hh(b), hh(c))));
    float p = hf(a) * hf(b), z = hf(c), r = zero_sign(p + z, 0.0f, pos0(p, z), FM_RND16);
    return FX_NAN && isnan(r) ? nan_fma16(a, b, c) : fin16(r, sum_err(p, z, r), a, b, c);

}

static inline uint mad16(uint a, uint b, uint c) {
    a = in16(a); b = in16(b); c = in16(c);
    uint p = fin16(hf(a) * hf(b), 0.0f, a, b, 0u);
    return FX_NAN && nan16(p) ? nan_fma16(a, b, c) : add16(p, c);
}

static inline uint ldexp16(uint a, int n) {
    a = in16(a);
    if (nan16(a)) return quiet16(a);
    return fin16(ldexp(hf(a), clamp(n, -64, 64)), 0.0f, a, 0u, 0u);
}

static inline uint max16(uint a, uint b) {
    a = in16(a); b = in16(b);
    if (!FX_NAN) { float x = hf(a); return out16(fmax(x, hf(b)) == x ? a : b); }
    if (FM_IEEE && snan16(a)) return quiet16(a);
    if (FM_IEEE && snan16(b)) return quiet16(b);
    if (nan16(a)) return nan16(b) ? a : out16(b);
    if (nan16(b)) return out16(a);
    int x = key16(a), y = key16(b);
    return out16(x > y ? a : y > x ? b : (a & 0x8000u) != 0u ? b : a);
}

static inline uint min16(uint a, uint b) {
    a = in16(a); b = in16(b);
    if (!FX_NAN) { float x = hf(a); return out16(fmin(x, hf(b)) == x ? a : b); }
    if (FM_IEEE && snan16(a)) return quiet16(a);
    if (FM_IEEE && snan16(b)) return quiet16(b);
    if (nan16(a)) return nan16(b) ? a : out16(b);
    if (nan16(b)) return out16(a);
    int x = key16(a), y = key16(b);
    return out16(x < y ? a : y < x ? b : (a & 0x8000u) != 0u ? a : b);
}

static inline uint med3_16(uint a, uint b, uint c) {
    a = in16(a); b = in16(b); c = in16(c);
    if (!FX_NAN) return max16(min16(a, b), min16(max16(a, b), c));
    if (nan16(a) || nan16(b) || nan16(c)) return min16(min16(a, b), c);
    uint m = max16(max16(a, b), c);
    if (m == a) return max16(b, c);
    if (m == b) return max16(a, c);
    return max16(a, b);
}

// A one-source f16 op computed in f32, rounded to nearest.
static inline uint op1_16(float r, uint a) {
    return FX_NAN && isnan(r) ? nan_of16(a, 0u, 0u) : h16(r, 0.0f, 0u, (FM_DN16 & 2) != 0);
}

static inline uint floor16(uint a) { a = in16(a); return op1_16(floor(hf(a)), a); }
static inline uint ceil16(uint a) { a = in16(a); return op1_16(ceil(hf(a)), a); }
static inline uint trunc16(uint a) { a = in16(a); return op1_16(trunc(hf(a)), a); }
static inline uint rndne16(uint a) { a = in16(a); return op1_16(rint(hf(a)), a); }
static inline uint rcp16(uint a) { a = in16(a); return op1_16(1.0f / hf(a), a); }
static inline uint sqrt16(uint a) { a = in16(a); return op1_16(sqrt(hf(a)), a); }
static inline uint rsq16(uint a) { a = in16(a); return op1_16(rsqrt(hf(a)), a); }
static inline uint exp16(uint a) { a = in16(a); return op1_16(exp2(hf(a)), a); }
static inline uint log16(uint a) { a = in16(a); return op1_16(log2(hf(a)), a); }

static inline uint sin16(uint a) {
    a = in16(a);
    float x = hf(a);
    if (zero16(a)) return a;
    float r = isinf(x) ? NAN : sinpi(2.0f * (x - rint(x)));
    return op1_16(r == 0.0f ? 0.0f : r, a);
}

static inline uint cos16(uint a) {
    a = in16(a);
    float x = hf(a);
    return op1_16(isinf(x) ? NAN : cospi(2.0f * (x - rint(x))), a);
}

static inline uint fract16(uint a) {
    a = in16(a);
    float x = hf(a);
    if (nan16(a)) return quiet16(a);
    if (isinf(x)) return 0xfe00u;
    return min(h16(x - floor(x), 0.0f, FM_RND16, (FM_DN16 & 2) != 0), 0x3bffu);
}

static inline uint omod16(uint r, float k) {
    if (!FX_MODS) return h16(hf(r) * k, 0.0f, FM_RND16, (FM_DN16 & 2) != 0);
    if (FM_IEEE || (FM_DN16 & 2) != 0 || nan16(r)) return r;
    uint x = h16(hf(r) * k, 0.0f, FM_RND16, false);
    return zero16(x) ? 0u : x;
}

static inline uint clamp16(uint r) {
    if (!FX_MODS) return nan16(r) || (r & 0x8000u) != 0u ? 0u : min(r, 0x3c00u);
    if (nan16(r)) return FM_DX10 ? 0u : r;
    if ((r & 0x8000u) != 0u) return 0u;
    return min(r, 0x3c00u);
}

// An f16 NaN widened: sign, quiet bit and payload move up.
static inline uint cvt_f32_f16(uint a) {
    a = in16(a);
    if (nan16(a)) return quiet32((a & 0x8000u) << 16 | 0x7f800000u | (a & 0x3ffu) << 13);
    return out32(as_type<uint>(hf(a)));
}

// An f32 NaN narrowed, quieted: sign and the top of the payload move down.
static inline uint nan_f16_of32(uint a) {
    return (a >> 16) & 0x8000u | 0x7e00u | (a >> 13 & 0x3ffu);
}


static inline uint cvt_f16_f32(uint a, uint m) {
    a = in32(a);
    if (nan32(a)) return nan_f16_of32(a);
    return h16(F(a), 0.0f, m, (FM_DN16 & 2) != 0);
}

static inline bool cmp_code(bool un, bool lt, bool eq, bool gt, uint c) {
    switch (c) {
    case 0u: return false;
    case 1u: return lt;
    case 2u: return eq;
    case 3u: return lt || eq;
    case 4u: return gt;
    case 5u: return lt || gt;
    case 6u: return gt || eq;
    case 7u: return !un;
    case 8u: return un;
    case 9u: return !(gt || eq);
    case 10u: return !(lt || gt);
    case 11u: return !gt;
    case 12u: return !(lt || eq);
    case 13u: return !eq;
    case 14u: return !lt;
    default: return true;
    }
}

static inline bool cmp32(uint a, uint b, uint c) {
    if (!FX_DN) {
        float x = F(a), y = F(b);
        bool un = isnan(x) || isnan(y);
        return cmp_code(un, x < y, x == y, x > y, c);
    }
    a = in32(a); b = in32(b);
    bool un = nan32(a) || nan32(b);
    int x = key32(a), y = key32(b);
    return cmp_code(un, !un && x < y, !un && x == y, !un && x > y, c);
}

static inline bool cmp16(uint a, uint b, uint c) {
    a = in16(a); b = in16(b);
    bool un = nan16(a) || nan16(b);
    int x = key16(a), y = key16(b);
    return cmp_code(un, !un && x < y, !un && x == y, !un && x > y, c);
}

// v_cmp_class masks: signalling NaN, quiet NaN, -inf, -normal, -denormal,
// -0, +0, +denormal, +normal, +inf.
static inline bool class32(uint b, uint mask) {
    bool neg = (b >> 31) != 0u;
    uint e = (b >> 23) & 0xffu, f = b & 0x7fffffu;
    uint k;
    if (e == 0xffu) k = f != 0u ? ((f >> 22) != 0u ? 1u : 0u) : (neg ? 2u : 9u);
    else if (e == 0u) k = f != 0u ? (neg ? 4u : 7u) : (neg ? 5u : 6u);
    else k = neg ? 3u : 8u;
    return (mask >> k & 1u) != 0u;
}

static inline bool class16(uint b, uint mask) {
    bool neg = (b & 0x8000u) != 0u;
    uint e = (b >> 10) & 0x1fu, f = b & 0x3ffu;
    uint k;
    if (e == 0x1fu) k = f != 0u ? ((f >> 9) != 0u ? 1u : 0u) : (neg ? 2u : 9u);
    else if (e == 0u) k = f != 0u ? (neg ? 4u : 7u) : (neg ? 5u : 6u);
    else k = neg ? 3u : 8u;
    return (mask >> k & 1u) != 0u;
}

// An f16 source of v_fma_mix as f32 bits; NaNs keep their bits.
static inline uint widen16(uint a) {
    a = in16(a);
    if (nan16(a)) return (a & 0x8000u) << 16 | 0x7f800000u | (a & 0x3ffu) << 13;
    return as_type<uint>(hf(a));
}

// v_fma_mixlo/hi_f16: f32 sources, one rounding to f16.
static inline uint fma_mix16(uint a, uint b, uint c) {
    float x = F(a), y = F(b), z = F(c), r = fma(x, y, z);
    if (isnan(r)) return nan_f16_of32(nan_fma32(a, b, c));
    float e = fma_err(x, y, z, r);
    return h16(zero_sign(r, e, pos0(prod0(a, b), z), FM_RND16), e, FM_RND16, (FM_DN16 & 2) != 0);

}

// frexp: the exponent and significand of x = significand * 2^exponent,
// significand in [0.5, 1); infinities and NaNs give exponent 0 and
// themselves.
static inline uint frexp_exp32(uint a) {
    a = in32(a);
    uint e = (a >> 23) & 0xffu, m = a & 0x7fffffu;
    if (e == 0xffu || zero32(a)) return 0u;
    if (e == 0u) return uint(int(31u - clz(m)) - 148);
    return uint(int(e) - 126);
}

static inline uint frexp_mant32(uint a) {
    a = in32(a);
    uint e = (a >> 23) & 0xffu, m = a & 0x7fffffu;
    if (e == 0xffu) return nan32(a) ? quiet32(a) : a;
    if (zero32(a)) return a;
    if (e == 0u) m = (m << (clz(m) - 8u)) & 0x7fffffu;
    return (a & 0x80000000u) | 0x3f000000u | m;
}

static inline uint frexp_exp16(uint a) {
    a = in16(a);
    uint e = (a >> 10) & 0x1fu, m = a & 0x3ffu;
    if (e == 0x1fu || zero16(a)) return 0u;
    if (e == 0u) return uint(int(31u - clz(m)) - 23) & 0xffffu;
    return uint(int(e) - 14) & 0xffffu;
}

static inline uint frexp_mant16(uint a) {
    a = in16(a);
    uint e = (a >> 10) & 0x1fu, m = a & 0x3ffu;
    if (e == 0x1fu) return nan16(a) ? quiet16(a) : a;
    if (zero16(a)) return a;
    if (e == 0u) m = (m << (clz(m) - 21u)) & 0x3ffu;
    return (a & 0x8000u) | 0x3800u | m;
}

// v_div_scale_f32: s0 (the denominator s1 or the numerator s2) scaled so
// that the division's Newton-Raphson steps meet no denormals; vcc says
// v_div_fmas_f32 must scale the quotient back.
static inline uint div_scale32(uint s0, uint s1, uint s2, thread bool& vcc) {
    s0 = in32(s0); s1 = in32(s1); s2 = in32(s2);
    // As RDNA2 hardware does it, which differs from the ISA guide: the steps
    // look only at exponent fields, so NaNs are scaled like other values and
    // a zero source changes only the result, a huge denominator is scaled
    // down, and a quotient counts as denormal when the exponents differ by
    // 96 or more (seen at 99, not at 94; 96 mirrors the overflow test).

    int e1 = int((s1 >> 23) & 0xffu), e2 = int((s2 >> 23) & 0xffu);
    bool rcp_den = e1 >= 253, quot_den = e2 - e1 <= -96;
    uint r = s0;
    vcc = false;
    if (e2 - e1 >= 96) {
        vcc = true;
        if (s0 == s1) r = ldexp32(s0, 64);
    } else if (den32(s1)) {
        r = ldexp32(s0, 64);
    } else if (rcp_den && quot_den) {
        vcc = true;
        if (s0 == s1) r = ldexp32(s0, -64);
    } else if (rcp_den) {
        r = ldexp32(s0, -64);
    } else if (quot_den) {
        vcc = true;
        if (s0 != s1) r = ldexp32(s0, 64);
    } else if (e2 <= 23) {
        r = ldexp32(s0, 64);
    }
    return zero32(s1) || zero32(s2) ? 0xffc00000u : r;
}

// v_div_fmas_f32: an fma, scaled when vcc is set by 2^64 if c is 2 or more
// in magnitude, else by 2^-64 (RDNA2 hardware; the ISA guide says 2^32).
static inline uint div_fmas32(uint a, uint b, uint c, bool vcc) {
    float x = F(a), y = F(b), z = F(c);
    int k = vcc ? (((c >> 23) & 0xffu) >= 128u ? 64 : -64) : 0;
    if (!isfinite(x) || !isfinite(y) || !isfinite(z)) {
        uint r = fma32(den32(a) ? (a & 0x80000000u) | 0x800000u : a, den32(b) ? (b & 0x80000000u) | 0x800000u : b, den32(c) ? (c & 0x80000000u) | 0x800000u : c);

        return vcc && !nan32(r) && !zero32(r) ? ldexp32(r, k) : r;
    }
    bool na, nb, nc; int ea, eb, ec; ulong ma, mb, mc;
    unpack32(a, na, ea, ma); unpack32(b, nb, eb, mb); unpack32(c, nc, ec, mc);
    return soft_sum(na != nb, ea + eb + k, ma * mb, nc, ec + k, mc, FM_RND32, (FM_DN32 & 2) != 0);
}

// v_div_fixup_f32: s0 the quotient, s1 the denominator, s2 the numerator.
// The quotient s0 is not flushed; an infinite or NaN one is an overflow.
static inline uint div_fixup32(uint s0, uint s1, uint s2) {
    s1 = in32(s1); s2 = in32(s2);
    uint sign = (s1 ^ s2) & 0x80000000u;
    uint a1 = s1 & 0x7fffffffu, a2 = s2 & 0x7fffffffu;
    if (nan32(s2)) return quiet32(s2);
    if (nan32(s1)) return quiet32(s1);
    if (a1 == 0u && a2 == 0u) return 0xffc00000u;
    if (a1 == 0x7f800000u && a2 == 0x7f800000u) return 0xffc00000u;
    if (a1 == 0u || a2 == 0x7f800000u) return sign | 0x7f800000u;
    if (a1 == 0x7f800000u || a2 == 0u) return sign;
    if (int(a2 >> 23) - int(a1 >> 23) < -150) return sign;
    if ((s0 & 0x7f800000u) == 0x7f800000u) return as_type<uint>(round_ovf(as_type<float>(sign | 0x7f800000u), FM_RND32));
    return sign | (s0 & 0x7fffffffu);
}

static inline uint div_fixup16(uint s0, uint s1, uint s2) {
    s0 &= 0xffffu; s1 = in16(s1); s2 = in16(s2);
    uint sign = (s1 ^ s2) & 0x8000u;
    uint a1 = s1 & 0x7fffu, a2 = s2 & 0x7fffu;
    if (nan16(s2)) return quiet16(s2);
    if (nan16(s1)) return quiet16(s1);
    if (a1 == 0u && a2 == 0u) return 0xfe00u;
    if (a1 == 0x7c00u && a2 == 0x7c00u) return 0xfe00u;
    if (a1 == 0u || a2 == 0x7c00u) return sign | 0x7c00u;
    if (a1 == 0x7c00u || a2 == 0u) return sign;
    if ((s0 & 0x7c00u) == 0x7c00u) {
        bool toward = FM_RND16 == 3u || (FM_RND16 == 1u && sign != 0u) || (FM_RND16 == 2u && sign == 0u);
        return sign | (toward ? 0x7bffu : 0x7c00u);
    }
    return sign | (s0 & 0x7fffu);
}

// x rounded to an integer under mode m.
static inline float round_int(float x, uint m) {
    return m == 0u ? rint(x) : m == 1u ? ceil(x) : m == 2u ? floor(x) : trunc(x);
}

// s0 to an unsigned byte (rounded under the mode, saturated, NaN 0) into
// byte s1 of s2.
static inline uint cvt_pk_u8_f32(uint s0, uint s1, uint s2) {
    s0 = in32(s0);
    float x = F(s0);
    uint b = nan32(s0) ? 0u : uint(round_int(clamp(x, 0.0f, 255.0f), FM_RND32));

    uint sh = 8u * (s1 & 3u);
    return (s2 & ~(0xffu << sh)) | (b << sh);
}

static inline uint cvt_i16_f16(uint a) {
    a = in16(a);
    if (nan16(a)) return 0u;
    return uint(int(clamp(trunc(hf(a)), -32768.0f, 32767.0f))) & 0xffffu;
}

static inline uint cvt_u16_f16(uint a) {
    a = in16(a);
    if (nan16(a)) return 0u;
    return uint(clamp(trunc(hf(a)), 0.0f, 65535.0f));
}

// x * k rounded to the nearest integer, ties to even, from the exact product.
static inline float rint_mul(float x, float k) {
    float p = x * k, e = fma(x, k, -p), r = rint(p);
    if (abs(p - trunc(p)) == 0.5f && e != 0.0f) r = e > 0.0f ? ceil(p) : floor(p);
    return r;
}

static inline uint cvt_snorm16_f16(uint a) {
    a = in16(a);
    if (nan16(a)) return 0u;
    return uint(int(rint_mul(clamp(hf(a), -1.0f, 1.0f), 32767.0f))) & 0xffffu;
}

static inline uint cvt_unorm16_f16(uint a) {
    a = in16(a);
    if (nan16(a)) return 0u;
    return uint(rint_mul(clamp(hf(a), 0.0f, 1.0f), 65535.0f));
}


// v_pack_b32_f16 moves each half as a float: flushed and quieted by the
// mode.
static inline uint pack16(uint a) {
    a = in16(a);
    return nan16(a) ? quiet16(a) : out16(a);
}

// v_mullit_f32, for LIT (RDNA2 hardware): -FLT_MAX (whose exp is 0) unless
// c is above zero and b is a number above -FLT_MAX, else a * b by the
// legacy rules.




static inline uint mullit32(uint a, uint b, uint c) {
    a = in32(a); b = in32(b); c = in32(c);
    bool c_pos = (c >> 31) == 0u && !zero32(c) && !nan32(c);
    if (!c_pos || b >= 0xff7fffffu || nan32(b)) return 0xff7fffffu;

    return mul_legacy32(a, b);
}
