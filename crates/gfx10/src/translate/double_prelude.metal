// f64 VALU semantics. Metal has no doubles, so an f64 is its bit pattern in
// a ulong and every operation is integer arithmetic on the significand,
// rounded once. f64 is always exact, whatever Request::plain_float says:
//   FD_RND: round mode of f64 (FLOAT_MODE bits 2-3, shared with f16);
//   FD_DN: f64 denormals (bit 0 inputs kept, bit 1 results kept);
//   FD_RND32: the f32 round mode, for conversions to f32.

static inline bool nan64(ulong x) { return (x & 0x7ffffffffffffffful) > 0x7ff0000000000000ul; }
static inline bool snan64(ulong x) { return nan64(x) && (x & 0x8000000000000ul) == 0ul; }
static inline bool inf64(ulong x) { return (x & 0x7ffffffffffffffful) == 0x7ff0000000000000ul; }
static inline bool den64(ulong x) { return (x & 0x7ff0000000000000ul) == 0ul && (x & 0xffffffffffffful) != 0ul; }
static inline bool zero64(ulong x) { return (x & 0x7ffffffffffffffful) == 0ul; }
static inline ulong flush64(ulong x) { return den64(x) ? x & 0x8000000000000000ul : x; }
static inline ulong in64(ulong x) { return (FD_DN & 1) == 0 ? flush64(x) : x; }
static inline ulong out64(ulong x) { return (FD_DN & 2) == 0 ? flush64(x) : x; }
static inline ulong quiet64(ulong x) { return FM_IEEE ? x | 0x8000000000000ul : x; }
static inline ulong sign64(bool neg) { return neg ? 0x8000000000000000ul : 0ul; }

// The NaN an instruction returns: its first NaN source, or the default NaN.
static inline ulong nan_of64(ulong a, ulong b, ulong c) {
    return nan64(a) ? quiet64(a) : nan64(b) ? quiet64(b) : nan64(c) ? quiet64(c) : 0xfff8000000000000ul;
}

// An order-preserving integer for a non-NaN double; both zeros map to 0.
static inline long key64(ulong x) { return (x >> 63) != 0ul ? -long(x & 0x7ffffffffffffffful) : long(x); }

// A finite nonzero double is (-1)^neg m 2^e with m an integer.
static inline void unpack64(ulong x, thread bool& neg, thread int& e, thread ulong& m) {
    neg = (x >> 63) != 0ul;
    int ex = int((x >> 52) & 0x7fful);
    m = ex == 0 ? x & 0xffffffffffffful : (x & 0xffffffffffffful) | 0x10000000000000ul;
    e = (ex == 0 ? 1 : ex) - 1075;
}

// (-1)^neg m 2^e, plus something below m's last bit if sticky, as f64 bits
// rounded under mode md; results below 2^-1022 before rounding are flushed
// unless keep.
static inline ulong pack64(bool neg, int e, ulong m, bool sticky, uint md, bool keep) {
    ulong s = sign64(neg);
    if (m == 0ul) return s;
    int top = 63 - int(clz(m)) + e;
    bool toward = md == 3u || (md == 1u && neg) || (md == 2u && !neg);
    if (top < -1022 && !keep) return s;
    if (top > 1023) return s | (toward ? 0x7feffffffffffffful : 0x7ff0000000000000ul);
    int lsb = max(top - 52, -1074);
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
    if (md == 0u) {
        if (rem > mid || (rem == mid && (sticky || (q & 1ul) != 0ul))) q += 1ul;
    } else if (away && (rem != 0ul || sticky)) {
        q += 1ul;
    }
    ulong bits = (ulong(lsb + 1074) << 52) + q;
    if (bits >= 0x7ff0000000000000ul) bits = toward ? 0x7feffffffffffffful : 0x7ff0000000000000ul;
    return s | bits;
}

// 128-bit significands are (h, l) pairs. narrow64 keeps the top 64 bits of
// a nonzero one, adding what it drops to e and sticky.
static inline ulong narrow64(ulong h, ulong l, thread int& e, thread bool& sticky) {
    if (h == 0ul) return l;
    uint n = 64u - clz(h);
    sticky = sticky || (l << (64u - n)) != 0ul;
    e += int(n);
    return n == 64u ? h : (h << (64u - n)) | (l >> n);
}

static inline void shl128(thread ulong& h, thread ulong& l, uint s) {
    if (s == 0u) return;
    if (s >= 64u) {
        h = l << (s - 64u);
        l = 0ul;
    } else {
        h = (h << s) | (l >> (64u - s));
        l <<= s;
    }
}

static inline void shr128(thread ulong& h, thread ulong& l, uint s, thread bool& sticky) {
    if (s == 0u) return;
    if (s >= 128u) {
        sticky = sticky || (h | l) != 0ul;
        h = 0ul;
        l = 0ul;
    } else if (s >= 64u) {
        sticky = sticky || l != 0ul || (s > 64u && (h << (128u - s)) != 0ul);
        l = h >> (s - 64u);
        h = 0ul;
    } else {
        sticky = sticky || (l << (64u - s)) != 0ul;
        l = (l >> s) | (h << (64u - s));
        h >>= s;
    }
}

static inline uint clz128(ulong h, ulong l) { return h != 0ul ? clz(h) : 64u + clz(l); }

// (-1)^na (ha, la) 2^ea + (-1)^nb (hb, lb) 2^eb, rounded once. The sum is
// exact before rounding: both are moved up to bit 125 and the smaller one's
// lost bits only make it sticky.
static inline ulong sum64(bool na, int ea, ulong ha, ulong la, bool nb, int eb, ulong hb, ulong lb, uint md, bool keep) {
    bool za = (ha | la) == 0ul, zb = (hb | lb) == 0ul;
    bool st = false;
    if (za && zb) return sign64((na && nb) || (na != nb && md == 2u));
    if (za) { ulong m = narrow64(hb, lb, eb, st); return pack64(nb, eb, m, st, md, keep); }
    if (zb) { ulong m = narrow64(ha, la, ea, st); return pack64(na, ea, m, st, md, keep); }
    uint sa = clz128(ha, la) - 2u, sb = clz128(hb, lb) - 2u;
    shl128(ha, la, sa); ea -= int(sa);
    shl128(hb, lb, sb); eb -= int(sb);
    if (ea < eb || (ea == eb && (hb > ha || (hb == ha && lb > la)))) {
        bool tn = na; na = nb; nb = tn;
        int te = ea; ea = eb; eb = te;
        ulong t = ha; ha = hb; hb = t;
        t = la; la = lb; lb = t;
    }
    shr128(hb, lb, uint(min(ea - eb, 128)), st);
    ulong h, l;
    if (na == nb) {
        l = la + lb;
        h = ha + hb + (l < la ? 1ul : 0ul);
    } else {
        ulong d = st ? 1ul : 0ul;
        ulong l1 = la - lb;
        ulong b1 = la < lb ? 1ul : 0ul;
        l = l1 - d;
        ulong b2 = l1 < d ? 1ul : 0ul;
        h = ha - hb - b1 - b2;
    }
    if ((h | l) == 0ul && !st) return sign64(md == 2u);
    ulong m = narrow64(h, l, ea, st);
    return pack64(na, ea, m, st, md, keep);
}

static inline ulong add64(ulong a, ulong b) {
    a = in64(a); b = in64(b);
    if (nan64(a) || nan64(b)) return nan_of64(a, b, 0ul);
    if (inf64(a) && inf64(b) && a != b) return 0xfff8000000000000ul;
    if (inf64(a)) return a;
    if (inf64(b)) return b;
    bool na, nb; int ea, eb; ulong ma, mb;
    unpack64(a, na, ea, ma); unpack64(b, nb, eb, mb);
    return sum64(na, ea, 0ul, ma, nb, eb, 0ul, mb, FD_RND, (FD_DN & 2) != 0);
}

static inline ulong mul64(ulong a, ulong b) {
    a = in64(a); b = in64(b);
    if (nan64(a) || nan64(b)) return nan_of64(a, b, 0ul);
    ulong s = (a ^ b) & 0x8000000000000000ul;
    if ((inf64(a) && zero64(b)) || (zero64(a) && inf64(b))) return 0xfff8000000000000ul;
    if (inf64(a) || inf64(b)) return s | 0x7ff0000000000000ul;
    bool na, nb; int ea, eb; ulong ma, mb;
    unpack64(a, na, ea, ma); unpack64(b, nb, eb, mb);
    bool st = false;
    int e = ea + eb;
    ulong m = narrow64(mulhi(ma, mb), ma * mb, e, st);
    return pack64(na != nb, e, m, st, FD_RND, (FD_DN & 2) != 0);
}

// a * b + c with the product exact, scaled by 2^k, rounded once.
static inline ulong fma_scaled64(ulong a, ulong b, ulong c, int k) {
    if (nan64(a) || nan64(b)) return nan_of64(a, b, 0ul);
    bool bad = (inf64(a) && zero64(b)) || (zero64(a) && inf64(b));
    if (bad) return 0xfff8000000000000ul;
    if (nan64(c)) return quiet64(c);
    ulong ps = (a ^ b) & 0x8000000000000000ul;
    if (inf64(a) || inf64(b)) return inf64(c) && (c & 0x8000000000000000ul) != ps ? 0xfff8000000000000ul : ps | 0x7ff0000000000000ul;
    if (inf64(c)) return c;
    bool na, nb, nc; int ea, eb, ec; ulong ma, mb, mc;
    unpack64(a, na, ea, ma); unpack64(b, nb, eb, mb); unpack64(c, nc, ec, mc);
    return sum64(na != nb, ea + eb + k, mulhi(ma, mb), ma * mb, nc, ec + k, 0ul, mc, FD_RND, (FD_DN & 2) != 0);
}

static inline ulong fma64(ulong a, ulong b, ulong c) {
    return fma_scaled64(in64(a), in64(b), in64(c), 0);
}

static inline ulong max64(ulong a, ulong b) {
    a = in64(a); b = in64(b);
    if (FM_IEEE && snan64(a)) return quiet64(a);
    if (FM_IEEE && snan64(b)) return quiet64(b);
    if (nan64(a)) return nan64(b) ? a : out64(b);
    if (nan64(b)) return out64(a);
    long x = key64(a), y = key64(b);
    return out64(x > y ? a : y > x ? b : (a >> 63) != 0ul ? b : a);
}

static inline ulong min64(ulong a, ulong b) {
    a = in64(a); b = in64(b);
    if (FM_IEEE && snan64(a)) return quiet64(a);
    if (FM_IEEE && snan64(b)) return quiet64(b);
    if (nan64(a)) return nan64(b) ? a : out64(b);
    if (nan64(b)) return out64(a);
    long x = key64(a), y = key64(b);
    return out64(x < y ? a : y < x ? b : (a >> 63) != 0ul ? a : b);
}

static inline ulong ldexp64(ulong a, int n) {
    a = in64(a);
    if (nan64(a)) return quiet64(a);
    if (inf64(a) || zero64(a)) return a;
    bool neg; int e; ulong m;
    unpack64(a, neg, e, m);
    return pack64(neg, e + clamp(n, -2200, 2200), m, false, FD_RND, (FD_DN & 2) != 0);
}

// a rounded to an integer: up rounds toward +inf, down toward -inf, both
// toward zero, neither to nearest even.
static inline ulong round64(ulong a, bool up, bool down) {
    a = in64(a);
    if (nan64(a)) return quiet64(a);
    int ex = int((a >> 52) & 0x7fful) - 1023;
    if (ex >= 52) return a;
    bool neg = (a >> 63) != 0ul;
    ulong s = a & 0x8000000000000000ul;
    ulong mag = a & 0x7ffffffffffffffful;
    bool away;
    ulong r;
    if (ex < 0) {
        if (mag == 0ul) return a;
        if (up && down) away = false;
        else if (up) away = !neg;
        else if (down) away = neg;
        else away = ex == -1 && mag > 0x3fe0000000000000ul;
        return s | (away ? 0x3ff0000000000000ul : 0ul);
    }
    ulong frac = (1ul << uint(52 - ex)) - 1ul;
    if ((mag & frac) == 0ul) return a;
    r = mag & ~frac;
    if (up && down) away = false;
    else if (up) away = !neg;
    else if (down) away = neg;
    else {
        ulong mid = 1ul << uint(51 - ex);
        ulong rem = mag & frac;
        away = rem > mid || (rem == mid && ((mag >> uint(52 - ex)) & 1ul) != 0ul);
    }
    return s | (away ? r + (frac + 1ul) : r);
}

static inline ulong trunc64(ulong a) { return round64(a, true, true); }
static inline ulong ceil64(ulong a) { return round64(a, true, false); }
static inline ulong floor64(ulong a) { return round64(a, false, true); }
static inline ulong rndne64(ulong a) { return round64(a, false, false); }

// x - floor(x), rounded under the mode, below 1.
static inline ulong fract64(ulong a) {
    a = in64(a);
    if (nan64(a)) return quiet64(a);
    if (inf64(a)) return 0xfff8000000000000ul;
    ulong r = add64(a, floor64(a) ^ 0x8000000000000000ul);
    return min(r & 0x7ffffffffffffffful, 0x3feffffffffffffful);
}

static inline uint frexp_exp64(ulong a) {
    a = in64(a);
    int ex = int((a >> 52) & 0x7fful);
    if (ex == 0x7ff || zero64(a)) return 0u;
    if (ex == 0) return uint(int(63u - clz(a & 0xffffffffffffful)) - 1073);
    return uint(ex - 1022);
}

static inline ulong frexp_mant64(ulong a) {
    a = in64(a);
    int ex = int((a >> 52) & 0x7fful);
    ulong m = a & 0xffffffffffffful;
    if (ex == 0x7ff) return nan64(a) ? quiet64(a) : a;
    if (zero64(a)) return a;
    if (ex == 0) m = (m << (clz(m) - 11u)) & 0xffffffffffffful;
    return (a & 0x8000000000000000ul) | 0x3fe0000000000000ul | m;
}

static inline ulong cvt_f64_i32(int i) {
    if (i == 0) return 0ul;
    ulong m = ulong(abs(long(i)));
    return pack64(i < 0, 0, m, false, 0u, true);
}

static inline ulong cvt_f64_u32(uint i) {
    return i == 0u ? 0ul : pack64(false, 0, ulong(i), false, 0u, true);
}

// The integer part of a, saturated to [lo, hi]; NaN gives 0.
static inline long trunc_int64(ulong a, long lo, long hi) {
    a = in64(a);
    if (nan64(a)) return 0;
    int ex = int((a >> 52) & 0x7fful) - 1075;
    bool neg = (a >> 63) != 0ul;
    if ((a & 0x7ff0000000000000ul) == 0ul) return 0;
    if (ex >= 11) return neg ? lo : hi;
    ulong m = (a & 0xffffffffffffful) | 0x10000000000000ul;
    long v = ex >= 0 ? long(m << uint(ex)) : ex <= -64 ? 0 : long(m >> uint(-ex));
    v = neg ? -v : v;
    return clamp(v, lo, hi);
}

static inline uint cvt_i32_f64(ulong a) { return uint(int(trunc_int64(a, -2147483648l, 2147483647l))); }
static inline uint cvt_u32_f64(ulong a) { return uint(trunc_int64(a, 0l, 4294967295l)); }

// f32 bits widened: exact; denormal f32 inputs follow the f32 mode.
static inline ulong cvt_f64_f32(uint a) {
    a = (FM_DN32 & 1) == 0 && (a & 0x7f800000u) == 0u ? a & 0x80000000u : a;
    ulong s = ulong(a & 0x80000000u) << 32;
    uint ex = (a >> 23) & 0xffu, f = a & 0x7fffffu;
    if (ex == 0xffu) {
        ulong x = s | 0x7ff0000000000000ul | ulong(f) << 29;
        return f != 0u ? quiet64(x) : x;
    }
    if (ex == 0u && f == 0u) return s;
    if (ex == 0u) return pack64(s != 0ul, -149, ulong(f), false, 0u, true);
    return s | ulong(ex + 896u) << 52 | ulong(f) << 29;
}

// Narrowed to f32 under the f32 mode.
static inline uint cvt_f32_f64(ulong a) {
    a = in64(a);
    uint s = uint(a >> 32) & 0x80000000u;
    if (nan64(a)) {
        uint x = s | 0x7f800000u | uint(a >> 29) & 0x7fffffu;
        return FM_IEEE || (x & 0x7fffffu) == 0u ? x | 0x400000u : x;
    }
    if (inf64(a)) return s | 0x7f800000u;
    if (zero64(a)) return s;
    bool neg; int e; ulong m;
    unpack64(a, neg, e, m);
    return pack32(neg, e, m, false, FD_RND32, (FM_DN32 & 2) != 0);
}

static inline bool cmp64(ulong a, ulong b, uint c) {
    a = in64(a); b = in64(b);
    bool un = nan64(a) || nan64(b);
    long x = key64(a), y = key64(b);
    return cmp_code(un, !un && x < y, !un && x == y, !un && x > y, c);
}

static inline bool class64(ulong b, uint mask) {
    bool neg = (b >> 63) != 0ul;
    ulong e = (b >> 52) & 0x7fful, f = b & 0xffffffffffffful;
    uint k;
    if (e == 0x7fful) k = f != 0ul ? ((f >> 51) != 0ul ? 1u : 0u) : (neg ? 2u : 9u);
    else if (e == 0ul) k = f != 0ul ? (neg ? 4u : 7u) : (neg ? 5u : 6u);
    else k = neg ? 3u : 8u;
    return (mask >> k & 1u) != 0u;
}

// Output modifiers make -0 +0.
static inline ulong omod64(ulong r, int n) {
    if (FM_IEEE || (FD_DN & 2) != 0 || nan64(r)) return r;
    ulong x = flush64(ldexp64(r, n));
    return zero64(x) ? 0ul : x;
}

static inline ulong clamp64(ulong r) {
    if (nan64(r)) return FM_DX10 ? 0ul : r;
    if ((r >> 63) != 0ul) return 0ul;
    return min(r, 0x3ff0000000000000ul);
}

// ldexp64 that leaves NaNs as they are.
static inline ulong scale64(ulong x, int n) { return nan64(x) ? x : ldexp64(x, n); }

// v_div_scale_f64, as div_scale32 with the f64 field widths.
static inline ulong div_scale64(ulong s0, ulong s1, ulong s2, thread bool& vcc) {
    s0 = in64(s0); s1 = in64(s1); s2 = in64(s2);
    int e1 = int((s1 >> 52) & 0x7fful), e2 = int((s2 >> 52) & 0x7fful);
    bool rcp_den = e1 >= 2045, quot_den = e2 - e1 <= -768;
    ulong r = s0;
    vcc = false;
    if (e2 - e1 >= 768) {
        vcc = true;
        if (s0 == s1) r = scale64(s0, 128);
    } else if (den64(s1)) {
        r = scale64(s0, 128);
    } else if (rcp_den && quot_den) {
        vcc = true;
        if (s0 == s1) r = scale64(s0, -128);
    } else if (rcp_den) {
        r = scale64(s0, -128);
    } else if (quot_den) {
        vcc = true;
        if (s0 != s1) r = scale64(s0, 128);
    } else if (e2 <= 53) {
        r = scale64(s0, 128);
    }
    return zero64(s1) || zero64(s2) ? 0xfff8000000000000ul : r;
}

// v_div_fmas_f64: an fma, scaled when vcc is set by 2^128 if c is 2 or more
// in magnitude, else by 2^-128.
static inline ulong div_fmas64(ulong a, ulong b, ulong c, bool vcc) {
    int k = vcc ? (((c >> 52) & 0x7fful) >= 1024ul ? 128 : -128) : 0;
    return fma_scaled64(a, b, c, k);
}

// v_div_fixup_f64: s0 the quotient, s1 the denominator, s2 the numerator.
static inline ulong div_fixup64(ulong s0, ulong s1, ulong s2) {
    s1 = in64(s1); s2 = in64(s2);
    ulong sign = (s1 ^ s2) & 0x8000000000000000ul;
    ulong a1 = s1 & 0x7ffffffffffffffful, a2 = s2 & 0x7ffffffffffffffful;
    if (nan64(s2)) return quiet64(s2);
    if (nan64(s1)) return quiet64(s1);
    if (a1 == 0ul && a2 == 0ul) return 0xfff8000000000000ul;
    if (a1 == 0x7ff0000000000000ul && a2 == 0x7ff0000000000000ul) return 0xfff8000000000000ul;
    if (a1 == 0ul || a2 == 0x7ff0000000000000ul) return sign | 0x7ff0000000000000ul;
    if (a1 == 0x7ff0000000000000ul || a2 == 0ul) return sign;
    bool toward = FD_RND == 3u || (FD_RND == 1u && sign != 0ul) || (FD_RND == 2u && sign == 0ul);
    bool away = (FD_RND == 1u && sign == 0ul) || (FD_RND == 2u && sign != 0ul);
    if (int(a2 >> 52) - int(a1 >> 52) < -1075) return sign | (away && (FD_DN & 2) != 0 ? 1ul : 0ul);
    if ((s0 & 0x7ff0000000000000ul) == 0x7ff0000000000000ul) return sign | (toward ? 0x7feffffffffffffful : 0x7ff0000000000000ul);
    return sign | (s0 & 0x7ffffffffffffffful);
}

// 64 bits of 2/pi from bit i on (bit 0 is the 2^-1 place). The table ends
// after 1183 bits, as RDNA2 hardware's does (the ISA guide says 1201).
static inline ulong two_over_pi64(int i) {
    const ulong w[19] = {
        0xa2f9836e4e441529ul, 0xfc2757d1f534ddc0ul, 0xdb6295993c439041ul, 0xfe5163abdebbc561ul,
        0xb7246e3a424dd2e0ul, 0x06492eea09d1921cul, 0xfe1deb1cb129a73eul, 0xe88235f52ebb4484ul,
        0xe99c7026b45f7e41ul, 0x3991d639835339f4ul, 0x9c845f8bbdf9283bul, 0x1ff897ffde05980ful,
        0xef2f118b5a0a6d1ful, 0x6d367ecf27cb09b7ul, 0x4f463f669e5fea2dul, 0x7527bac7ebe5f17bul,
        0x3d0739f78a5292eaul, 0x6bfb5fb11f8d5d08ul, 0x56033046fc7b6babul};
    if (i >= 1183) return 0ul;
    int k = i >> 6, s = i & 63;
    ulong hi = w[k], lo = k + 1 < 19 ? w[k + 1] : 0ul;
    ulong x = s == 0 ? hi : (hi << uint(s)) | (lo >> uint(64 - s));
    int left = 1183 - i;
    return left >= 64 ? x : x & ~((1ul << uint(64 - left)) - 1ul);
}

// v_trig_preop_f64: 53 bits of 2/pi starting at bit 53 * segment, past a
// shift that keeps the product with a large a exact.
static inline ulong trig_preop64(ulong a, uint seg) {
    a = in64(a);
    int ex = int((a >> 52) & 0x7fful);
    int shift = int(seg & 31u) * 53 + (ex > 1077 ? ex - 1077 : 0);
    ulong m = two_over_pi64(shift) >> 11;
    int scale = -53 - shift + (ex >= 1968 ? 128 : 0);
    return pack64(false, scale, m, false, 3u, (FD_DN & 2) != 0);
}

// floor(n 2^k / d) for n <= d, with sticky if inexact.
static inline ulong div_bits64(ulong n, ulong d, int k, thread bool& sticky) {
    ulong q = 0ul, r = n;
    for (int i = 0; i <= k; i++) {
        q <<= 1;
        if (r >= d) { r -= d; q |= 1ul; }
        r <<= 1;
    }
    sticky = sticky || r != 0ul;
    return q;
}

static inline ulong rcp64(ulong a) {
    a = in64(a);
    if (nan64(a)) return quiet64(a);
    ulong s = a & 0x8000000000000000ul;
    if (inf64(a)) return s;
    if (zero64(a)) return s | 0x7ff0000000000000ul;
    bool neg; int e; ulong m;
    unpack64(a, neg, e, m);
    uint z = clz(m) - 11u;
    m <<= z; e -= int(z);
    bool st = false;
    ulong q = div_bits64(1ul << 52, m, 62, st);
    return pack64(neg, -114 - e, q, st, FD_RND, (FD_DN & 2) != 0);
}

// floor(sqrt(m 2^66)) for m < 2^54, with sticky if inexact.
static inline ulong sqrt_bits64(ulong m, thread bool& sticky) {
    ulong root = 0ul, rem = 0ul;
    for (int i = 0; i < 60; i++) {
        int b = 52 - 2 * i;
        ulong two = b >= 0 ? (m >> uint(b)) & 3ul : 0ul;
        rem = (rem << 2) | two;
        ulong t = (root << 2) | 1ul;
        root <<= 1;
        if (rem >= t) { rem -= t; root |= 1ul; }
    }
    sticky = sticky || rem != 0ul;
    return root;
}

static inline ulong sqrt64(ulong a) {
    a = in64(a);
    if (nan64(a)) return quiet64(a);
    if (zero64(a)) return a;
    if ((a >> 63) != 0ul) return 0xfff8000000000000ul;
    if (inf64(a)) return a;
    bool neg; int e; ulong m;
    unpack64(a, neg, e, m);
    if ((e & 1) != 0) { m <<= 1; e -= 1; }
    bool st = false;
    ulong r = sqrt_bits64(m, st);
    return pack64(false, e / 2 - 33, r, st, FD_RND, (FD_DN & 2) != 0);
}

static inline ulong rsq64(ulong a) {
    a = in64(a);
    if (nan64(a)) return quiet64(a);
    if (zero64(a)) return (a & 0x8000000000000000ul) | 0x7ff0000000000000ul;
    if ((a >> 63) != 0ul) return 0xfff8000000000000ul;
    if (inf64(a)) return 0ul;
    return rcp64(sqrt64(a));
}
