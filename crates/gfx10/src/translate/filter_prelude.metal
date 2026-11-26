// Filtering in the program as RDNA2's texture unit does it: the sample
// position rounded to 8 fractional bits, bilinear weights of 8 bits.
// `sw` is the sampler's S#.

static inline bool filters8(uint4 sw) {
    uint mag = (sw.z >> 20) & 3u;
    uint minf = (sw.z >> 22) & 3u;
    uint mip = (sw.z >> 26) & 3u;
    uint aniso = (sw.x >> 9) & 7u;
    uint unnormalized = (sw.x >> 15) & 1u;
    return mag == 1u && minf == 1u && mip <= 1u && aniso == 0u && unnormalized == 0u;
}

// Texel index `i` of a row of `n` under CLAMP mode `mode`; `border` is set
// when the border colour stands in for the texel.
static inline int address8(int i, int n, uint mode, thread bool& border) {
    switch (mode) {
    case 0u:
        return ((i % n) + n) % n;
    case 1u: {
        int m = ((i % (2 * n)) + 2 * n) % (2 * n);
        return m < n ? m : 2 * n - 1 - m;
    }
    case 3u:
    case 5u:
    case 7u:
        return clamp(i < 0 ? -1 - i : i, 0, n - 1);
    case 6u:
        border = border || i < 0 || i >= n;
        return clamp(i, 0, n - 1);
    default:
        return clamp(i, 0, n - 1);
    }
}

static inline float4 texel8(texture2d<float> t, uint lvl, int2 p, int2 n, uint4 sw) {
    bool border = false;
    int x = address8(p.x, n.x, sw.x & 7u, border);
    int y = address8(p.y, n.y, (sw.x >> 3) & 7u, border);
    if (border) {
        uint type = sw.w >> 30;
        return type == 0u ? float4(0.0f) : type == 1u ? float4(0.0f, 0.0f, 0.0f, 1.0f) : float4(1.0f);
    }
    return t.read(uint2(x, y), lvl);
}

// sRGB decoded as the hardware decodes it: rounded to bfloat16.
static inline float bf16(float x) {
    uint b = as_type<uint>(x);
    return as_type<float>((b + 0x7FFFu + ((b >> 16) & 1u)) & 0xFFFF0000u);
}

// A bilinear sample at `uv` with `lod` (before the sampler's LOD bias), of
// a texture whose T# dword 1 is `tw1`. 8-bit unorm channels blend exactly
// in integers, as the hardware's results round.
static inline float4 sample8(texture2d<float> t, float2 uv, float lod, uint4 sw, uint tw1) {
    uint lvl = 0u;
    if (((sw.z >> 26) & 3u) == 1u) {
        float l = lod + float(int(sw.z << 18) >> 18) / 256.0f;
        lvl = uint(clamp(floor(l + 0.5f), 0.0f, float(t.get_num_mip_levels() - 1u)));
    }
    int2 n = int2(t.get_width(lvl), t.get_height(lvl));
    int2 q = int2(rint((uv * float2(n) - 0.5f) * 256.0f));
    int2 i = q >> 8;
    int2 f = q & 255;
    float4 a = texel8(t, lvl, i, n, sw);
    float4 b = texel8(t, lvl, i + int2(1, 0), n, sw);
    float4 c = texel8(t, lvl, i + int2(0, 1), n, sw);
    float4 d = texel8(t, lvl, i + int2(1, 1), n, sw);
    uint4 w = uint4((256 - f.x) * (256 - f.y), f.x * (256 - f.y), (256 - f.x) * f.y, f.x * f.y);
    uint fmt = (tw1 >> 20) & 0x1FFu;
    bool srgb = fmt >= 128u && fmt <= 130u;
    if (fmt == 1u || fmt == 14u || fmt == 56u || srgb) {
        uint4 sum = uint4(rint(a * 255.0f)) * w.x + uint4(rint(b * 255.0f)) * w.y
            + uint4(rint(c * 255.0f)) * w.z + uint4(rint(d * 255.0f)) * w.w;
        float4 r = precise::divide(float4(sum), float4(255.0f * 65536.0f));
        if (srgb) {
            float4 v = float4(w) / 65536.0f;
            for (uint k = 0u; k < 3u; k++) {
                r[k] = fma(bf16(d[k]), v.w, fma(bf16(c[k]), v.z, fma(bf16(b[k]), v.y, bf16(a[k]) * v.x)));
            }
        }
        return r;
    }
    float2 g = float2(f) / 256.0f;
    return mix(mix(a, b, g.x), mix(c, d, g.x), g.y);
}
