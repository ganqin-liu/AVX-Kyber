#include "poly8.h"

#include <string.h>

#define Q 3329
#define QINV (-3327) /* q^-1 mod 2^16 */
#define BARR 20159   /* round((1<<26)/q) */
#define FINV 1441    /* mont^2/128, ref invntt scaling */

/* ref zeta table (Montgomery domain), exported by the linked ref objects */
extern const int16_t pqcrystals_kyber768_ref_zetas[128];
#define zetas pqcrystals_kyber768_ref_zetas

static int16_t zetas_q[128]; /* zetas[i] * qinv mod 2^16 */
static int16_t finv_q;
static __m512i bm_z[64], bm_zq[64]; /* basemul [z,z,-z,-z] per window */
static int tables_ready;

void poly8_init_tables(void)
{
    int i;
    if (tables_ready)
        return;
    for (i = 0; i < 128; i++)
        zetas_q[i] = (int16_t)((int32_t)zetas[i] * QINV);
    finv_q = (int16_t)((int32_t)FINV * QINV);
    for (i = 0; i < 64; i++)
    {
        int16_t z = zetas[64 + i];
        int16_t zq = zetas_q[64 + i];
        uint64_t wz, wq;
        uint16_t uz = (uint16_t)z, unz = (uint16_t)(-z);
        uint16_t uzq = (uint16_t)zq, unzq = (uint16_t)(-zq);
        wz = (uint64_t)uz | ((uint64_t)uz << 16) | ((uint64_t)unz << 32) |
             ((uint64_t)unz << 48);
        wq = (uint64_t)uzq | ((uint64_t)uzq << 16) | ((uint64_t)unzq << 32) |
             ((uint64_t)unzq << 48);
        bm_z[i] = _mm512_set1_epi64((long long)wz);
        bm_zq[i] = _mm512_set1_epi64((long long)wq);
    }
    tables_ready = 1;
}

/* Scalar Montgomery multiply, for building operands. */
static int16_t fqmul_s(int16_t a, int16_t b)
{
    int32_t t = (int32_t)a * (int32_t)b;
    int16_t m = (int16_t)((int16_t)t * (int16_t)QINV);
    int32_t u = (int32_t)m * (int32_t)Q;
    return (int16_t)((t - u) >> 16);
}

/* ---------------- primitive vector ops ---------------- */

#define QV _mm512_set1_epi16(Q)

/* Montgomery mul by broadcast constant zeta (companion zq = z*qinv) */
static inline __m512i fqmul_bc(__m512i a, int16_t z, int16_t zq)
{
    __m512i m = _mm512_mullo_epi16(a, _mm512_set1_epi16(zq));
    __m512i hi = _mm512_mulhi_epi16(a, _mm512_set1_epi16(z));
    __m512i t = _mm512_mulhi_epi16(m, QV);
    return _mm512_sub_epi16(hi, t);
}

/* Montgomery mul by a full vector operand with precomputed companion */
static inline __m512i fqmul_pc(__m512i a, __m512i b, __m512i bq)
{
    __m512i m = _mm512_mullo_epi16(a, bq);
    __m512i hi = _mm512_mulhi_epi16(a, b);
    __m512i t = _mm512_mulhi_epi16(m, QV);
    return _mm512_sub_epi16(hi, t);
}

static inline __m512i barrett16(__m512i a)
{
    __m512i t = _mm512_mulhi_epi16(a, _mm512_set1_epi16(BARR));
    t = _mm512_srai_epi16(t, 10);
    t = _mm512_mullo_epi16(t, QV);
    return _mm512_sub_epi16(a, t);
}

/* forward butterfly: t = zeta*b; b = a-t; a = a+t */
#define BF(a, b, z, zq)                       \
    do                                        \
    {                                         \
        __m512i t_ = fqmul_bc((b), (z), (zq)); \
        (b) = _mm512_sub_epi16((a), t_);      \
        (a) = _mm512_add_epi16((a), t_);      \
    } while (0)

/* inverse butterfly: t=a; a = barrett(t+b); b = zeta*(b-t) */
#define IBF(a, b, z, zq)                                \
    do                                                  \
    {                                                   \
        __m512i s_ = _mm512_add_epi16((a), (b));        \
        __m512i d_ = _mm512_sub_epi16((b), (a));        \
        (a) = barrett16(s_);                            \
        (b) = fqmul_bc(d_, (z), (zq));                  \
    } while (0)

/* Lazy inverse sum; input ranges are documented in docs/invntt-bounds.md. */
#define IBF_LAZY(a, b, z, zq)                           \
    do                                                  \
    {                                                   \
        __m512i s_ = _mm512_add_epi16((a), (b));         \
        __m512i d_ = _mm512_sub_epi16((b), (a));         \
        (a) = s_;                                       \
        (b) = fqmul_bc(d_, (z), (zq));                   \
    } while (0)

/* len=2 layers act inside each 64-bit window [c0,c1,c2,c3]:
 * pairs (c0,c2),(c1,c3); 0xB1 swaps the two 32-bit halves per window. */
static inline __m512i bf_len2(__m512i x, int16_t z, int16_t zq)
{
    __m512i s = _mm512_shuffle_epi32(x, 0xB1); /* [c2,c3,c0,c1] */
    __m512i t = fqmul_bc(s, z, zq);
    __m512i lo = _mm512_add_epi16(x, t);       /* valid at c0,c1 */
    __m512i ts = _mm512_shuffle_epi32(t, 0xB1);
    __m512i hi = _mm512_sub_epi16(s, ts);      /* valid at c2,c3 */
    return _mm512_mask_blend_epi32(0xAAAA, lo, hi);
}

static inline __m512i ibf_len2(__m512i x, int16_t z, int16_t zq)
{
    __m512i s = _mm512_shuffle_epi32(x, 0xB1);
    __m512i sum = _mm512_add_epi16(x, s); /* valid at c0,c1; input scaled */
    __m512i d = _mm512_sub_epi16(x, s);              /* c2-c0 at c2 pos */
    __m512i fm = fqmul_bc(d, z, zq);
    return _mm512_mask_blend_epi32(0xAAAA, sum, fm);
}

/* ---------------- fold / unfold ---------------- */

/* 8x8 transpose of 64-bit lanes (same network as fips202x8; involution) */
static inline __attribute__((always_inline)) void transpose8x8_u64(__m512i r[8])
{
    __m512i t[8];
    int i;

    t[0] = _mm512_unpacklo_epi64(r[0], r[1]);
    t[1] = _mm512_unpackhi_epi64(r[0], r[1]);
    t[2] = _mm512_unpacklo_epi64(r[2], r[3]);
    t[3] = _mm512_unpackhi_epi64(r[2], r[3]);
    t[4] = _mm512_unpacklo_epi64(r[4], r[5]);
    t[5] = _mm512_unpackhi_epi64(r[4], r[5]);
    t[6] = _mm512_unpacklo_epi64(r[6], r[7]);
    t[7] = _mm512_unpackhi_epi64(r[6], r[7]);

    r[0] = _mm512_shuffle_i64x2(t[0], t[2], 0x88);
    r[1] = _mm512_shuffle_i64x2(t[1], t[3], 0x88);
    r[2] = _mm512_shuffle_i64x2(t[0], t[2], 0xdd);
    r[3] = _mm512_shuffle_i64x2(t[1], t[3], 0xdd);
    r[4] = _mm512_shuffle_i64x2(t[4], t[6], 0x88);
    r[5] = _mm512_shuffle_i64x2(t[5], t[7], 0x88);
    r[6] = _mm512_shuffle_i64x2(t[4], t[6], 0xdd);
    r[7] = _mm512_shuffle_i64x2(t[5], t[7], 0xdd);

    t[0] = _mm512_shuffle_i64x2(r[0], r[4], 0x88);
    t[1] = _mm512_shuffle_i64x2(r[1], r[5], 0x88);
    t[2] = _mm512_shuffle_i64x2(r[2], r[6], 0x88);
    t[3] = _mm512_shuffle_i64x2(r[3], r[7], 0x88);
    t[4] = _mm512_shuffle_i64x2(r[0], r[4], 0xdd);
    t[5] = _mm512_shuffle_i64x2(r[1], r[5], 0xdd);
    t[6] = _mm512_shuffle_i64x2(r[2], r[6], 0xdd);
    t[7] = _mm512_shuffle_i64x2(r[3], r[7], 0xdd);

    for (i = 0; i < 8; i++)
        r[i] = t[i];
}

void poly8_fold(poly8 *f, const int16_t *c[8])
{
    int g, l;
    for (g = 0; g < 8; g++)
    {
        __m512i r[8];
        for (l = 0; l < 8; l++)
            r[l] = _mm512_loadu_si512((const void *)(c[l] + 32 * g));
        transpose8x8_u64(r);
        for (l = 0; l < 8; l++)
            f->v[8 * g + l] = r[l];
    }
}

void poly8_unfold(int16_t *c[8], const poly8 *f)
{
    int g, l;
    for (g = 0; g < 8; g++)
    {
        __m512i r[8];
        for (l = 0; l < 8; l++)
            r[l] = f->v[8 * g + l];
        transpose8x8_u64(r);
        for (l = 0; l < 8; l++)
            _mm512_storeu_si512((void *)(c[l] + 32 * g), r[l]);
    }
}

void fop8_from_poly(fop8 *op, const int16_t c[256])
{
    fop8c c8;
    int v;

    /* Same pre-twiddling as the compact form (see fop8c_from_poly), just
     * broadcast into full vectors. */
    fop8c_from_poly(&c8, c);
    for (v = 0; v < 64; v++)
    {
        op->v[v] = _mm512_set1_epi64((long long)c8.v[v]);
        op->w[v] = _mm512_set1_epi64((long long)c8.w[v]);
    }
}

/* Folded operand from 8 DIFFERENT polys (heterogeneous-key case):
 * fold + pre-zeta and swapped forms. */
void fop8_from_polys(fop8 *op, const int16_t *c[8])
{
    int g, l;
    for (g = 0; g < 8; g++)
    {
        __m512i r[8];
        for (l = 0; l < 8; l++)
            r[l] = _mm512_loadu_si512((const void *)(c[l] + 32 * g));
        transpose8x8_u64(r);
        for (l = 0; l < 8; l++)
        {
            int vi = 8 * g + l;
            /* zt = [z*b0, z*b1, -z*b2, -z*b3]; keep only the odd lanes. */
            __m512i zt = fqmul_pc(r[l], bm_z[vi], bm_zq[vi]);
            __m512i b1 = _mm512_mask_blend_epi16(0xAAAAAAAAu, r[l], zt);
            __m512i b2 = _mm512_rol_epi32(r[l], 16);
            op->v[vi] = b1;
            op->w[vi] = b2;
        }
    }
}

/* ---------------- NTT ---------------- */

/* Pass A: layers len=128,64,32,16 on the 4 column subsets v = r (mod 4);
 * 16 register-resident vectors per subset. Local index i <-> v = r+4i. */
static void ntt_passA(poly8 *f)
{
    int r, i;
    for (r = 0; r < 4; r++)
    {
        __m512i x[16];
        for (i = 0; i < 16; i++)
            x[i] = f->v[r + 4 * i];

        /* len=128: pairs (i,i+8), one block */
        for (i = 0; i < 8; i++)
            BF(x[i], x[i + 8], zetas[1], zetas_q[1]);

        /* len=64: pairs (i,i+4); block = v/32 */
        for (i = 0; i < 4; i++)
            BF(x[i], x[i + 4], zetas[2], zetas_q[2]);
        for (i = 8; i < 12; i++)
            BF(x[i], x[i + 4], zetas[3], zetas_q[3]);

        /* len=32: pairs (i,i+2); block = v/16 */
        for (i = 0; i < 16; i += 4)
        {
            int b = (r + 4 * i) / 16;
            BF(x[i], x[i + 2], zetas[4 + b], zetas_q[4 + b]);
            b = (r + 4 * (i + 1)) / 16;
            BF(x[i + 1], x[i + 3], zetas[4 + b], zetas_q[4 + b]);
        }

        /* len=16: pairs (i,i+1), even i; block = v/8 */
        for (i = 0; i < 16; i += 2)
        {
            int b = (r + 4 * i) / 8;
            BF(x[i], x[i + 1], zetas[8 + b], zetas_q[8 + b]);
        }

        for (i = 0; i < 16; i++)
            f->v[r + 4 * i] = x[i];
    }
}

/* Pass B: interleave four independent blocks, preserving each block's
 * butterfly order and reductions while exposing more independent work. */
static void ntt_passB(poly8 *f)
{
    int bb, b, j;
    for (bb = 0; bb < 16; bb += 4)
    {
        __m512i y[16];
        for (j = 0; j < 16; j++)
            y[j] = f->v[4 * bb + j];
        for (b = bb; b < bb + 4; b++)
        {
            int o = 4 * (b - bb);
            BF(y[o], y[o + 2], zetas[16 + b], zetas_q[16 + b]);
            BF(y[o + 1], y[o + 3], zetas[16 + b], zetas_q[16 + b]);
        }
        for (b = bb; b < bb + 4; b++)
        {
            int o = 4 * (b - bb);
            BF(y[o], y[o + 1], zetas[32 + 2 * b], zetas_q[32 + 2 * b]);
            BF(y[o + 2], y[o + 3], zetas[33 + 2 * b], zetas_q[33 + 2 * b]);
        }
        for (b = bb; b < bb + 4; b++)
            for (j = 0; j < 4; j++)
                y[4 * (b - bb) + j] = bf_len2(y[4 * (b - bb) + j],
                    zetas[64 + 4 * b + j], zetas_q[64 + 4 * b + j]);
        for (j = 0; j < 16; j++)
            f->v[4 * bb + j] = y[j];
    }
}

void poly8_reduce(poly8 *f)
{
    int v;
    for (v = 0; v < 64; v++)
        f->v[v] = barrett16(f->v[v]);
}

void poly8_ntt(poly8 *f)
{
    ntt_passA(f);
    ntt_passB(f);
    poly8_reduce(f); /* ref poly_ntt ends with poly_reduce */
}

/* Scale by FINV before len=2. For |input|<=16383, scaled inputs
 * have magnitude <2025. First three output bounds: 4050,8100,16200.
 * len=16 sums/differences fit 32400; retain its Barrett to reset to q.
 * Later layers end at 8q=26632. See docs/invntt-bounds.md. */
static void invntt_passB(poly8 *f)
{
    int b, j;
    for (b = 0; b < 16; b++)
    {
        __m512i y[4];
        for (j = 0; j < 4; j++)
            y[j] = fqmul_bc(f->v[4 * b + j], FINV, finv_q);

        /* len=2: window v -> zetas[127 - v] */
        for (j = 0; j < 4; j++)
            y[j] = ibf_len2(y[j], zetas[127 - (4 * b + j)],
                            zetas_q[127 - (4 * b + j)]);

        /* len=4: blocks v/2: (0,1)->63-2b, (2,3)->62-2b */
        IBF_LAZY(y[0], y[1], zetas[63 - 2 * b], zetas_q[63 - 2 * b]);
        IBF_LAZY(y[2], y[3], zetas[62 - 2 * b], zetas_q[62 - 2 * b]);

        /* len=8: block b -> zetas[31-b] */
        IBF_LAZY(y[0], y[2], zetas[31 - b], zetas_q[31 - b]);
        IBF_LAZY(y[1], y[3], zetas[31 - b], zetas_q[31 - b]);

        for (j = 0; j < 4; j++)
            f->v[4 * b + j] = y[j];
    }
}

static void invntt_passA(poly8 *f)
{
    int r, i;
    for (r = 0; r < 4; r++)
    {
        __m512i x[16];
        for (i = 0; i < 16; i++)
            x[i] = f->v[r + 4 * i];

        /* len=16: pairs (i,i+1), even i; block v/8 -> zetas[15 - v/8] */
        for (i = 0; i < 16; i += 2)
        {
            int b = (r + 4 * i) / 8;
            IBF(x[i], x[i + 1], zetas[15 - b], zetas_q[15 - b]);
        }

        /* len=32: pairs (i,i+2); block v/16 -> zetas[7 - v/16] */
        for (i = 0; i < 16; i += 4)
        {
            int b = (r + 4 * i) / 16;
            IBF_LAZY(x[i], x[i + 2], zetas[7 - b], zetas_q[7 - b]);
            b = (r + 4 * (i + 1)) / 16;
            IBF_LAZY(x[i + 1], x[i + 3], zetas[7 - b], zetas_q[7 - b]);
        }

        /* len=64: pairs (i,i+4); block v/32 -> zetas[3 - v/32] */
        for (i = 0; i < 4; i++)
            IBF_LAZY(x[i], x[i + 4], zetas[3], zetas_q[3]);
        for (i = 8; i < 12; i++)
            IBF_LAZY(x[i], x[i + 4], zetas[2], zetas_q[2]);

        /* len=128; FINV was applied before len=2. */
        for (i = 0; i < 8; i++)
            IBF_LAZY(x[i], x[i + 8], zetas[1], zetas_q[1]);

        for (i = 0; i < 16; i++)
            f->v[r + 4 * i] = x[i];
    }
}

void poly8_invntt(poly8 *f)
{
    invntt_passB(f);
    invntt_passA(f);
}

void poly8_sub(poly8 *r, const poly8 *a, const poly8 *b)
{
    int v;
    for (v = 0; v < 64; v++)
        r->v[v] = _mm512_sub_epi16(a->v[v], b->v[v]);
}

/* ---------------- basemul ---------------- */

/* Signed 32-bit Montgomery RED. Only the low word of each dword is
 * significant on return; the upper word is discarded by wide_finish.
 * For |T| <= 6*3329*4095 < q*32768 the exact quotient fits int16. */
static inline __m512i wide_red(__m512i t)
{
    __m512i m = _mm512_mullo_epi16(t, _mm512_set1_epi16(QINV));
    __m512i mqhi = _mm512_mulhi_epi16(m, QV);
    __m512i thi = _mm512_srli_epi32(t, 16);
    return _mm512_sub_epi16(thi, mqhi);
}

static inline __m512i wide_finish(__m512i even, __m512i odd)
{
    __m512i e = wide_red(even), o = wide_red(odd);
    return _mm512_mask_blend_epi16(0xAAAAAAAAu, e,
                                   _mm512_slli_epi32(o, 16));
}

void poly8_basemul_acc(poly8 *r, const fop8 op[3], const poly8 b[3])
{
    for (int v = 0; v < 64; v++)
    {
        __m512i e = _mm512_madd_epi16(b[0].v[v], op[0].v[v]);
        __m512i o = _mm512_madd_epi16(b[0].v[v], op[0].w[v]);
        for (int j = 1; j < 3; j++)
        {
            e = _mm512_add_epi32(e, _mm512_madd_epi16(b[j].v[v], op[j].v[v]));
            o = _mm512_add_epi32(o, _mm512_madd_epi16(b[j].v[v], op[j].w[v]));
        }
        r->v[v] = wide_finish(e, o);
    }
}

void fop8c_from_poly(fop8c *op, const int16_t c[256])
{
    int v, k;
    for (v = 0; v < 64; v++)
    {
        int16_t z = zetas[64 + v];
        int16_t b[4], p1[4], p2[4];
        uint64_t w = 0, x = 0;

        for (k = 0; k < 4; k++)
            b[k] = c[4 * v + k];

        /* v-form: zeta folded in, matching bm_z = [z, z, -z, -z] on the odd
         * lanes only. fqmul keeps it in the Montgomery domain, so the product
         * a*(z*b) equals the old (a*b)*z as a residue -- though not necessarily
         * as a representative, since Montgomery reduction lands in (-q, q)
         * rather than [0, q). Everything downstream normalizes before
         * compress/tomsg, and the reference comparison is mod q. */
        p1[0] = b[0];
        p1[1] = fqmul_s(b[1], z);
        p1[2] = b[2];
        p1[3] = fqmul_s(b[3], (int16_t)-z);
        /* w-form: the runtime rotate, precomputed. */
        p2[0] = b[1];
        p2[1] = b[0];
        p2[2] = b[3];
        p2[3] = b[2];

        for (k = 0; k < 4; k++)
        {
            uint16_t u1 = (uint16_t)p1[k];
            uint16_t u2 = (uint16_t)p2[k];
            w |= (uint64_t)u1 << (16 * k);
            x |= (uint64_t)u2 << (16 * k);
        }
        op->v[v] = w;
        op->w[v] = x;
    }
}

void poly8_basemul_acc_bc(poly8 *r, const fop8c op[3], const poly8 b[3])
{
    for (int v = 0; v < 64; v++)
    {
        __m512i e = _mm512_setzero_si512(), o = _mm512_setzero_si512();
        for (int j = 0; j < 3; j++)
        {
            __m512i a = b[j].v[v];
            __m512i p = _mm512_set1_epi64((long long)op[j].v[v]);
            __m512i w = _mm512_set1_epi64((long long)op[j].w[v]);
            __m512i ep = _mm512_madd_epi16(a, p);
            __m512i opart = _mm512_madd_epi16(a, w);
            e = j == 0 ? ep : _mm512_add_epi32(e, ep);
            o = j == 0 ? opart : _mm512_add_epi32(o, opart);
        }
        r->v[v] = wide_finish(e, o);
    }
}

/* Preabsorbed FINV: passB column bounds are 17592,3774,2111,1760.
 * Before len16, reduce one side of its eight r=0 butterflies. This keeps
 * sum/difference <=20921, then the ordinary len16 Barrett resets to q. */
static void invntt_passB_finv(poly8 *f)
{
    int b, j;
    for (b = 0; b < 16; b++)
    {
        __m512i y[4];
        for (j = 0; j < 4; j++)
            y[j] = f->v[4 * b + j];

        /* len=2: window v -> zetas[127 - v] */
        for (j = 0; j < 4; j++)
            y[j] = ibf_len2(y[j], zetas[127 - (4 * b + j)],
                            zetas_q[127 - (4 * b + j)]);

        /* len=4: blocks v/2: (0,1)->63-2b, (2,3)->62-2b */
        IBF_LAZY(y[0], y[1], zetas[63 - 2 * b], zetas_q[63 - 2 * b]);
        IBF_LAZY(y[2], y[3], zetas[62 - 2 * b], zetas_q[62 - 2 * b]);

        /* len=8: block b -> zetas[31-b] */
        IBF_LAZY(y[0], y[2], zetas[31 - b], zetas_q[31 - b]);
        IBF_LAZY(y[1], y[3], zetas[31 - b], zetas_q[31 - b]);

        for (j = 0; j < 4; j++)
            f->v[4 * b + j] = y[j];
    }
}

static void invntt_passA_finv(poly8 *f)
{
    int r, i;
    for (r = 0; r < 4; r++)
    {
        __m512i x[16];
        for (i = 0; i < 16; i++)
            x[i] = f->v[r + 4 * i];

        /* len=16: pairs (i,i+1), even i; block v/8 -> zetas[15 - v/8] */
        for (i = 0; i < 16; i += 2)
        {
            int b = (r + 4 * i) / 8;
            if (r == 0) x[i] = barrett16(x[i]);
            IBF(x[i], x[i + 1], zetas[15 - b], zetas_q[15 - b]);
        }

        /* len=32: pairs (i,i+2); block v/16 -> zetas[7 - v/16] */
        for (i = 0; i < 16; i += 4)
        {
            int b = (r + 4 * i) / 16;
            IBF_LAZY(x[i], x[i + 2], zetas[7 - b], zetas_q[7 - b]);
            b = (r + 4 * (i + 1)) / 16;
            IBF_LAZY(x[i + 1], x[i + 3], zetas[7 - b], zetas_q[7 - b]);
        }

        /* len=64: pairs (i,i+4); block v/32 -> zetas[3 - v/32] */
        for (i = 0; i < 4; i++)
            IBF_LAZY(x[i], x[i + 4], zetas[3], zetas_q[3]);
        for (i = 8; i < 12; i++)
            IBF_LAZY(x[i], x[i + 4], zetas[2], zetas_q[2]);

        /* len=128; FINV was absorbed into the fixed-key operands. */
        /* Each pair is final here; write it before the next pair to
         * shorten result lifetimes without changing the arithmetic. */
        for (i = 0; i < 8; i++) {
            IBF_LAZY(x[i], x[i + 8], zetas[1], zetas_q[1]);
            f->v[r + 4 * i] = x[i];
            f->v[r + 4 * (i + 8)] = x[i + 8];
        }
    }
}

void poly8_invntt_finv(poly8 *f)
{
    invntt_passB_finv(f);
    invntt_passA_finv(f);
}

void poly8_basemul_acc_bc_finv(poly8 *r, const fop8c_finv op[3], const poly8 b[3])
{
    for (int v = 0; v < 64; v++)
    {
        __m512i e = _mm512_setzero_si512(), o = _mm512_setzero_si512();
        for (int j = 0; j < 3; j++)
        {
            __m512i a = b[j].v[v];
            __m512i p = _mm512_set1_epi64((long long)op[j].v[v]);
            __m512i w = _mm512_set1_epi64((long long)op[j].w[v]);
            __m512i ep = _mm512_madd_epi16(a, p);
            __m512i opart = _mm512_madd_epi16(a, w);
            e = j == 0 ? ep : _mm512_add_epi32(e, ep);
            o = j == 0 ? opart : _mm512_add_epi32(o, opart);
        }
        r->v[v] = wide_finish(e, o);
    }
}

void fop8c_finv_from_poly(fop8c_finv *op, const int16_t c[256])
{
    fop8c plain;
    fop8c_from_poly(&plain, c);
    for (int v = 0; v < 64; v += 8)
    {
        __m512i a = _mm512_loadu_si512((const void *)(plain.v+v));
        __m512i b = _mm512_loadu_si512((const void *)(plain.w+v));
        a = fqmul_bc(a, FINV, (int16_t)(FINV * QINV));
        b = fqmul_bc(b, FINV, (int16_t)(FINV * QINV));
        _mm512_storeu_si512((void *)(op->v+v),a);
        _mm512_storeu_si512((void *)(op->w+v),b);
    }
}

/* With reduced a in [0,q], the exact message bit is 833<=a<=2496. */
static inline __mmask32 message_bits(__m512i a)
{
    return _mm512_cmpge_epi16_mask(a, _mm512_set1_epi16(833)) &
           _mm512_cmple_epi16_mask(a, _mm512_set1_epi16(2496));
}

/* Same exact 10-bit reciprocal correction as the AVX2 serializer,
 * now applied to 32 reduced coefficients at once. */
static inline __m512i quantize10x32(__m512i f0)
{
    const __m512i v = _mm512_set1_epi16(BARR);
    __m512i f1 = _mm512_mullo_epi16(f0, _mm512_slli_epi16(v,3));
    __m512i f2 = _mm512_add_epi16(f0,_mm512_set1_epi16(15));
    f0 = _mm512_mulhi_epi16(_mm512_slli_epi16(f0,3),v);
    f2 = _mm512_sub_epi16(f1,f2);
    f1 = _mm512_srli_epi16(_mm512_andnot_si512(f1,f2),15);
    f0 = _mm512_mulhrs_epi16(_mm512_sub_epi16(f0,f1),_mm512_set1_epi16(4096));
    return _mm512_and_si512(f0,_mm512_set1_epi16(1023));
}

#ifdef __AVX512VBMI__
/* Pack eight 40-bit words, then remove their three-byte gaps. Bytes 40..63
 * are not written, including when a destination ends at a page boundary. */
static inline void pack10x32(uint8_t out[40], __m512i x)
{
    const __m512i gather = _mm512_set_epi8(
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,60,59,58,57,56,52,51,50,
        49,48,44,43,42,41,40,36,35,34,33,32,28,27,26,25,
        24,20,19,18,17,16,12,11,10,9,8,4,3,2,1,0);
    x = _mm512_madd_epi16(x,_mm512_set1_epi32((1024<<16)+1));
    x = _mm512_sllv_epi32(x,_mm512_set1_epi64(12));
    x = _mm512_srli_epi64(x,12);
    x = _mm512_permutexvar_epi8(gather,x);
    _mm512_mask_storeu_epi8(out,(__mmask64)UINT64_C(0xffffffffff),x);
}
#else
/* Serialize 16 already-quantized coefficients into exactly 20 bytes. */
static inline void pack10x16(uint8_t out[20], __m256i f0)
{
    f0 = _mm256_madd_epi16(f0,_mm256_set1_epi64x((1024LL<<48)+(1LL<<32)+(1024<<16)+1));
    f0 = _mm256_sllv_epi32(f0,_mm256_set1_epi64x(12));
    f0 = _mm256_srli_epi64(f0,12);
    f0 = _mm256_shuffle_epi8(f0,_mm256_set_epi8(
        8,4,3,2,1,0,-1,-1,-1,-1,-1,-1,12,11,10,9,
        -1,-1,-1,-1,-1,-1,12,11,10,9,8,4,3,2,1,0));
    __m128i t1 = _mm256_extracti128_si256(f0,1);
    __m128i t0 = _mm_blend_epi16(_mm256_castsi256_si128(f0),t1,0xE0);
    _mm_storeu_si128((__m128i *)out,t0);
    uint32_t last = (uint32_t)_mm_cvtsi128_si32(t1);
    memcpy(out+16,&last,4);
}

static inline void pack10x32(uint8_t out[40], __m512i x)
{
    pack10x16(out,_mm512_castsi512_si256(x));
    pack10x16(out+20,_mm512_extracti64x4_epi64(x,1));
}
#endif

void poly8_unfold_add_compress10(uint8_t *out[8], const poly8 *f,
                               const int16_t *noise[8])
{
    for (int g = 0; g < 8; g++)
    {
        __m512i r[8];
        for (int l = 0; l < 8; l++) r[l] = f->v[8*g+l];
        transpose8x8_u64(r);
        for (int l = 0; l < 8; l++)
        {
            __m512i x = barrett16(_mm512_add_epi16(r[l],
                _mm512_loadu_si512((const void *)(noise[l]+32*g))));
            x = quantize10x32(x);
            pack10x32(out[l]+40*g,x);
        }
    }
}

void poly8_unfold_addmsg_compress4(uint8_t *out[8], const poly8 *f,
                                 const int16_t *noise[8], const uint8_t *m[8])
{
    for (int g = 0; g < 8; g++)
    {
        __m512i r[8];
        for (int l = 0; l < 8; l++) r[l] = f->v[8*g+l];
        transpose8x8_u64(r);
        for (int l = 0; l < 8; l++)
        {
            uint32_t bits; memcpy(&bits,m[l]+4*g,4);
            __m512i msg = _mm512_maskz_set1_epi16((__mmask32)bits,1665);
            __m512i x = _mm512_add_epi16(r[l],_mm512_loadu_si512((const void *)(noise[l]+32*g)));
            x = barrett16(_mm512_add_epi16(x,msg));
            x = _mm512_mulhi_epi16(x,_mm512_set1_epi16(BARR));
            x = _mm512_mulhrs_epi16(x,_mm512_set1_epi16(512));
            x = _mm512_and_si512(x,_mm512_set1_epi16(15));
            x = _mm512_madd_epi16(x,_mm512_set1_epi32((16<<16)+1));
            _mm_storeu_si128((__m128i *)(out[l]+16*g),_mm512_cvtepi32_epi8(x));
        }
    }
}

/* Recover eight messages from decompressed standard-order v and folded
 * inverse output. Each transpose produces 32 coefficients per request.
 * Fuse subtraction, reduction and message bits without polynomial stores. */
void poly8_std_sub_tomsg(uint8_t *m[8], const int16_t *a[8], const poly8 *b)
{
    for (int g = 0; g < 8; g++)
    {
        __m512i r[8];
        for (int l = 0; l < 8; l++) r[l] = b->v[8*g+l];
        transpose8x8_u64(r);
        for (int l = 0; l < 8; l++)
        {
            __m512i x = _mm512_loadu_si512((const void *)(a[l]+32*g));
            uint32_t bits = message_bits(barrett16(_mm512_sub_epi16(x,r[l])));
            memcpy(m[l]+4*g,&bits,4);
        }
    }
}

/* Decode one 40-byte block to 32 standard-order coefficients. The dword
 * permutation builds the same four 128-bit byte windows as two AVX2
 * decode16 operations; the byte shuffle consumes only input bytes 0..39. */
static inline __m512i decompress10_32(const uint8_t *p)
{
    const __m512i windows=_mm512_setr_epi32(0,1,2,3,2,3,4,5,
                                          5,6,7,8,7,8,9,10);
    const __m256i shuffle=_mm256_set_epi8(
        11,10,10,9,9,8,8,7,6,5,5,4,4,3,3,2,
        9,8,8,7,7,6,6,5,4,3,3,2,2,1,1,0);
    __m512i f=_mm512_maskz_loadu_epi8((__mmask64)0xffffffffffULL,p);
    f=_mm512_permutexvar_epi32(windows,f);
    f=_mm512_shuffle_epi8(f,_mm512_broadcast_i64x4(shuffle));
    f=_mm512_sllv_epi32(f,_mm512_set1_epi64(4));
    f=_mm512_srli_epi16(f,1);
    f=_mm512_and_si512(f,_mm512_set1_epi32((32736 << 16)+8184));
    return _mm512_mulhrs_epi16(f,_mm512_set1_epi32((Q << 16)+4*Q));
}

void poly8_decompress10_fold(poly8 *out, const uint8_t *in[8])
{
    for (int g=0;g<8;g++) {
        __m512i r[8];
        for (int l=0;l<8;l++) r[l]=decompress10_32(in[l]+40*g);
        transpose8x8_u64(r);
        for (int v=0;v<8;v++) out->v[8*g+v]=r[v];
    }
}

/* eta=2: exact signed CBD representative for one 4-bit input. All table
 * lookups remain inside vector registers; no secret-dependent address. */

void poly8_cbd2_fold(poly8 *out, const uint8_t *in[8])
{
    const __m512i lut=_mm512_broadcast_i32x4(
        _mm_setr_epi8(0,1,1,2,-1,0,0,1,-1,0,0,1,-2,-1,-1,0));
    const __m512i mask=_mm512_set1_epi8(15);
    const __m512i sel=_mm512_setr_epi32(0,16,4,20,8,24,12,28,
                                      0,0,0,0,0,0,0,0);
    for (int g=0;g<2;g++) {
        __m512i r[8];
        for (int l=0;l<8;l++)
            r[l]=_mm512_loadu_si512((const void *)(in[l]+64*g));
        transpose8x8_u64(r);
        for (int v=0;v<8;v++) {
            __m512i x=r[v];
            __m512i lo=_mm512_shuffle_epi8(lut,_mm512_and_si512(x,mask));
            __m512i hi=_mm512_shuffle_epi8(lut,_mm512_and_si512(_mm512_srli_epi16(x,4),mask));
            __m512i even=_mm512_unpacklo_epi8(lo,hi);
            __m512i odd =_mm512_unpackhi_epi8(lo,hi);
            for (int j=0;j<4;j++) {
                __m512i idx=_mm512_add_epi32(sel,_mm512_set1_epi32(j));
                __m512i bytes=_mm512_permutex2var_epi32(even,idx,odd);
                out->v[32*g+4*v+j]=_mm512_cvtepi8_epi16(_mm512_castsi512_si256(bytes));
            }
        }
    }
}

/* Same signed CBD expansion, accepting state-word order directly. */
void poly8_cbd2_fold_words(poly8 *out, const __m512i words[16])
{
    const __m512i lut=_mm512_broadcast_i32x4(
        _mm_setr_epi8(0,1,1,2,-1,0,0,1,-1,0,0,1,-2,-1,-1,0));
    const __m512i mask=_mm512_set1_epi8(15);
    const __m512i sel=_mm512_setr_epi32(0,16,4,20,8,24,12,28,
                                      0,0,0,0,0,0,0,0);
    for (int w=0;w<16;w++) {
            __m512i x=words[w];
            __m512i lo=_mm512_shuffle_epi8(lut,_mm512_and_si512(x,mask));
            __m512i hi=_mm512_shuffle_epi8(lut,_mm512_and_si512(_mm512_srli_epi16(x,4),mask));
            __m512i even=_mm512_unpacklo_epi8(lo,hi);
            __m512i odd =_mm512_unpackhi_epi8(lo,hi);
            for (int j=0;j<4;j++) {
                __m512i idx=_mm512_add_epi32(sel,_mm512_set1_epi32(j));
                __m512i bytes=_mm512_permutex2var_epi32(even,idx,odd);
                out->v[4*w+j]=_mm512_cvtepi8_epi16(_mm512_castsi512_si256(bytes));
            }
    }
}
