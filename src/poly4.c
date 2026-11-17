#include "poly4.h"
#include <pthread.h>

#define Q 3329
#define QINV (-3327)
#define BARR 20159
#define FINV 1441
#define QV _mm512_set1_epi16(Q)
extern const int16_t pqcrystals_kyber768_ref_zetas[128];
#define zetas pqcrystals_kyber768_ref_zetas

static int16_t zetas_q[128], finv_q;
static __m128i bm_z[32], bm_q[32];
/* Each 128-bit pattern is replicated across the four requests. After
 * transposing four rows, adjacent coefficient pairs belong to rows 4*g+m. */
static __m128i f4_z[8], f4_q[8], f2a_z[8], f2a_q[8], f2b_z[8], f2b_q[8];
static __m128i i4_z[8], i4_q[8], i2a_z[8], i2a_q[8], i2b_z[8], i2b_q[8];
static pthread_once_t tables_once = PTHREAD_ONCE_INIT;

static __m128i pattern(int start, int step)
{
    int16_t z[8];
    for (int m = 0; m < 4; m++)
        z[2*m] = z[2*m+1] = zetas[start + step*m];
    return _mm_loadu_si128((const __m128i *)z);
}

static void init_tables(void)
{
    for (int i = 0; i < 128; i++)
        zetas_q[i] = (int16_t)((int32_t)zetas[i] * QINV);
    finv_q = (int16_t)((int32_t)FINV * QINV);
    for (int r = 0; r < 32; r++) {
        int16_t a = zetas[64+2*r], b = zetas[65+2*r];
        bm_z[r] = _mm_setr_epi16(a,a,-a,-a,b,b,-b,-b);
        bm_q[r] = _mm_mullo_epi16(bm_z[r], _mm_set1_epi16(QINV));
    }
    for (int g = 0; g < 8; g++) {
        f4_z[g] = pattern(32+4*g, 1);
        f2a_z[g] = pattern(64+8*g, 2);
        f2b_z[g] = pattern(65+8*g, 2);
        i4_z[g] = pattern(63-4*g, -1);
        i2a_z[g] = pattern(127-8*g, -2);
        i2b_z[g] = pattern(126-8*g, -2);
        f4_q[g] = _mm_mullo_epi16(f4_z[g], _mm_set1_epi16(QINV));
        f2a_q[g] = _mm_mullo_epi16(f2a_z[g], _mm_set1_epi16(QINV));
        f2b_q[g] = _mm_mullo_epi16(f2b_z[g], _mm_set1_epi16(QINV));
        i4_q[g] = _mm_mullo_epi16(i4_z[g], _mm_set1_epi16(QINV));
        i2a_q[g] = _mm_mullo_epi16(i2a_z[g], _mm_set1_epi16(QINV));
        i2b_q[g] = _mm_mullo_epi16(i2b_z[g], _mm_set1_epi16(QINV));
    }
}

void poly4_init_tables(void) { (void)pthread_once(&tables_once, init_tables); }

static inline __m512i fqmul(__m512i x, __m512i z, __m512i zq)
{
    __m512i m = _mm512_mullo_epi16(x, zq);
    return _mm512_sub_epi16(_mm512_mulhi_epi16(x, z),
                            _mm512_mulhi_epi16(m, QV));
}

static inline __m512i fqmul_bc(__m512i x, int k)
{
    return fqmul(x, _mm512_set1_epi16(zetas[k]),
                    _mm512_set1_epi16(zetas_q[k]));
}

static inline __m512i barrett(__m512i x)
{
    __m512i t = _mm512_mulhi_epi16(x, _mm512_set1_epi16(BARR));
    t = _mm512_srai_epi16(t, 10);
    return _mm512_sub_epi16(x, _mm512_mullo_epi16(t, QV));
}

/* Transpose 128-bit blocks, shared geometry with ML-DSA B4 ingress. */
static inline void blocks4(__m512i x[4])
{
    __m512i a = _mm512_shuffle_i32x4(x[0],x[1],0x88);
    __m512i b = _mm512_shuffle_i32x4(x[0],x[1],0xdd);
    __m512i c = _mm512_shuffle_i32x4(x[2],x[3],0x88);
    __m512i d = _mm512_shuffle_i32x4(x[2],x[3],0xdd);
    x[0] = _mm512_shuffle_i32x4(a,c,0x88);
    x[1] = _mm512_shuffle_i32x4(b,d,0x88);
    x[2] = _mm512_shuffle_i32x4(a,c,0xdd);
    x[3] = _mm512_shuffle_i32x4(b,d,0xdd);
}

/* Transpose four coefficient-pair rows inside every request's 128-bit
 * slot. This is ML-DSA B4's XP4 network with each dword holding two int16s.
 * Both transposes are involutions; neither mixes request slots. */
static inline void pairs4(__m512i x[4])
{
    __m512i a = _mm512_unpacklo_epi32(x[0],x[1]);
    __m512i b = _mm512_unpackhi_epi32(x[0],x[1]);
    __m512i c = _mm512_unpacklo_epi32(x[2],x[3]);
    __m512i d = _mm512_unpackhi_epi32(x[2],x[3]);
    x[0] = _mm512_unpacklo_epi64(a,c);
    x[1] = _mm512_unpackhi_epi64(a,c);
    x[2] = _mm512_unpacklo_epi64(b,d);
    x[3] = _mm512_unpackhi_epi64(b,d);
}

void poly4_fold(poly4 *f, const int16_t *c[4])
{
    for (int g = 0; g < 8; g++) {
        __m512i x[4];
        for (int l = 0; l < 4; l++)
            x[l] = _mm512_loadu_si512(c[l]+32*g);
        blocks4(x);
        for (int l = 0; l < 4; l++) f->v[4*g+l] = x[l];
    }
}

void poly4_unfold(int16_t *c[4], const poly4 *f)
{
    for (int g = 0; g < 8; g++) {
        __m512i x[4];
        for (int l = 0; l < 4; l++) x[l] = f->v[4*g+l];
        blocks4(x);
        for (int l = 0; l < 4; l++)
            _mm512_storeu_si512(c[l]+32*g, x[l]);
    }
}

void fop4_from_polys(fop4 *f, const int16_t *c[4])
{
    for (int g = 0; g < 8; g++) {
        __m512i x[4];
        for (int l = 0; l < 4; l++)
            x[l] = _mm512_loadu_si512(c[l]+32*g);
        blocks4(x);
        for (int l = 0; l < 4; l++) {
            int r = 4*g+l;
            __m512i t = fqmul(x[l], _mm512_broadcast_i32x4(bm_z[r]),
                                   _mm512_broadcast_i32x4(bm_q[r]));
            f->v[r] = _mm512_mask_blend_epi16(0xaaaaaaaau,x[l],t);
            f->w[r] = _mm512_rol_epi32(x[l],16);
        }
    }
}

#define BF(a,b,k) do { \
    __m512i t_ = fqmul_bc(b,k); \
    (b) = _mm512_sub_epi16(a,t_); (a) = _mm512_add_epi16(a,t_); \
} while (0)
#define BFP(a,b,z,zq) do { \
    __m512i t_ = fqmul(b,_mm512_broadcast_i32x4(z),_mm512_broadcast_i32x4(zq)); \
    (b) = _mm512_sub_epi16(a,t_); (a) = _mm512_add_epi16(a,t_); \
} while (0)
#define IBF(a,b,k,reduce) do { \
    __m512i s_ = _mm512_add_epi16(a,b), d_ = _mm512_sub_epi16(b,a); \
    (a) = (reduce) ? barrett(s_) : s_; (b) = fqmul_bc(d_,k); \
} while (0)
#define IBFP(a,b,z,zq) do { \
    __m512i s_ = _mm512_add_epi16(a,b), d_ = _mm512_sub_epi16(b,a); \
    (a) = s_; (b) = fqmul(d_,_mm512_broadcast_i32x4(z),_mm512_broadcast_i32x4(zq)); \
} while (0)

void poly4_ntt(poly4 *f)
{
    /* Register-resident len=128,64,32,16; two strided column groups. */
    for (int r = 0; r < 2; r++) {
        __m512i x[16];
        for (int i = 0; i < 16; i++) x[i] = f->v[r+2*i];
        for (int i = 0; i < 8; i++) BF(x[i],x[i+8],1);
        for (int i = 0; i < 4; i++) BF(x[i],x[i+4],2);
        for (int i = 8; i < 12; i++) BF(x[i],x[i+4],3);
        for (int i = 0; i < 16; i += 4) {
            BF(x[i],x[i+2],4+i/4); BF(x[i+1],x[i+3],4+i/4);
        }
        for (int i = 0; i < 16; i += 2) BF(x[i],x[i+1],8+i/2);
        for (int i = 0; i < 16; i++) f->v[r+2*i] = x[i];
    }
    for (int g = 0; g < 8; g++) {
        __m512i x[4];
        for (int i = 0; i < 4; i++) x[i] = f->v[4*g+i];
        BF(x[0],x[1],16+2*g); BF(x[2],x[3],17+2*g); /* len=8 */
        pairs4(x);
        BFP(x[0],x[2],f4_z[g],f4_q[g]);
        BFP(x[1],x[3],f4_z[g],f4_q[g]);
        BFP(x[0],x[1],f2a_z[g],f2a_q[g]);
        BFP(x[2],x[3],f2b_z[g],f2b_q[g]);
        pairs4(x);
        for (int i = 0; i < 4; i++) f->v[4*g+i] = barrett(x[i]);
    }
}

void poly4_invntt(poly4 *f)
{
    /* Same reduction schedule as folded B8: FINV first; lazy len=2,4,8;
     * Barrett on len=16 sums; lazy len=32,64,128. See ARITHMETIC.txt. */
    for (int g = 0; g < 8; g++) {
        __m512i x[4];
        for (int i = 0; i < 4; i++)
            x[i] = fqmul(f->v[4*g+i],_mm512_set1_epi16(FINV),_mm512_set1_epi16(finv_q));
        pairs4(x);
        IBFP(x[0],x[1],i2a_z[g],i2a_q[g]);
        IBFP(x[2],x[3],i2b_z[g],i2b_q[g]);
        IBFP(x[0],x[2],i4_z[g],i4_q[g]);
        IBFP(x[1],x[3],i4_z[g],i4_q[g]);
        pairs4(x);
        IBF(x[0],x[1],31-2*g,0); IBF(x[2],x[3],30-2*g,0);
        for (int i = 0; i < 4; i++) f->v[4*g+i] = x[i];
    }
    for (int r = 0; r < 2; r++) {
        __m512i x[16];
        for (int i = 0; i < 16; i++) x[i] = f->v[r+2*i];
        for (int i = 0; i < 16; i += 2) IBF(x[i],x[i+1],15-i/2,1);
        for (int i = 0; i < 16; i += 4) {
            IBF(x[i],x[i+2],7-i/4,0); IBF(x[i+1],x[i+3],7-i/4,0);
        }
        for (int i = 0; i < 4; i++) IBF(x[i],x[i+4],3,0);
        for (int i = 8; i < 12; i++) IBF(x[i],x[i+4],2,0);
        for (int i = 0; i < 8; i++) IBF(x[i],x[i+8],1,0);
        for (int i = 0; i < 16; i++) f->v[r+2*i] = x[i];
    }
}

/* Preserve B8's widened dot products and its single final Montgomery RED.
 * The bound is independent of how request slots are arranged. */
static inline __m512i wide_red(__m512i t)
{
    __m512i m = _mm512_mullo_epi16(t,_mm512_set1_epi16(QINV));
    return _mm512_sub_epi16(_mm512_srli_epi32(t,16),_mm512_mulhi_epi16(m,QV));
}

void poly4_basemul_acc(poly4 *r, const fop4 op[3], const poly4 b[3])
{
    for (int v = 0; v < 32; v++) {
        __m512i e = _mm512_madd_epi16(b[0].v[v],op[0].v[v]);
        __m512i o = _mm512_madd_epi16(b[0].v[v],op[0].w[v]);
        for (int j = 1; j < 3; j++) {
            e = _mm512_add_epi32(e,_mm512_madd_epi16(b[j].v[v],op[j].v[v]));
            o = _mm512_add_epi32(o,_mm512_madd_epi16(b[j].v[v],op[j].w[v]));
        }
        r->v[v] = _mm512_mask_blend_epi16(0xaaaaaaaau,wide_red(e),
                                         _mm512_slli_epi32(wide_red(o),16));
    }
}
