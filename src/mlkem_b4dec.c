/* Ordinary compact B4 Decaps; cold preparation adapted from mlkem_b8f.c. */
#include "mlkem_b4dec.h"
#include "params.h"
#include "align.h"
#include "poly.h"
#include "polyvec.h"
#include "cbd.h"
#include "fips202.h"
#include "fips202x8.h"
#include "verify.h"
#include <openssl/crypto.h>
#include <string.h>
static void frombytes12(int16_t r[KYBER_N], const uint8_t *a)
{
    int i;
    for (i = 0; i < KYBER_N / 2; i++)
    {
        r[2 * i] = (int16_t)(((a[3 * i + 0] >> 0) |
                              ((uint16_t)a[3 * i + 1] << 8)) & 0xFFF);
        r[2 * i + 1] = (int16_t)(((a[3 * i + 1] >> 4) |
                                  ((uint16_t)a[3 * i + 2] << 4)) & 0xFFF);
    }
}

static unsigned int rej_uniform(int16_t *r, unsigned int len,
                                const uint8_t *buf, unsigned int buflen)
{
    unsigned int ctr, pos;
    uint16_t val0, val1;

    ctr = pos = 0;
    while (ctr < len && pos <= buflen - 3)
    {
        val0 = ((buf[pos + 0] >> 0) | ((uint16_t)buf[pos + 1] << 8)) & 0xFFF;
        val1 = ((buf[pos + 1] >> 4) | ((uint16_t)buf[pos + 2] << 4)) & 0xFFF;
        pos += 3;
        if (val0 < KYBER_Q)
            r[ctr++] = (int16_t)val0;
        if (ctr < len && val1 < KYBER_Q)
            r[ctr++] = (int16_t)val1;
    }
    return ctr;
}

/* A^T[i][j] in standard sequential order: XOF(seed || i || j) */
static void gen_at_std(int16_t r[KYBER_N], const uint8_t seed[KYBER_SYMBYTES],
                       int i, int j)
{
    uint8_t ext[34], buf[3 * SHAKE128_RATE];
    keccak_state st;
    unsigned int ctr;

    memcpy(ext, seed, KYBER_SYMBYTES);
    ext[32] = (uint8_t)i;
    ext[33] = (uint8_t)j;
    shake128_absorb_once(&st, ext, 34);
    shake128_squeezeblocks(buf, 3, &st);
    ctr = rej_uniform(r, KYBER_N, buf, sizeof buf);
    while (ctr < KYBER_N)
    {
        shake128_squeezeblocks(buf, 1, &st);
        ctr += rej_uniform(r + ctr, KYBER_N - ctr, buf, SHAKE128_RATE);
    }
}

void mlkem_skh4_init(mlkem_skh4 *h, const uint8_t sk[KYBER_SECRETKEYBYTES])
{
    const uint8_t *pk = sk + KYBER_INDCPA_SECRETKEYBYTES;
    const uint8_t *seed = pk + KYBER_POLYVECBYTES;
    int16_t tmp[KYBER_N];
    int i, j;

    poly8_init_tables();poly4_init_tables();
    for (i = 0; i < 3; i++)
    {
        frombytes12(tmp, sk + 384 * i);
        fop8c_from_poly(&h->s[i], tmp);
        frombytes12(tmp, pk + 384 * i);
        fop8c_from_poly(&h->t[i], tmp);
    }
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
        {
            gen_at_std(tmp, seed, i, j);
            fop8c_from_poly(&h->a[i][j], tmp);
        }
    memcpy(h->hpk, sk + KYBER_SECRETKEYBYTES - 2 * KYBER_SYMBYTES,
           KYBER_SYMBYTES);
    memcpy(h->z, sk + KYBER_SECRETKEYBYTES - KYBER_SYMBYTES, KYBER_SYMBYTES);
    OPENSSL_cleanse(tmp,sizeof tmp);
}

/* Two consecutive four-coefficient compact words form one B4 slot.
 * Ordinary operands preserve the existing 6*3329*4095 int32 bound. */
static __m512i red(__m512i t)
{
 __m512i m=_mm512_mullo_epi16(t,_mm512_set1_epi16(-3327));
 return _mm512_sub_epi16(_mm512_srli_epi32(t,16),_mm512_mulhi_epi16(m,_mm512_set1_epi16(3329)));
}
void poly4_basemul_acc_bc(poly4 *r,const fop8c op[3],const poly4 b[3])
{
 for(int v=0;v<32;v++){
  __m512i e=_mm512_setzero_si512(),o=e;
  for(int j=0;j<3;j++){
   __m512i a=_mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)&op[j].v[2*v]));
   __m512i c=_mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)&op[j].w[2*v]));
   e=_mm512_add_epi32(e,_mm512_madd_epi16(b[j].v[v],a));o=_mm512_add_epi32(o,_mm512_madd_epi16(b[j].v[v],c));
  }
  r->v[v]=_mm512_mask_blend_epi16(0xaaaaaaaa,red(e),_mm512_slli_epi32(red(o),16));
 }
}
static uint64_t ld(const uint8_t *p){uint64_t x;memcpy(&x,p,8);return x;}
static __m256i load4(const uint8_t *const p[4],size_t off)
{return _mm256_setr_epi64x(ld(p[0]+off),ld(p[1]+off),ld(p[2]+off),ld(p[3]+off));}
static __m512i pair(__m256i low,__m256i high)
{return _mm512_inserti64x4(_mm512_castsi256_si512(low),high,1);}
/* Four J streams remain in the low half throughout. The high half runs G,
 * then one fresh PRF per permutation. J is unconditional, including valid CTs. */
void mlkem_b4_hashes(uint8_t kr[4][64],uint8_t nb[7][4][160],uint8_t job[4][32],const uint8_t gin[4][64],const uint8_t z[32],const uint8_t *ct[4])
{
 keccakx8_state st;__m256i sw[4];const uint8_t *g[4];uint64_t words[8];
 for(int l=0;l<4;l++)g[l]=gin[l];
 for(int w=0;w<25;w++){
  __m256i low=w<4?_mm256_set1_epi64x(ld(z+8*w)):w<17?load4(ct,8*w-32):_mm256_setzero_si256();
  __m256i high=w<8?load4(g,8*w):w==8?_mm256_set1_epi64x(0x8000000000000006ULL):_mm256_setzero_si256();st.s[w]=pair(low,high);
 }
 keccakx8_f1600(&st);
 for(int w=0;w<8;w++){
  _mm512_storeu_si512(words,st.s[w]);for(int l=0;l<4;l++)memcpy(kr[l]+8*w,&words[l+4],8);
  if(w>=4)sw[w-4]=_mm512_extracti64x4_epi64(st.s[w],1);
 }
 for(int n=0;n<7;n++){
  for(int w=0;w<25;w++){
   __m256i low=_mm512_castsi512_si256(st.s[w]);if(w<17)low=_mm256_xor_si256(low,load4(ct,104+136*n+8*w));
   __m256i high=w<4?sw[w]:w==4?_mm256_set1_epi64x(n|0x1f00):w==16?_mm256_set1_epi64x(0x8000000000000000ULL):_mm256_setzero_si256();st.s[w]=pair(low,high);
  }
  keccakx8_f1600(&st);
  for(int w=0;w<17;w++){_mm512_storeu_si512(words,st.s[w]);for(int l=0;l<4;l++)memcpy(nb[n][l]+8*w,&words[l+4],8);}
 }
 for(int w=0;w<4;w++)st.s[w]=_mm512_xor_si512(st.s[w],pair(load4(ct,1056+8*w),_mm256_setzero_si256()));
 st.s[4]=_mm512_xor_si512(st.s[4],pair(_mm256_set1_epi64x(0x1f),_mm256_setzero_si256()));
 st.s[16]=_mm512_xor_si512(st.s[16],pair(_mm256_set1_epi64x(0x8000000000000000ULL),_mm256_setzero_si256()));
 keccakx8_f1600(&st);
 for(int w=0;w<4;w++){_mm512_storeu_si512(words,st.s[w]);for(int l=0;l<4;l++)memcpy(job[l]+8*w,&words[l],8);}
 OPENSSL_cleanse(&st,sizeof st);OPENSSL_cleanse(sw,sizeof sw);OPENSSL_cleanse(words,sizeof words);
}
void mlkem_dec_b4_folded(uint8_t ss[4][32],const uint8_t *ct[4],const mlkem_skh4 *h)
{
 struct {
  polyvec u[4],sp[4],ep[4];poly v[4],epp[4],message;
  poly4 b[3],sp4[3],result;
  _Alignas(32) uint8_t gin[4][64],kr[4][64],nb[7][4][160],job[4][32],cmp[4][1088];
 } q;
 const int16_t *cp[4];int16_t *mp[4];
 for(int l=0;l<4;l++){polyvec_decompress(&q.u[l],ct[l]);poly_decompress(&q.v[l],ct[l]+960);}
 for(int i=0;i<3;i++){
  for(int l=0;l<4;l++)cp[l]=q.u[l].vec[i].coeffs;
  poly4_fold(&q.b[i],cp);poly4_ntt(&q.b[i]);
 }
 poly4_basemul_acc_bc(&q.result,h->s,q.b);poly4_invntt(&q.result);
 for(int l=0;l<4;l++)mp[l]=q.u[l].vec[0].coeffs;
 poly4_unfold(mp,&q.result);
 for(int l=0;l<4;l++){poly_sub(&q.message,&q.v[l],&q.u[l].vec[0]);poly_reduce(&q.message);poly_tomsg(q.gin[l],&q.message);memcpy(q.gin[l]+32,h->hpk,32);}
 mlkem_b4_hashes(q.kr,q.nb,q.job,q.gin,h->z,ct);
 for(int l=0;l<4;l++){
  for(int i=0;i<3;i++){poly_cbd_eta1(&q.sp[l].vec[i],(const __m256i*)q.nb[i][l]);poly_cbd_eta1(&q.ep[l].vec[i],(const __m256i*)q.nb[i+3][l]);}
  poly_cbd_eta1(&q.epp[l],(const __m256i*)q.nb[6][l]);
 }
 for(int i=0;i<3;i++){for(int l=0;l<4;l++)cp[l]=q.sp[l].vec[i].coeffs;poly4_fold(&q.sp4[i],cp);poly4_ntt(&q.sp4[i]);}
 for(int i=0;i<3;i++){
  poly4_basemul_acc_bc(&q.result,h->a[i],q.sp4);poly4_invntt(&q.result);
  for(int l=0;l<4;l++)mp[l]=q.u[l].vec[i].coeffs;
  poly4_unfold(mp,&q.result);
 }
 poly4_basemul_acc_bc(&q.result,h->t,q.sp4);poly4_invntt(&q.result);
 for(int l=0;l<4;l++)mp[l]=q.v[l].coeffs;
 poly4_unfold(mp,&q.result);
 for(int l=0;l<4;l++){
  polyvec_add(&q.u[l],&q.u[l],&q.ep[l]);polyvec_reduce(&q.u[l]);
  poly_frommsg(&q.message,q.gin[l]);poly_add(&q.v[l],&q.v[l],&q.epp[l]);poly_add(&q.v[l],&q.v[l],&q.message);poly_reduce(&q.v[l]);
  polyvec_compress(q.cmp[l],&q.u[l]);poly_compress(q.cmp[l]+960,&q.v[l]);
  int fail=verify(ct[l],q.cmp[l],1088);memcpy(ss[l],q.job[l],32);cmov(ss[l],q.kr[l],32,!fail);
 }
 OPENSSL_cleanse(&q,sizeof q);
}
