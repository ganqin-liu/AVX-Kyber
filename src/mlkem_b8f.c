#include "mlkem_hash_spwords.h"
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

#include "params.h"
#include "align.h"
#include "poly.h"
#include "polyvec.h"
#include "cbd.h"
#include "rejsample.h"
#include "symmetric.h"
#include "verify.h"
#include "fips202.h" /* avx2 single-stream shake128 for handle init */
#include "fips202x8.h"
#include "poly8.h"
#include "mlkem_b8f.h"

#if KYBER_K != 3
#error "spike is ML-KEM-768 only"
#endif


#define CT_BYTES KYBER_CIPHERTEXTBYTES
#define JIN_BYTES (KYBER_SYMBYTES + CT_BYTES)

#if LS_OPERAND_KIND == 0
#define SHARED_FROM_POLY fop8_from_poly
#define SHARED_BASEMUL poly8_basemul_acc
#define SHARED_INVNTT poly8_invntt
#elif LS_OPERAND_KIND == 1
#define SHARED_FROM_POLY fop8c_from_poly
#define SHARED_BASEMUL poly8_basemul_acc_bc
#define SHARED_INVNTT poly8_invntt
#else
#define SHARED_FROM_POLY fop8c_finv_from_poly
#define SHARED_BASEMUL poly8_basemul_acc_bc_finv
#define SHARED_INVNTT poly8_invntt_finv
#endif

/* ---------------- handle init (cold path, standard order) ------------- */

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

void mlkem_skhf_init(mlkem_skhf *h, const uint8_t sk[KYBER_SECRETKEYBYTES])
{
    const uint8_t *pk = sk + KYBER_INDCPA_SECRETKEYBYTES;
    const uint8_t *seed = pk + KYBER_POLYVECBYTES;
    int16_t tmp[KYBER_N];
    int i, j;

    poly8_init_tables();
    for (i = 0; i < 3; i++)
    {
        frombytes12(tmp, sk + 384 * i);
        SHARED_FROM_POLY(&h->s[i], tmp);
        frombytes12(tmp, pk + 384 * i);
        SHARED_FROM_POLY(&h->t[i], tmp);
    }
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
        {
            gen_at_std(tmp, seed, i, j);
            SHARED_FROM_POLY(&h->a[i][j], tmp);
        }
    memcpy(h->hpk, sk + KYBER_SECRETKEYBYTES - 2 * KYBER_SYMBYTES,
           KYBER_SYMBYTES);
    memcpy(h->z, sk + KYBER_SECRETKEYBYTES - KYBER_SYMBYTES, KYBER_SYMBYTES);
}

/* ---------------- batch encaps (heterogeneous keys, folded poly) ------ */

/* A^T x8 expansion, one matrix element at a time: SHAKE-128 x8, per-lane
 * AVX2 rejection into STANDARD order (no nttunpack), lane-exact stream
 * continuation on refill, immediate fold into the fop8 operand. */
static void gen_at8_fop(fop8 aF[3][3], const uint8_t *seed[8])
{
    ALIGNED_UINT8(REJ_UNIFORM_AVX_BUFLEN) buf[8];
    ALIGNED_UINT8(34) xin[8];
    static int16_t astd[8][KYBER_N];
    const uint8_t *inp[8];
    uint8_t *outp[8];
    const int16_t *cp[8];
    keccakx8_state st;
    unsigned int ctr[8];
    int i, j, l, short_lanes;

    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
        {
            for (l = 0; l < 8; l++)
            {
                memcpy(xin[l].coeffs, seed[l], KYBER_SYMBYTES);
                xin[l].coeffs[32] = (uint8_t)i; /* transposed order */
                xin[l].coeffs[33] = (uint8_t)j;
                inp[l] = xin[l].coeffs;
                outp[l] = buf[l].coeffs;
            }
            shake128x8_absorb_once(&st, inp, 34);
            shake128x8_squeezeblocks(outp, REJ_UNIFORM_AVX_NBLOCKS, &st);
            for (l = 0; l < 8; l++)
                ctr[l] = rej_uniform_avx(astd[l], buf[l].coeffs);
            do
            {
                short_lanes = 0;
                for (l = 0; l < 8; l++)
                    if (ctr[l] < KYBER_N)
                        short_lanes = 1;
                if (!short_lanes)
                    break;
                shake128x8_squeezeblocks(outp, 1, &st);
                for (l = 0; l < 8; l++)
                    if (ctr[l] < KYBER_N)
                        ctr[l] += rej_uniform(astd[l] + ctr[l],
                                              KYBER_N - ctr[l],
                                              buf[l].coeffs, SHAKE128_RATE);
            } while (1);
            for (l = 0; l < 8; l++)
                cp[l] = astd[l];
            fop8_from_polys(&aF[i][j], cp);
        }
}

void mlkem_enc_b8_folded(uint8_t ct[8][KYBER_CIPHERTEXTBYTES],
                         uint8_t ss[8][KYBER_SSBYTES],
                         const uint8_t *pk[8], const uint8_t *coins[8])
{
    static fop8 aF[3][3], tF[3]; /* 12 operands, 96 KiB total */
    static int16_t tstd[8][KYBER_N];
    static polyvec ustd[8], epstd[8], spstd[8];
    static poly vstd[8], eppstd[8];
    poly8 spF[3], uF, vF;
    ALIGNED_UINT8(2 * KYBER_SYMBYTES) gin[8];
    ALIGNED_UINT8(2 * KYBER_SYMBYTES) kr[8];
    ALIGNED_UINT8(SHAKE256_RATE + 32) nb[7][8];
    const uint8_t *inp[8], *seed[8];
    const int16_t *cp[8];
    int16_t *mp[8];
    uint8_t *outp[8];
    keccakx8_state st;
    poly k;
    int l, i, n;

    poly8_init_tables();

    /* t-hat: unpack std order per lane, fold per row */
    for (l = 0; l < 8; l++)
        seed[l] = pk[l] + KYBER_POLYVECBYTES;
    for (i = 0; i < 3; i++)
    {
        for (l = 0; l < 8; l++)
        {
            frombytes12(tstd[l], pk[l] + 384 * i);
            cp[l] = tstd[l];
        }
        fop8_from_polys(&tF[i], cp);
    }

    /* H(ek) x8 */
    for (l = 0; l < 8; l++)
    {
        inp[l] = pk[l];
        outp[l] = gin[l].coeffs + KYBER_SYMBYTES;
    }
    sha3_256x8(outp, inp, KYBER_PUBLICKEYBYTES);

    /* (K, r) = G(m || H(ek)) x8, 64-byte specialized absorb */
    for (l = 0; l < 8; l++)
    {
        memcpy(gin[l].coeffs, coins[l], KYBER_SYMBYTES);
        inp[l] = gin[l].coeffs;
        outp[l] = kr[l].coeffs;
    }
    sha3_512x8_64B(outp, inp);

    gen_at8_fop(aF, seed);

    /* noise PRF x8 + per-lane CBD */
    {
        __m512i sw[4];
        for (l = 0; l < 8; l++)
            inp[l] = kr[l].coeffs + KYBER_SYMBYTES;
        keccakx8_seed32_transpose(sw, inp);
        for (n = 0; n < 7; n++)
        {
            for (l = 0; l < 8; l++)
                outp[l] = nb[n][l].coeffs;
            shake256x8_absorb_seed32_nonce(&st, sw, (uint8_t)n);
            shake256x8_squeezeblocks(outp, 1, &st);
        }
    }
    for (l = 0; l < 8; l++)
    {
        poly_cbd_eta1(&spstd[l].vec[0], nb[0][l].vec);
        poly_cbd_eta1(&spstd[l].vec[1], nb[1][l].vec);
        poly_cbd_eta1(&spstd[l].vec[2], nb[2][l].vec);
        poly_cbd_eta1(&epstd[l].vec[0], nb[3][l].vec);
        poly_cbd_eta1(&epstd[l].vec[1], nb[4][l].vec);
        poly_cbd_eta1(&epstd[l].vec[2], nb[5][l].vec);
        poly_cbd_eta1(&eppstd[l], nb[6][l].vec);
    }

    /* folded: r-hat, u = INTT(A^T o r-hat), v = INTT(t-hat o r-hat) */
    for (i = 0; i < 3; i++)
    {
        for (l = 0; l < 8; l++)
            cp[l] = spstd[l].vec[i].coeffs;
        poly8_fold(&spF[i], cp);
        poly8_ntt(&spF[i]);
    }
    for (i = 0; i < 3; i++)
    {
        poly8_basemul_acc(&uF, aF[i], spF);
        poly8_invntt(&uF);
        for (l = 0; l < 8; l++)
            mp[l] = ustd[l].vec[i].coeffs;
        poly8_unfold(mp, &uF);
    }
    poly8_basemul_acc(&vF, tF, spF);
    poly8_invntt(&vF);
    for (l = 0; l < 8; l++)
        mp[l] = vstd[l].coeffs;
    poly8_unfold(mp, &vF);

    /* per-lane tail: add noise, message, compress, output */
    for (l = 0; l < 8; l++)
    {
        polyvec_add(&ustd[l], &ustd[l], &epstd[l]);
        polyvec_reduce(&ustd[l]);
        poly_frommsg(&k, gin[l].coeffs);
        poly_add(&vstd[l], &vstd[l], &eppstd[l]);
        poly_add(&vstd[l], &vstd[l], &k);
        poly_reduce(&vstd[l]);
        polyvec_compress(ct[l], &ustd[l]);
        poly_compress(ct[l] + KYBER_POLYVECCOMPRESSEDBYTES, &vstd[l]);
        memcpy(ss[l], kr[l].coeffs, KYBER_SSBYTES);
    }
}

/* ---------------- batch decaps ---------------- */

#ifdef B8F_PROFILE
#include <stdio.h>
static inline uint64_t prof_tsc(void)
{
    unsigned lo, hi;
    __asm__ volatile("lfence; rdtsc; lfence" : "=a"(lo), "=d"(hi) : : "memory");
    return ((uint64_t)hi << 32) | lo;
}
static uint64_t prof_acc[7];
static uint64_t prof_calls;
#define PROF_MARK(slot, t0)                    \
    do                                         \
    {                                          \
        uint64_t t1_ = prof_tsc();             \
        prof_acc[slot] += t1_ - (t0);          \
        (t0) = t1_;                            \
    } while (0)
void mlkem_b8f_profile_dump(void)
{
    static const char *names[7] = {"decompress(v)",  "u decode+decrypt+msg",
                                   "append hpk",       "G+PRF+J+cbd(ep/epp)",
                                   "cbd+ntt(sp)", "reencrypt + compress",
                                   "verify + select"};
    int i;
    if (!prof_calls)
        return;
    printf("[b8f profile] %lu calls, cyc/op (batch/8):\n",
           (unsigned long)prof_calls);
    for (i = 0; i < 7; i++)
        printf("  %-22s %8.1f\n", names[i],
               (double)prof_acc[i] / (double)prof_calls / 8.0);
}
#else
#define PROF_MARK(slot, t0) (void)0
#endif

void mlkem_dec_b8_folded(uint8_t ss[8][KYBER_SSBYTES], const uint8_t *ct[8],
                         const mlkem_skhf *h)
{
    static polyvec epstd[8];
    static poly vstd[8], eppstd[8];
    poly8 bF[3], vF, mpF, spF[3];
    ALIGNED_UINT8(2 * KYBER_SYMBYTES) gin[8];
    ALIGNED_UINT8(2 * KYBER_SYMBYTES) kr[8];
#if LS_PRF_WORDS
    ALIGNED_UINT8(SHAKE256_RATE + 32) nb[4][8];
    __m512i sp_words[3][16];
#define NB_COUNT 4
#define EP_OFFSET 0
#else
    ALIGNED_UINT8(SHAKE256_RATE + 32) nb[7][8];
#define NB_COUNT 7
#define EP_OFFSET 3
#endif
    ALIGNED_UINT8(SHAKE256_RATE) job[8];
    uint8_t cmp8[8][KYBER_CIPHERTEXTBYTES];
    const int16_t *cp[8];
    const uint8_t *inp[8];
    uint8_t *outp[8], *nbp[NB_COUNT][8], *jp[8];
    int l, i, n, fail;
#ifdef B8F_PROFILE
    uint64_t t0_ = prof_tsc();
    prof_calls++;
#endif

    /* stage 1: decrypt in folded domain */
    for (l = 0; l < 8; l++)
    {
        poly_decompress(&vstd[l], ct[l] + KYBER_POLYVECCOMPRESSEDBYTES);
    }
    PROF_MARK(0, t0_);
    for (i = 0; i < 3; i++)
    {
        for (l = 0; l < 8; l++)
            inp[l] = ct[l] + 320 * i;
        poly8_decompress10_fold(&bF[i], inp);
        poly8_ntt(&bF[i]);
    }
    for (l = 0; l < 8; l++)
        cp[l] = vstd[l].coeffs;

    SHARED_BASEMUL(&mpF, h->s, bF);
    SHARED_INVNTT(&mpF);
    for (l = 0; l < 8; l++) outp[l] = gin[l].coeffs;
    poly8_std_sub_tomsg(outp, cp, &mpF);
    PROF_MARK(1, t0_);
    for (l = 0; l < 8; l++)
    {
        memcpy(gin[l].coeffs + KYBER_SYMBYTES, h->hpk, KYBER_SYMBYTES);
    }
    PROF_MARK(2, t0_);

    /* G, PRFs and the unconditional rejection J share independent rounds. */
    for (l = 0; l < 8; l++)
    {
        inp[l] = gin[l].coeffs;
        outp[l] = kr[l].coeffs;
        jp[l] = job[l].coeffs;
        for (n = 0; n < NB_COUNT; n++)
            nbp[n][l] = nb[n][l].coeffs;
    }
#if LS_PRF_WORDS
    mlkem768x8_hashes_spwords(outp, sp_words, nbp, jp, inp, h->z, ct);
#else
    mlkem768x8_hashes(outp, nbp, jp, inp, h->z, ct);
#endif
    for (l = 0; l < 8; l++)
    {
        poly_cbd_eta1(&epstd[l].vec[0], nb[EP_OFFSET+0][l].vec);
        poly_cbd_eta1(&epstd[l].vec[1], nb[EP_OFFSET+1][l].vec);
        poly_cbd_eta1(&epstd[l].vec[2], nb[EP_OFFSET+2][l].vec);
        poly_cbd_eta1(&eppstd[l], nb[EP_OFFSET+3][l].vec);
    }
    PROF_MARK(3, t0_);

    /* stage 3b: re-encryption in folded domain */
    for (i = 0; i < 3; i++)
    {
#if LS_PRF_WORDS
        poly8_cbd2_fold_words(&spF[i], sp_words[i]);
#else
        for (l = 0; l < 8; l++) inp[l] = nb[i][l].coeffs;
        poly8_cbd2_fold(&spF[i], inp);
#endif
        poly8_ntt(&spF[i]);
    }
    PROF_MARK(4, t0_);
    for (i = 0; i < 3; i++)
    {
        SHARED_BASEMUL(&bF[i], h->a[i], spF);
        SHARED_INVNTT(&bF[i]);
        for (l = 0; l < 8; l++)
        {
            cp[l] = epstd[l].vec[i].coeffs;
            outp[l] = cmp8[l] + 320*i;
        }
        poly8_unfold_add_compress10(outp, &bF[i], cp);
    }
    SHARED_BASEMUL(&vF, h->t, spF);
    SHARED_INVNTT(&vF);
    for (l = 0; l < 8; l++)
    {
        cp[l] = eppstd[l].coeffs;
        inp[l] = gin[l].coeffs;
        outp[l] = cmp8[l] + KYBER_POLYVECCOMPRESSEDBYTES;
    }
    poly8_unfold_addmsg_compress4(outp, &vF, cp, inp);
    PROF_MARK(5, t0_);

    /* J was computed unconditionally with G/PRF above. */

    /* Complete ciphertext comparison and constant-time secret selection. */
    for (l = 0; l < 8; l++)
    {
        fail = verify(ct[l], cmp8[l], KYBER_CIPHERTEXTBYTES);

        memcpy(ss[l], job[l].coeffs, KYBER_SSBYTES);
        cmov(ss[l], kr[l].coeffs, KYBER_SSBYTES, !fail);
    }
    PROF_MARK(6, t0_);
}
