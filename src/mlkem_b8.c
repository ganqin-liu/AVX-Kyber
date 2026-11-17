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
#include "indcpa.h" /* gen_matrix */
#include "fips202x8.h"
#include "mlkem_b8.h"
#include "mlkem_enc_flow.h"

#if KYBER_K != 3
#error "spike is ML-KEM-768 only"
#endif

#define CT_BYTES KYBER_CIPHERTEXTBYTES
#define JIN_BYTES (KYBER_SYMBYTES + CT_BYTES) /* z || c */

/* ------------------------------------------------------------------ */
/* Handle preparation                                                  */
/* ------------------------------------------------------------------ */

void mlkem_skh_init(mlkem_skh *h, const uint8_t sk[KYBER_SECRETKEYBYTES])
{
    const uint8_t *pk = sk + KYBER_INDCPA_SECRETKEYBYTES;
    const uint8_t *seed = pk + KYBER_POLYVECBYTES;

    mlkem_single_prepare(&h->single, sk);
    mlkem_skh4_init(&h->b4, sk);
    polyvec_frombytes(&h->skpv, sk);
    polyvec_frombytes(&h->pkpv, pk);
    gen_matrix(h->at, seed, 1); /* transposed: A^T for re-encryption */
    memcpy(h->hpk, sk + KYBER_SECRETKEYBYTES - 2 * KYBER_SYMBYTES,
           KYBER_SYMBYTES);
    memcpy(h->z, sk + KYBER_SECRETKEYBYTES - KYBER_SYMBYTES, KYBER_SYMBYTES);
}

/* ------------------------------------------------------------------ */
/* Per-instance stages (AVX2 kernels, mirroring upstream indcpa.c)     */
/* ------------------------------------------------------------------ */

static void indcpa_dec_cached(uint8_t m[KYBER_INDCPA_MSGBYTES],
                              const uint8_t *c, const polyvec *skpv)
{
    polyvec b;
    poly v, mp;

    polyvec_decompress(&b, c);
    poly_decompress(&v, c + KYBER_POLYVECCOMPRESSEDBYTES);

    polyvec_ntt(&b);
    polyvec_basemul_acc_montgomery(&mp, skpv, &b);
    poly_invntt_tomont(&mp);

    poly_sub(&mp, &v, &mp);
    poly_reduce(&mp);

    poly_tomsg(m, &mp);
}

/* Re-encryption tail once sp/ep/epp are sampled: NTT, matrix-vector
 * product against the cached A^T and t-hat, pack the candidate ct. */
static void reenc_tail(uint8_t c[KYBER_INDCPA_BYTES],
                       const uint8_t m[KYBER_INDCPA_MSGBYTES],
                       const mlkem_skh *h, polyvec *sp, polyvec *ep,
                       poly *epp)
{
    unsigned int i;
    polyvec b;
    poly v, k;

    poly_frommsg(&k, m);
    polyvec_ntt(sp);

    for (i = 0; i < KYBER_K; i++)
        polyvec_basemul_acc_montgomery(&b.vec[i], &h->at[i], sp);
    polyvec_basemul_acc_montgomery(&v, &h->pkpv, sp);

    polyvec_invntt_tomont(&b);
    poly_invntt_tomont(&v);

    polyvec_add(&b, &b, ep);
    poly_add(&v, &v, epp);
    poly_add(&v, &v, &k);
    polyvec_reduce(&b);
    poly_reduce(&v);

    polyvec_compress(c, &b);
    poly_compress(c + KYBER_POLYVECCOMPRESSEDBYTES, &v);
}

/* ------------------------------------------------------------------ */
/* v2: single-instance decaps on an expanded-key handle                */
/* ------------------------------------------------------------------ */

int mlkem_dec_cached(uint8_t ss[KYBER_SSBYTES],
                     const uint8_t ct[KYBER_CIPHERTEXTBYTES],
                     const mlkem_skh *h)
{
    int fail;
    uint8_t buf[2 * KYBER_SYMBYTES];
    uint8_t kr[2 * KYBER_SYMBYTES];
    uint8_t cmp[KYBER_CIPHERTEXTBYTES];
    polyvec sp, ep;
    poly epp, scratch;

    indcpa_dec_cached(buf, ct, &h->skpv);
    memcpy(buf + KYBER_SYMBYTES, h->hpk, KYBER_SYMBYTES);
    hash_g(kr, buf, 2 * KYBER_SYMBYTES);

    /* upstream nonce assignment: sp=0,1,2 ep=3,4,5 epp=6 (K=3) */
    poly_getnoise_eta1_4x(sp.vec + 0, sp.vec + 1, sp.vec + 2, ep.vec + 0,
                          kr + KYBER_SYMBYTES, 0, 1, 2, 3);
    poly_getnoise_eta1_4x(ep.vec + 1, ep.vec + 2, &epp, &scratch,
                          kr + KYBER_SYMBYTES, 4, 5, 6, 7);
    reenc_tail(cmp, buf, h, &sp, &ep, &epp);

    fail = verify(ct, cmp, KYBER_CIPHERTEXTBYTES);

    rkprf(ss, h->z, ct);
    cmov(ss, kr, KYBER_SYMBYTES, !fail);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Batch-8 core: PRF/G/J 8-way across instances, poly work per lane    */
/* ------------------------------------------------------------------ */

#define NOISE_POLYS 7 /* sp0..2, ep0..2, epp */

static void dec_b8_core(uint8_t ss[8][KYBER_SSBYTES], const uint8_t *ct[8],
                        const mlkem_skh *h[8])
{
    ALIGNED_UINT8(2 * KYBER_SYMBYTES) gin[8];
    ALIGNED_UINT8(2 * KYBER_SYMBYTES) kr[8];
    ALIGNED_UINT8(SHAKE256_RATE + 32) nb[NOISE_POLYS][8];
    ALIGNED_UINT8(33) pin[8];
    static uint8_t jin[8][JIN_BYTES];
    ALIGNED_UINT8(SHAKE256_RATE) job[8];
    uint8_t cmp[KYBER_CIPHERTEXTBYTES];
    const uint8_t *inp[8];
    uint8_t *outp[8];
    keccakx8_state st;
    int l, n, fail;

    /* stage 1: PKE decrypt per lane, assemble m' || H(ek) */
    for (l = 0; l < 8; l++)
    {
        indcpa_dec_cached(gin[l].coeffs, ct[l], &h[l]->skpv);
        memcpy(gin[l].coeffs + KYBER_SYMBYTES, h[l]->hpk, KYBER_SYMBYTES);
    }

    /* stage 2: G = SHA3-512 x8 */
    for (l = 0; l < 8; l++)
    {
        inp[l] = gin[l].coeffs;
        outp[l] = kr[l].coeffs;
    }
    sha3_512x8(outp, inp, 2 * KYBER_SYMBYTES);

    /* stage 3a: noise PRF x8 — 7 rounds of SHAKE-256(coins || nonce),
     * batched across instances (same nonce, 8 different coins) */
    for (n = 0; n < NOISE_POLYS; n++)
    {
        for (l = 0; l < 8; l++)
        {
            memcpy(pin[l].coeffs, kr[l].coeffs + KYBER_SYMBYTES,
                   KYBER_SYMBYTES);
            pin[l].coeffs[KYBER_SYMBYTES] = (uint8_t)n;
            inp[l] = pin[l].coeffs;
            outp[l] = nb[n][l].coeffs;
        }
        shake256x8_absorb_once(&st, inp, KYBER_SYMBYTES + 1);
        shake256x8_squeezeblocks(outp, 1, &st);
    }

    /* stage 3b: per-lane CBD + re-encryption tail + compare.
     * (Comparison result folds into a per-lane fail flag; the final
     * select happens after J below, as in upstream kem.c.) */
    /* stage 4: J = SHAKE-256(z || c) x8, needed on every path for
     * implicit rejection; independent of the re-encryption result */
    for (l = 0; l < 8; l++)
    {
        memcpy(jin[l], h[l]->z, KYBER_SYMBYTES);
        memcpy(jin[l] + KYBER_SYMBYTES, ct[l], CT_BYTES);
        inp[l] = jin[l];
        outp[l] = job[l].coeffs;
    }
    shake256x8_absorb_once(&st, inp, JIN_BYTES);
    shake256x8_squeezeblocks(outp, 1, &st);

    for (l = 0; l < 8; l++)
    {
        polyvec sp, ep;
        poly epp;

        poly_cbd_eta1(&sp.vec[0], nb[0][l].vec);
        poly_cbd_eta1(&sp.vec[1], nb[1][l].vec);
        poly_cbd_eta1(&sp.vec[2], nb[2][l].vec);
        poly_cbd_eta1(&ep.vec[0], nb[3][l].vec);
        poly_cbd_eta1(&ep.vec[1], nb[4][l].vec);
        poly_cbd_eta1(&ep.vec[2], nb[5][l].vec);
        poly_cbd_eta1(&epp, nb[6][l].vec);

        reenc_tail(cmp, gin[l].coeffs, h[l], &sp, &ep, &epp);
        fail = verify(ct[l], cmp, KYBER_CIPHERTEXTBYTES);

        memcpy(ss[l], job[l].coeffs, KYBER_SSBYTES);
        cmov(ss[l], kr[l].coeffs, KYBER_SSBYTES, !fail);
    }
}

void mlkem_dec_b8_handles(uint8_t ss[8][KYBER_SSBYTES], const uint8_t *ct[8],
                          const mlkem_skh *h[8])
{
    dec_b8_core(ss, ct, h);
}

/* ------------------------------------------------------------------ */
/* v3: from serialized keys — adds unpack + matrix XOF x8              */
/* ------------------------------------------------------------------ */

/* rej_uniform is static in upstream indcpa.c; verbatim copy */
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

/* A^T expansion for 8 instances: 9 rounds (one matrix element across all
 * 8 seeds) of SHAKE-128 x8; per-lane AVX2 rejection sampling; lane-exact
 * stream continuation on refill (all lanes advance, finished lanes
 * discard), matching upstream's x4 refill semantics. */
static void gen_at_x8(mlkem_skh *hs[8], const uint8_t *seed[8])
{
    ALIGNED_UINT8(REJ_UNIFORM_AVX_BUFLEN) buf[8];
    ALIGNED_UINT8(34) xin[8];
    const uint8_t *inp[8];
    uint8_t *outp[8];
    keccakx8_state st;
    unsigned int ctr[8];
    int i, j, l, short_lanes;

    for (i = 0; i < KYBER_K; i++)
    {
        for (j = 0; j < KYBER_K; j++)
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
                ctr[l] = rej_uniform_avx(hs[l]->at[i].vec[j].coeffs,
                                         buf[l].coeffs);

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
                        ctr[l] += rej_uniform(
                            hs[l]->at[i].vec[j].coeffs + ctr[l],
                            KYBER_N - ctr[l], buf[l].coeffs, SHAKE128_RATE);
            } while (1);

            for (l = 0; l < 8; l++)
                poly_nttunpack(&hs[l]->at[i].vec[j]);
        }
    }
}

/* ------------------------------------------------------------------ */
/* enc_b8: batch-8 encapsulation to 8 (heterogeneous) public keys.     */
/* All Keccak stages x8 (H(ek), G, matrix XOF, noise PRF); polynomial  */
/* arithmetic per-lane AVX2 via reenc_tail (encaps == re-encryption).  */
/* Bit-exact vs upstream crypto_kem_enc_derand (FIPS 203: m = coins).  */
/* ------------------------------------------------------------------ */

void mlkem_enc_b8(uint8_t ct[8][KYBER_CIPHERTEXTBYTES],
                  uint8_t ss[8][KYBER_SSBYTES],
                  const uint8_t *pk[8], const uint8_t *coins[8])
{
    mlkem_enc_b8_active(ct, ss, pk, coins, 8);
}

void mlkem_enc_b8_active(uint8_t ct[8][KYBER_CIPHERTEXTBYTES],
                         uint8_t ss[8][KYBER_SSBYTES],
                         const uint8_t *pk[8], const uint8_t *coins[8], int active)
{
    static mlkem_skh scratch[8]; /* bench scratch; not reentrant */
    mlkem_skh *hs[8];
    ALIGNED_UINT8(2 * KYBER_SYMBYTES) gin[8];
    ALIGNED_UINT8(2 * KYBER_SYMBYTES) kr[8];
    ALIGNED_UINT8(SHAKE256_RATE + 32) nb[NOISE_POLYS][8];
    uint8_t *kp[8], *noise[NOISE_POLYS][8];
    polyvec *matrix[8];
    int l, n;

    if (active < 1 || active > 8) return;

    for (l = 0; l < 8; l++)
    {
        hs[l] = &scratch[l];
        polyvec_frombytes(&scratch[l].pkpv, pk[l]);
    }

    for (l=0;l<8;l++) {
        memcpy(gin[l].coeffs,coins[l],32);
        kp[l]=kr[l].coeffs;
        matrix[l]=hs[l]->at;
        for (n=0;n<NOISE_POLYS;n++) noise[n][l]=nb[n][l].coeffs;
    }
    mlkem_enc_hash_flow(kp,noise,matrix,pk,coins);

    for (l = 0; l < active; l++)
    {
        polyvec sp, ep;
        poly epp;

        poly_cbd_eta1(&sp.vec[0], nb[0][l].vec);
        poly_cbd_eta1(&sp.vec[1], nb[1][l].vec);
        poly_cbd_eta1(&sp.vec[2], nb[2][l].vec);
        poly_cbd_eta1(&ep.vec[0], nb[3][l].vec);
        poly_cbd_eta1(&ep.vec[1], nb[4][l].vec);
        poly_cbd_eta1(&ep.vec[2], nb[5][l].vec);
        poly_cbd_eta1(&epp, nb[6][l].vec);

        reenc_tail(ct[l], gin[l].coeffs, hs[l], &sp, &ep, &epp);
        memcpy(ss[l], kr[l].coeffs, KYBER_SSBYTES);
    }
}

void mlkem_dec_b8_serialized(uint8_t ss[8][KYBER_SSBYTES],
                             const uint8_t *ct[8], const uint8_t *sk[8])
{
    static mlkem_skh scratch[8]; /* bench scratch; not reentrant */
    mlkem_skh *hs[8];
    const mlkem_skh *hc[8];
    const uint8_t *seed[8];
    int l;

    for (l = 0; l < 8; l++)
    {
        const uint8_t *pk = sk[l] + KYBER_INDCPA_SECRETKEYBYTES;

        hs[l] = &scratch[l];
        polyvec_frombytes(&scratch[l].skpv, sk[l]);
        polyvec_frombytes(&scratch[l].pkpv, pk);
        seed[l] = pk + KYBER_POLYVECBYTES;
        memcpy(scratch[l].hpk,
               sk[l] + KYBER_SECRETKEYBYTES - 2 * KYBER_SYMBYTES,
               KYBER_SYMBYTES);
        memcpy(scratch[l].z, sk[l] + KYBER_SECRETKEYBYTES - KYBER_SYMBYTES,
               KYBER_SYMBYTES);
        hc[l] = &scratch[l];
    }

    gen_at_x8(hs, seed);
    dec_b8_core(ss, ct, hc);
}
