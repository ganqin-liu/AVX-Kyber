#include "mlkem_b4.h"
#include "poly4.h"
#include "params.h"
#include "align.h"
#include "poly.h"
#include "polyvec.h"
#include "cbd.h"
#include "rejsample.h"
#include "fips202x8.h"
#include <openssl/crypto.h>
#include <string.h>

#if KYBER_K != 3
#error "B4 is ML-KEM-768 only"
#endif

static void frombytes12(int16_t r[256], const uint8_t *a)
{
    for (int i = 0; i < 128; i++) {
        r[2*i] = (int16_t)((a[3*i] | ((uint16_t)a[3*i+1] << 8)) & 4095);
        r[2*i+1] = (int16_t)(((a[3*i+1] >> 4) | ((uint16_t)a[3*i+2] << 4)) & 4095);
    }
}

static unsigned rej_uniform(int16_t *r, unsigned len, const uint8_t *buf,
                            unsigned buflen)
{
    unsigned ctr = 0;
    for (unsigned pos = 0; ctr < len && pos+3 <= buflen; pos += 3) {
        unsigned a = (buf[pos] | ((unsigned)buf[pos+1] << 8)) & 4095;
        unsigned b = ((buf[pos+1] >> 4) | ((unsigned)buf[pos+2] << 4)) & 4095;
        if (a < 3329) r[ctr++] = (int16_t)a;
        if (ctr < len && b < 3329) r[ctr++] = (int16_t)b;
    }
    return ctr;
}

/* Retain the established eight-stream Keccak kernel, filling its two
 * groups of four with different matrix elements of the same four keys.
 * Nine matrix elements require five groups; the last four streams repeat
 * element eight and are discarded. Refills advance every stream, but only
 * incomplete streams consume additional bytes. */
static void gen_at4(fop4 out[3][3], const uint8_t *seed[4])
{
    ALIGNED_UINT8(REJ_UNIFORM_AVX_BUFLEN) buf[8];
    ALIGNED_UINT8(34) xin[8];
    int16_t astd[8][256];
    const uint8_t *inp[8];
    uint8_t *outp[8];
    const int16_t *cp[4];
    keccakx8_state st;
    unsigned ctr[8];
    for (int e = 0; e < 9; e += 2) {
        for (int l = 0; l < 8; l++) {
            int element = e+l/4;
            if (element == 9) element = 8;
            memcpy(xin[l].coeffs,seed[l%4],32);
            xin[l].coeffs[32] = (uint8_t)(element/3);
            xin[l].coeffs[33] = (uint8_t)(element%3);
            inp[l] = xin[l].coeffs;
            outp[l] = buf[l].coeffs;
        }
        shake128x8_absorb_once(&st,inp,34);
        shake128x8_squeezeblocks(outp,REJ_UNIFORM_AVX_NBLOCKS,&st);
        for (int l = 0; l < 8; l++) ctr[l] = rej_uniform_avx(astd[l],buf[l].coeffs);
        for (;;) {
            int pending = 0;
            for (int l = 0; l < 8; l++) pending |= ctr[l] < 256;
            if (!pending) break;
            shake128x8_squeezeblocks(outp,1,&st);
            for (int l = 0; l < 8; l++)
                if (ctr[l] < 256)
                    ctr[l] += rej_uniform(astd[l]+ctr[l],256-ctr[l],
                                           buf[l].coeffs,SHAKE128_RATE);
        }
        for (int group = 0; group < 2 && e+group < 9; group++) {
            for (int l = 0; l < 4; l++) cp[l] = astd[4*group+l];
            fop4_from_polys(&out[(e+group)/3][(e+group)%3],cp);
        }
    }
    /* The matrix and its XOF state contain public information only. */
}

void mlkem_enc_b4_folded(uint8_t ct[4][1088], uint8_t ss[4][32],
                         const uint8_t *pk[4], const uint8_t *coins[4])
{
    fop4 aF[3][3], tF[3]; /* 48 KiB; four requests, no dummy KEM lanes. */
    int16_t tstd[4][256];
    polyvec ustd[4], epstd[4], spstd[4];
    poly vstd[4], eppstd[4], k;
    poly4 spF[3], uF, vF;
    ALIGNED_UINT8(64) gin[8], kr[8];
    ALIGNED_UINT8(SHAKE256_RATE+32) nb[7][4];
    uint8_t unused[4][SHAKE256_RATE];
    const uint8_t *inp[8], *seed[4];
    const int16_t *cp[4];
    int16_t *mp[4];
    uint8_t *outp[8];
    keccakx8_state st;
    __m512i sw[4];

    poly4_init_tables();
    for (int l = 0; l < 4; l++) seed[l] = pk[l]+1152;
    for (int i = 0; i < 3; i++) {
        for (int l = 0; l < 4; l++) {
            frombytes12(tstd[l],pk[l]+384*i);
            cp[l] = tstd[l];
        }
        fop4_from_polys(&tF[i],cp);
    }

    /* H and G have only four independent jobs. Duplicating their streams
     * permits reuse of the reviewed x8 hash API without running extra KEMs. */
    for (int l = 0; l < 8; l++) {
        inp[l] = pk[l%4];
        outp[l] = gin[l].coeffs+32;
    }
    sha3_256x8(outp,inp,1184);
    for (int l = 0; l < 8; l++) {
        memcpy(gin[l].coeffs,coins[l%4],32);
        inp[l] = gin[l].coeffs;
        outp[l] = kr[l].coeffs;
    }
    sha3_512x8_64B(outp,inp);
    gen_at4(aF,seed);

    /* Pack nonce n for four requests and nonce n+1 for those same requests
     * into one Keccak permutation. Nonce seven is computed but discarded.
     * The absorbed message is exactly seed32 || nonce with SHAKE padding. */
    for (int l = 0; l < 8; l++) inp[l] = kr[l%4].coeffs+32;
    keccakx8_seed32_transpose(sw,inp);
    for (int n = 0; n < 7; n += 2) {
        shake256x8_absorb_seed32_nonce(&st,sw,(uint8_t)n);
        st.s[4] = _mm512_setr_epi64(n|0x1f00,n|0x1f00,n|0x1f00,n|0x1f00,
                                     (n+1)|0x1f00,(n+1)|0x1f00,
                                     (n+1)|0x1f00,(n+1)|0x1f00);
        for (int l = 0; l < 8; l++)
            outp[l] = n+l/4 < 7 ? nb[n+l/4][l%4].coeffs : unused[l%4];
        shake256x8_squeezeblocks(outp,1,&st);
    }
    for (int l = 0; l < 4; l++) {
        for (int i = 0; i < 3; i++) {
            poly_cbd_eta1(&spstd[l].vec[i],nb[i][l].vec);
            poly_cbd_eta1(&epstd[l].vec[i],nb[i+3][l].vec);
        }
        poly_cbd_eta1(&eppstd[l],nb[6][l].vec);
    }
    for (int i = 0; i < 3; i++) {
        for (int l = 0; l < 4; l++) cp[l] = spstd[l].vec[i].coeffs;
        poly4_fold(&spF[i],cp);
        poly4_ntt(&spF[i]);
    }
    for (int i = 0; i < 3; i++) {
        poly4_basemul_acc(&uF,aF[i],spF);
        poly4_invntt(&uF);
        for (int l = 0; l < 4; l++) mp[l] = ustd[l].vec[i].coeffs;
        poly4_unfold(mp,&uF);
    }
    poly4_basemul_acc(&vF,tF,spF);
    poly4_invntt(&vF);
    for (int l = 0; l < 4; l++) mp[l] = vstd[l].coeffs;
    poly4_unfold(mp,&vF);
    for (int l = 0; l < 4; l++) {
        polyvec_add(&ustd[l],&ustd[l],&epstd[l]);
        polyvec_reduce(&ustd[l]);
        poly_frommsg(&k,gin[l].coeffs);
        poly_add(&vstd[l],&vstd[l],&eppstd[l]);
        poly_add(&vstd[l],&vstd[l],&k);
        poly_reduce(&vstd[l]);
        polyvec_compress(ct[l],&ustd[l]);
        poly_compress(ct[l]+960,&vstd[l]);
        memcpy(ss[l],kr[l].coeffs,32);
    }
    OPENSSL_cleanse(ustd,sizeof(ustd));
    OPENSSL_cleanse(epstd,sizeof(epstd));
    OPENSSL_cleanse(spstd,sizeof(spstd));
    OPENSSL_cleanse(vstd,sizeof(vstd));
    OPENSSL_cleanse(eppstd,sizeof(eppstd));
    OPENSSL_cleanse(&k,sizeof(k));
    OPENSSL_cleanse(spF,sizeof(spF));
    OPENSSL_cleanse(&uF,sizeof(uF));
    OPENSSL_cleanse(&vF,sizeof(vF));
    OPENSSL_cleanse(gin,sizeof(gin));
    OPENSSL_cleanse(kr,sizeof(kr));
    OPENSSL_cleanse(nb,sizeof(nb));
    OPENSSL_cleanse(unused,sizeof(unused));
    OPENSSL_cleanse(&st,sizeof(st));
    OPENSSL_cleanse(sw,sizeof(sw));
}
