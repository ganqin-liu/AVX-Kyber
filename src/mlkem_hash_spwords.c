/* Specialized output adapter for shared-key ML-KEM-768 decapsulation.
 * The private I/O/seed helpers below are copied from fips202x8.c, preserving
 * that module's source and compiled object. Keep these copies in sync when
 * changing the generic implementation; hashschedtest compares both paths.
 * The Keccak permutations remain external calls to the unchanged module. */
#include "mlkem_hash_spwords.h"
#include <string.h>
#define XOR(a,b) _mm512_xor_si512((a),(b))

static void transpose8x8_u64(__m512i r[8])
{
    __m512i t[8];

    t[0] = _mm512_unpacklo_epi64(r[0], r[1]);
    t[1] = _mm512_unpackhi_epi64(r[0], r[1]);
    t[2] = _mm512_unpacklo_epi64(r[2], r[3]);
    t[3] = _mm512_unpackhi_epi64(r[2], r[3]);
    t[4] = _mm512_unpacklo_epi64(r[4], r[5]);
    t[5] = _mm512_unpackhi_epi64(r[4], r[5]);
    t[6] = _mm512_unpacklo_epi64(r[6], r[7]);
    t[7] = _mm512_unpackhi_epi64(r[6], r[7]);

    r[0] = _mm512_shuffle_i64x2(t[0], t[2], 0x88); /* 128-bit lanes 0,2 */
    r[1] = _mm512_shuffle_i64x2(t[1], t[3], 0x88);
    r[2] = _mm512_shuffle_i64x2(t[0], t[2], 0xdd); /* 128-bit lanes 1,3 */
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

    for (int i = 0; i < 8; i++)
        r[i] = t[i];
}

static __m512i load_word8(const uint8_t *in[8], size_t off)
{
    uint64_t w[8];
    int l;
    for (l = 0; l < 8; l++)
        memcpy(&w[l], in[l] + off, 8);
    return _mm512_loadu_si512((const void *)w);
}

static void absorb_block(keccakx8_state *st, const uint8_t *in[8], size_t off,
                         size_t rate)
{
    size_t nwords = rate / 8;
    size_t w = 0;

    while (w + 8 <= nwords)
    {
        __m512i r[8];
        int l;
        for (l = 0; l < 8; l++)
            r[l] = _mm512_loadu_si512((const void *)(in[l] + off + 8 * w));
        transpose8x8_u64(r);
        for (l = 0; l < 8; l++)
            st->s[w + (size_t)l] = XOR(st->s[w + (size_t)l], r[l]);
        w += 8;
    }
    for (; w < nwords; w++)
        st->s[w] = XOR(st->s[w], load_word8(in, off + 8 * w));
}

static void store_block(uint8_t *out[8], size_t off, const keccakx8_state *st,
                        size_t rate)
{
    size_t nwords = rate / 8;
    size_t w = 0;

    while (w + 8 <= nwords)
    {
        __m512i r[8];
        int l;
        for (l = 0; l < 8; l++)
            r[l] = st->s[w + (size_t)l];
        transpose8x8_u64(r);
        /* after transpose, r[l] = 8 consecutive words of stream l */
        for (l = 0; l < 8; l++)
            _mm512_storeu_si512((void *)(out[l] + off + 8 * w), r[l]);
        w += 8;
    }
    for (; w < nwords; w++)
    {
        uint64_t words[8];
        int l;
        _mm512_storeu_si512((void *)words, st->s[w]);
        for (l = 0; l < 8; l++)
            memcpy(out[l] + off + 8 * w, &words[l], 8);
    }
}

static void spwords_absorb_seed32_nonce(keccakx8_state *st, const __m512i sw[4],
                                    uint8_t nonce)
{
    __m512i zero = _mm512_setzero_si512();
    int w;

    st->s[0] = sw[0];
    st->s[1] = sw[1];
    st->s[2] = sw[2];
    st->s[3] = sw[3];
    st->s[4] = _mm512_set1_epi64((long long)((uint64_t)nonce |
                                             ((uint64_t)0x1F << 8)));
    for (w = 5; w < 16; w++)
        st->s[w] = zero;
    st->s[16] = _mm512_set1_epi64((long long)0x8000000000000000ULL);
    for (w = 17; w < 25; w++)
        st->s[w] = zero;
}

void mlkem768x8_hashes_spwords(uint8_t *kr[8], __m512i sp_words[3][16],
                               uint8_t *nb_tail[4][8], uint8_t *job[8],
                               const uint8_t *gin[8], const uint8_t z[32],
                               const uint8_t *ct[8])
{
    keccakx8_state jst, other;
    __m512i r[8], sw[4];
    uint64_t zw[4];
    int w, l, n;

    memset(&jst, 0, sizeof jst);
    memcpy(zw, z, sizeof zw);
    for (w = 0; w < 4; w++)
        jst.s[w] = _mm512_set1_epi64((long long)zw[w]);
    for (w = 4; w < 17; w++)
        jst.s[w] = load_word8(ct, 8 * (size_t)w - 32);

    for (l = 0; l < 8; l++)
        r[l] = _mm512_loadu_si512((const void *)gin[l]);
    transpose8x8_u64(r);
    for (w = 0; w < 8; w++)
        other.s[w] = r[w];
    other.s[8] = _mm512_set1_epi64((long long)0x8000000000000006ULL);
    for (w = 9; w < 25; w++)
        other.s[w] = _mm512_setzero_si512();
    keccakx8_f1600x2(&jst, &other);
    for (w = 0; w < 4; w++)
        sw[w] = other.s[4 + w];
    store_block(kr, 0, &other, 64);

    for (n = 0; n < 7; n++)
    {
        absorb_block(&jst, ct, 104 + 136 * (size_t)n, SHAKE256_RATE);
        spwords_absorb_seed32_nonce(&other, sw, (uint8_t)n);
        keccakx8_f1600x2(&jst, &other);
        if (n < 3) {
            for (w = 0; w < 16; w++) sp_words[n][w] = other.s[w];
        } else {
            store_block(nb_tail[n-3], 0, &other, SHAKE256_RATE);
        }
    }
    for (w = 0; w < 4; w++)
        jst.s[w] = XOR(jst.s[w], load_word8(ct, 1056 + 8 * (size_t)w));
    jst.s[4] = XOR(jst.s[4], _mm512_set1_epi64(0x1f));
    jst.s[16] = XOR(jst.s[16],
        _mm512_set1_epi64((long long)0x8000000000000000ULL));
    keccakx8_f1600(&jst);
    store_block(job, 0, &jst, SHAKE256_RATE);
}
