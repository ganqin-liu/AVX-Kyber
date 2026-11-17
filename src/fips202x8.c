#include "fips202x8.h"

#include <assert.h>
#include <string.h>

/* Keccak-f[1600] round constants */
static const uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};

#define ROL(v, r) _mm512_rol_epi64((v), (r))
#define XOR(a, b) _mm512_xor_si512((a), (b))
#define AND(a, b) _mm512_and_si512((a), (b))
#define ANDNOT(a, b) _mm512_andnot_si512((a), (b)) /* (~a) & b */
/* 3-input fused logic (T1): 0x96 = a^b^c, 0xD2 = a^(~b&c) */
#define XOR3(a, b, c) _mm512_ternarylogic_epi64((a), (b), (c), 0x96)
#define CHI(a, b, c) _mm512_ternarylogic_epi64((a), (b), (c), 0xD2)

/* One Keccak-f round from named lanes S0..S24 into named lanes T0..T24
 * (ping-pong): the pi permutation is baked into which T each chi writes,
 * so it costs zero instructions, and every S lane is consumed exactly
 * once (theta-apply fused into the rho rotation input). Round op count:
 * 10 ternlog (parity) + 5 rol + 5 xor (D) + 25 xor + 24 rol + 25 ternlog
 * (chi) + 1 xor (iota) — no shadow-array copies. The previous in-place
 * formulation compiled to ~195 insns/round (~50 register moves + spills);
 * this one is single-use throughout, which is what the allocator needs. */
#define KR(S, T, rc)                                                      \
    do                                                                    \
    {                                                                     \
        __m512i c0 = XOR3(XOR3(S##0, S##5, S##10), S##15, S##20);         \
        __m512i c1 = XOR3(XOR3(S##1, S##6, S##11), S##16, S##21);         \
        __m512i c2 = XOR3(XOR3(S##2, S##7, S##12), S##17, S##22);         \
        __m512i c3 = XOR3(XOR3(S##3, S##8, S##13), S##18, S##23);         \
        __m512i c4 = XOR3(XOR3(S##4, S##9, S##14), S##19, S##24);         \
        __m512i d0 = XOR(c4, ROL(c1, 1));                                 \
        __m512i d1 = XOR(c0, ROL(c2, 1));                                 \
        __m512i d2 = XOR(c1, ROL(c3, 1));                                 \
        __m512i d3 = XOR(c2, ROL(c4, 1));                                 \
        __m512i d4 = XOR(c3, ROL(c0, 1));                                 \
        __m512i b0, b1, b2, b3, b4;                                       \
        b0 = XOR(S##0, d0);                                               \
        b1 = ROL(XOR(S##6, d1), 44);                                      \
        b2 = ROL(XOR(S##12, d2), 43);                                     \
        b3 = ROL(XOR(S##18, d3), 21);                                     \
        b4 = ROL(XOR(S##24, d4), 14);                                     \
        T##0 = XOR(CHI(b0, b1, b2),                                       \
                   _mm512_set1_epi64((long long)(rc)));                   \
        T##1 = CHI(b1, b2, b3);                                           \
        T##2 = CHI(b2, b3, b4);                                           \
        T##3 = CHI(b3, b4, b0);                                           \
        T##4 = CHI(b4, b0, b1);                                           \
        b0 = ROL(XOR(S##3, d3), 28);                                      \
        b1 = ROL(XOR(S##9, d4), 20);                                      \
        b2 = ROL(XOR(S##10, d0), 3);                                      \
        b3 = ROL(XOR(S##16, d1), 45);                                     \
        b4 = ROL(XOR(S##22, d2), 61);                                     \
        T##5 = CHI(b0, b1, b2);                                           \
        T##6 = CHI(b1, b2, b3);                                           \
        T##7 = CHI(b2, b3, b4);                                           \
        T##8 = CHI(b3, b4, b0);                                           \
        T##9 = CHI(b4, b0, b1);                                           \
        b0 = ROL(XOR(S##1, d1), 1);                                       \
        b1 = ROL(XOR(S##7, d2), 6);                                       \
        b2 = ROL(XOR(S##13, d3), 25);                                     \
        b3 = ROL(XOR(S##19, d4), 8);                                      \
        b4 = ROL(XOR(S##20, d0), 18);                                     \
        T##10 = CHI(b0, b1, b2);                                          \
        T##11 = CHI(b1, b2, b3);                                          \
        T##12 = CHI(b2, b3, b4);                                          \
        T##13 = CHI(b3, b4, b0);                                          \
        T##14 = CHI(b4, b0, b1);                                          \
        b0 = ROL(XOR(S##4, d4), 27);                                      \
        b1 = ROL(XOR(S##5, d0), 36);                                      \
        b2 = ROL(XOR(S##11, d1), 10);                                     \
        b3 = ROL(XOR(S##17, d2), 15);                                     \
        b4 = ROL(XOR(S##23, d3), 56);                                     \
        T##15 = CHI(b0, b1, b2);                                          \
        T##16 = CHI(b1, b2, b3);                                          \
        T##17 = CHI(b2, b3, b4);                                          \
        T##18 = CHI(b3, b4, b0);                                          \
        T##19 = CHI(b4, b0, b1);                                          \
        b0 = ROL(XOR(S##2, d2), 62);                                      \
        b1 = ROL(XOR(S##8, d3), 55);                                      \
        b2 = ROL(XOR(S##14, d4), 39);                                     \
        b3 = ROL(XOR(S##15, d0), 41);                                     \
        b4 = ROL(XOR(S##21, d1), 2);                                      \
        T##20 = CHI(b0, b1, b2);                                          \
        T##21 = CHI(b1, b2, b3);                                          \
        T##22 = CHI(b2, b3, b4);                                          \
        T##23 = CHI(b3, b4, b0);                                          \
        T##24 = CHI(b4, b0, b1);                                          \
    } while (0)

void keccakx8_f1600(keccakx8_state *st)
{
    __m512i A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13,
        A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24;
    __m512i E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13,
        E14, E15, E16, E17, E18, E19, E20, E21, E22, E23, E24;
    int r;

    A0 = st->s[0];   A1 = st->s[1];   A2 = st->s[2];   A3 = st->s[3];
    A4 = st->s[4];   A5 = st->s[5];   A6 = st->s[6];   A7 = st->s[7];
    A8 = st->s[8];   A9 = st->s[9];   A10 = st->s[10]; A11 = st->s[11];
    A12 = st->s[12]; A13 = st->s[13]; A14 = st->s[14]; A15 = st->s[15];
    A16 = st->s[16]; A17 = st->s[17]; A18 = st->s[18]; A19 = st->s[19];
    A20 = st->s[20]; A21 = st->s[21]; A22 = st->s[22]; A23 = st->s[23];
    A24 = st->s[24];

    for (r = 0; r < 24; r += 2)
    {
        KR(A, E, RC[r]);
        KR(E, A, RC[r + 1]);
    }

    st->s[0] = A0;   st->s[1] = A1;   st->s[2] = A2;   st->s[3] = A3;
    st->s[4] = A4;   st->s[5] = A5;   st->s[6] = A6;   st->s[7] = A7;
    st->s[8] = A8;   st->s[9] = A9;   st->s[10] = A10; st->s[11] = A11;
    st->s[12] = A12; st->s[13] = A13; st->s[14] = A14; st->s[15] = A15;
    st->s[16] = A16; st->s[17] = A17; st->s[18] = A18; st->s[19] = A19;
    st->s[20] = A20; st->s[21] = A21; st->s[22] = A22; st->s[23] = A23;
    st->s[24] = A24;
}


/* In-register 8x8 transpose of 64-bit lanes: rows r[0..7] (one zmm per
 * input stream, 8 consecutive u64 each) -> columns (one zmm per word
 * position across the 8 streams). Standard unpack + 128-bit shuffle +
 * 256-bit shuffle network. */
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

/* Gather one 64-bit word (byte offset off, may be unaligned) from each of
 * the 8 streams into one vector. Used for absorb tails and partial words. */
static __m512i load_word8(const uint8_t *in[8], size_t off)
{
    uint64_t w[8];
    int l;
    for (l = 0; l < 8; l++)
        memcpy(&w[l], in[l] + off, 8);
    return _mm512_loadu_si512((const void *)w);
}

/* XOR a full rate-block from each stream into the state.
 * rate/8 words: 21 (SHAKE128) or 17 (SHAKE256). Bulk in groups of 8 words
 * via the transpose network, remainder via load_word8. */
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

static void absorb_once(keccakx8_state *st, const uint8_t *in[8], size_t inlen,
                        size_t rate, uint8_t domain)
{
    size_t off = 0;
    size_t i;

    for (i = 0; i < 25; i++)
        st->s[i] = _mm512_setzero_si512();

    while (inlen >= rate)
    {
        absorb_block(st, in, off, rate);
        keccakx8_f1600(st);
        off += rate;
        inlen -= rate;
    }

    /* final partial block + padding, staged through a per-lane buffer */
    {
        uint64_t words[25][8]; /* [word][lane] */
        uint8_t tail[8][SHAKE128_RATE];
        int l;
        size_t w;

        memset(tail, 0, sizeof(tail));
        for (l = 0; l < 8; l++)
        {
            memcpy(tail[l], in[l] + off, inlen);
            tail[l][inlen] = domain;
            tail[l][rate - 1] |= 0x80;
        }
        memset(words, 0, sizeof(words));
        for (w = 0; w < rate / 8; w++)
            for (l = 0; l < 8; l++)
                memcpy(&words[w][l], &tail[l][8 * w], 8);
        for (w = 0; w < rate / 8; w++)
            st->s[w] = XOR(st->s[w],
                           _mm512_loadu_si512((const void *)words[w]));
    }
}

void shake128x8_absorb_once(keccakx8_state *st, const uint8_t *in[8],
                            size_t inlen)
{
    absorb_once(st, in, inlen, SHAKE128_RATE, 0x1F);
}

void shake256x8_absorb_once(keccakx8_state *st, const uint8_t *in[8],
                            size_t inlen)
{
    absorb_once(st, in, inlen, SHAKE256_RATE, 0x1F);
}

void shake256x8_absorb_prefix64_suffix2(keccakx8_state *st,
                                        const uint8_t *prefix64[8],
                                        const uint16_t suffix[8])
{
    __m512i r[8];
    uint64_t tail[8];
    __m512i zero = _mm512_setzero_si512();
    int l, w;

    /* The first 64 bytes are eight complete words per stream. */
    for (l = 0; l < 8; l++)
        r[l] = _mm512_loadu_si512((const void *)prefix64[l]);
    transpose8x8_u64(r);
    for (w = 0; w < 8; w++)
        st->s[w] = r[w];

    /* Bytes 64..66 are the suffix, byte 66 is SHAKE's 0x1f domain, and
     * byte 135 carries the final 0x80 bit.  All other state bytes are zero. */
    for (l = 0; l < 8; l++)
        tail[l] = (uint64_t)suffix[l] | ((uint64_t)0x1F << 16);
    st->s[8] = _mm512_loadu_si512((const void *)tail);
    for (w = 9; w < 16; w++)
        st->s[w] = zero;
    st->s[16] = _mm512_set1_epi64((long long)0x8000000000000000ULL);
    for (w = 17; w < 25; w++)
        st->s[w] = zero;
}

/* Store rate bytes of the current state to each of the 8 output streams.
 * Bulk via transpose (8-word groups), remainder via scalar stores. */
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

static void squeezeblocks(uint8_t *out[8], size_t nblocks, keccakx8_state *st,
                          size_t rate)
{
    size_t off = 0;
    while (nblocks-- > 0)
    {
        keccakx8_f1600(st);
        store_block(out, off, st, rate);
        off += rate;
    }
}

void shake128x8_squeezeblocks(uint8_t *out[8], size_t nblocks,
                              keccakx8_state *st)
{
    squeezeblocks(out, nblocks, st, SHAKE128_RATE);
}

void shake256x8_squeezeblocks(uint8_t *out[8], size_t nblocks,
                              keccakx8_state *st)
{
    squeezeblocks(out, nblocks, st, SHAKE256_RATE);
}

/* --- AVX-Kyber addition: 8-way SHA3-512 (ML-KEM hash G) --- */
#define SHA3_512_RATE 72

void sha3_512x8(uint8_t *out[8], const uint8_t *in[8], size_t inlen)
{
    keccakx8_state st;
    __m512i r[8];
    int l;

    absorb_once(&st, in, inlen, SHA3_512_RATE, 0x06);
    keccakx8_f1600(&st);

    /* 64 output bytes = state words 0..7: exactly one transpose group */
    for (l = 0; l < 8; l++)
        r[l] = st.s[l];
    transpose8x8_u64(r);
    for (l = 0; l < 8; l++)
        _mm512_storeu_si512((void *)out[l], r[l]);
}

/* --- AVX-Kyber addition: 8-way SHA3-256 (ML-KEM hash H over pk/ct) --- */
#define SHA3_256_RATE 136

void sha3_256x8(uint8_t *out[8], const uint8_t *in[8], size_t inlen)
{
    keccakx8_state st;
    __m512i r[8];
    int l;

    absorb_once(&st, in, inlen, SHA3_256_RATE, 0x06);
    keccakx8_f1600(&st);

    /* 32 output bytes = state words 0..3: lower half of one transpose
     * group */
    for (l = 0; l < 8; l++)
        r[l] = st.s[l];
    transpose8x8_u64(r);
    for (l = 0; l < 8; l++)
        _mm256_storeu_si256((__m256i *)(void *)out[l],
                            _mm512_castsi512_si256(r[l]));
}

/* --- AVX-Kyber additions: ML-KEM-specialized absorbs --- */

/* Transpose 8 lanes' 32-byte seeds into 4 state words (hoisted once per
 * batch; the per-nonce absorb below then costs no marshalling). */
void keccakx8_seed32_transpose(__m512i sw[4], const uint8_t *in[8])
{
    int w;
    for (w = 0; w < 4; w++)
        sw[w] = load_word8(in, 8 * (size_t)w);
}

/* SHAKE-256 x8 one-shot absorb of (seed32[l] || nonce): direct transposed
 * state construction (input 33 bytes < rate 136). */
void shake256x8_absorb_seed32_nonce(keccakx8_state *st, const __m512i sw[4],
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

/* SHAKE-256 x8 absorb of (z32 || ct[l]) without staging copies: z is
 * lane-invariant (broadcast), ct is read in place at offset -32. */
void shake256x8_absorb_z32_ct(keccakx8_state *st, const uint8_t z[32],
                              const uint8_t *ct[8], size_t ctlen)
{
    size_t total = 32 + ctlen;
    size_t nfull = total / SHAKE256_RATE;
    size_t off, w, k;
    uint64_t zw[4];

    memcpy(zw, z, 32);
    for (w = 0; w < 25; w++)
        st->s[w] = _mm512_setzero_si512();

    /* block 0: z || ct[0..103] */
    for (w = 0; w < 4; w++)
        st->s[w] = _mm512_set1_epi64((long long)zw[w]);
    for (w = 4; w < 17; w++)
        st->s[w] = XOR(st->s[w], load_word8(ct, 8 * w - 32));
    keccakx8_f1600(st);

    /* blocks 1..nfull-1: ct at offset 136k - 32 */
    for (k = 1; k < nfull; k++)
    {
        absorb_block(st, ct, k * SHAKE256_RATE - 32, SHAKE256_RATE);
        keccakx8_f1600(st);
    }

    /* tail: remaining ct bytes + padding (assumes tail is word-aligned
     * in length, true for z32||ct1088: tail = 32 bytes) */
    off = nfull * SHAKE256_RATE - 32;
    assert((total % SHAKE256_RATE) % 8 == 0);
    for (w = 0; w < (total % SHAKE256_RATE) / 8; w++)
        st->s[w] = XOR(st->s[w], load_word8(ct, off + 8 * w));
    st->s[(total % SHAKE256_RATE) / 8] =
        XOR(st->s[(total % SHAKE256_RATE) / 8],
            _mm512_set1_epi64((long long)0x1F));
    st->s[SHAKE256_RATE / 8 - 1] =
        XOR(st->s[SHAKE256_RATE / 8 - 1],
            _mm512_set1_epi64((long long)0x8000000000000000ULL));
}

/* SHAKE-256 x8 absorb of (pfx32[l] || data[l]): per-lane 32-byte prefix
 * (transcript hash chaining) followed by 8 equal-length data buffers.
 * Requires (32+datalen) mod rate to be word-aligned (all Noise pqNK
 * message sizes qualify). Generalizes absorb_z32_ct to per-lane prefix. */
void shake256x8_absorb_pfx32(keccakx8_state *st, const uint8_t *pfx[8],
                             const uint8_t *data[8], size_t datalen)
{
    size_t total = 32 + datalen;
    size_t nfull = total / SHAKE256_RATE;
    size_t tail = total % SHAKE256_RATE;
    size_t off, w, k;

    for (w = 0; w < 25; w++)
        st->s[w] = _mm512_setzero_si512();
    for (w = 0; w < 4; w++)
        st->s[w] = load_word8(pfx, 8 * w);

    if (nfull > 0)
    {
        for (w = 4; w < SHAKE256_RATE / 8; w++)
            st->s[w] = XOR(st->s[w], load_word8(data, 8 * w - 32));
        keccakx8_f1600(st);
        for (k = 1; k < nfull; k++)
        {
            absorb_block(st, data, k * SHAKE256_RATE - 32, SHAKE256_RATE);
            keccakx8_f1600(st);
        }
        off = nfull * SHAKE256_RATE - 32;
        assert(tail % 8 == 0);
        for (w = 0; w < tail / 8; w++)
            st->s[w] = XOR(st->s[w], load_word8(data, off + 8 * w));
    }
    else
    {
        assert(tail % 8 == 0);
        for (w = 4; w < tail / 8; w++)
            st->s[w] = XOR(st->s[w], load_word8(data, 8 * w - 32));
    }
    st->s[tail / 8] =
        XOR(st->s[tail / 8], _mm512_set1_epi64((long long)0x1F));
    st->s[SHAKE256_RATE / 8 - 1] =
        XOR(st->s[SHAKE256_RATE / 8 - 1],
            _mm512_set1_epi64((long long)0x8000000000000000ULL));
}

/* Permute + extract the first 64 output bytes of each lane (one transpose
 * group). One-shot SHAKE-256 x8 tail for the pfx32 absorb above. */
void shake256x8_pfx32_hash(uint8_t *out[8], const uint8_t *pfx[8],
                           const uint8_t *data[8], size_t datalen)
{
    keccakx8_state st;
    __m512i r[8];
    int l;

    shake256x8_absorb_pfx32(&st, pfx, data, datalen);
    keccakx8_f1600(&st);
    for (l = 0; l < 8; l++)
        r[l] = st.s[l];
    transpose8x8_u64(r);
    for (l = 0; l < 8; l++)
        _mm512_storeu_si512((void *)out[l], r[l]);
}

/* SHA3-512 x8 over exactly 64 bytes: one transpose group + padding word. */
void sha3_512x8_64B(uint8_t *out[8], const uint8_t *in[8])
{
    keccakx8_state st;
    __m512i r[8];
    int l, w;

    for (l = 0; l < 8; l++)
        r[l] = _mm512_loadu_si512((const void *)in[l]);
    transpose8x8_u64(r);
    for (w = 0; w < 8; w++)
        st.s[w] = r[w];
    st.s[8] = _mm512_set1_epi64((long long)(0x06ULL |
                                            (0x80ULL << 56)));
    for (w = 9; w < 25; w++)
        st.s[w] = _mm512_setzero_si512();
    keccakx8_f1600(&st);

    for (l = 0; l < 8; l++)
        r[l] = st.s[l];
    transpose8x8_u64(r);
    for (l = 0; l < 8; l++)
        _mm512_storeu_si512((void *)out[l], r[l]);
}

/* ================= generic x8 for the Noise symmetric layer =================
 *
 * See fips202x8.h. Rate is fixed at 136 (SHAKE256_RATE), shared by SHA3-256 and
 * SHAKE-256; the domain byte is the only difference between them.
 */

#define KX8_RATE SHAKE256_RATE      /* 136 */
#define KX8_RATEW (KX8_RATE / 8)    /* 17 words */

/* XOR a final partial block, byte-granular. `pre` bytes at the front of the
 * block are already in the state (the 32-byte prefix when nfull == 0), so they
 * are staged as zero here and the XOR is a no-op over them.
 *
 * No word-alignment precondition: len is arbitrary in [0, rate - pre). */
static void absorb_tail_bytes(keccakx8_state *st, const uint8_t *d[8],
                              size_t off, size_t len, size_t pre,
                              uint8_t domain)
{
    /* 8 x 136 staged, then transposed in. Padded to 192 so the two 64-byte
     * transpose groups can load past word 16 without reading out of bounds;
     * the slack is never absorbed. */
    uint8_t tb[8][192];
    uint64_t ww[8];
    __m512i r[8];
    size_t w;
    int l;

    for (l = 0; l < 8; l++)
    {
        memset(tb[l], 0, KX8_RATE);
        if (len != 0)
            memcpy(tb[l] + pre, d[l] + off, len);
        tb[l][pre + len] = domain;
        tb[l][KX8_RATE - 1] |= 0x80;
    }

    /* Words 0..15 via two 8x8 transposes rather than 128 scalar 8-byte copies.
     * This path runs ~26 times per handshake -- every MixHash and both halves
     * of every HMAC -- so the scalar staging it replaces was a large part of
     * the gap between the permutation-count model and measured cycles. */
    for (w = 0; w + 8 <= KX8_RATEW; w += 8)
    {
        for (l = 0; l < 8; l++)
            r[l] = _mm512_loadu_si512((const void *)&tb[l][8 * w]);
        transpose8x8_u64(r);
        for (l = 0; l < 8; l++)
            st->s[w + (size_t)l] = XOR(st->s[w + (size_t)l], r[l]);
    }
    for (; w < KX8_RATEW; w++)
    {
        for (l = 0; l < 8; l++)
            memcpy(&ww[l], &tb[l][8 * w], 8);
        st->s[w] = XOR(st->s[w], _mm512_loadu_si512((const void *)ww));
    }
}

void keccakx8_absorb_pfx32_gen(keccakx8_state *st, const uint8_t *pfx[8],
                               const uint8_t *data[8], size_t datalen,
                               uint8_t domain)
{
    size_t total = 32 + datalen;
    size_t nfull = total / KX8_RATE;
    size_t tail = total % KX8_RATE;
    size_t w, k;

    for (w = 0; w < 25; w++)
        st->s[w] = _mm512_setzero_si512();
    for (w = 0; w < 4; w++)
        st->s[w] = load_word8(pfx, 8 * w);

    if (nfull > 0)
    {
        /* first block: 32 prefix bytes already placed, then data[0 .. 103] */
        for (w = 4; w < KX8_RATEW; w++)
            st->s[w] = XOR(st->s[w], load_word8(data, 8 * w - 32));
        keccakx8_f1600(st);
        for (k = 1; k < nfull; k++)
        {
            absorb_block(st, data, k * KX8_RATE - 32, KX8_RATE);
            keccakx8_f1600(st);
        }
        absorb_tail_bytes(st, data, nfull * KX8_RATE - 32, tail, 0, domain);
    }
    else
    {
        absorb_tail_bytes(st, data, 0, datalen, 32, domain);
    }
}

void keccakx8_squeeze_le96(uint8_t *out[8], size_t outlen, keccakx8_state *st)
{
    __m512i r[8];
    uint8_t buf[8][64];
    int l;

    keccakx8_f1600(st);

    for (l = 0; l < 8; l++)
        r[l] = st->s[l];
    transpose8x8_u64(r);
    if (outlen <= 64)
    {
        /* Masked store rather than staging through a buffer and memcpy'ing:
         * outlen is always a multiple of 8 here (32, 64 or 96). */
        __mmask8 m = (__mmask8)((1u << ((outlen + 7) / 8)) - 1u);
        for (l = 0; l < 8; l++)
            _mm512_mask_storeu_epi64((void *)out[l], m, r[l]);
        return;
    }

    for (l = 0; l < 8; l++)
        _mm512_storeu_si512((void *)out[l], r[l]);

    {
        __mmask8 m = (__mmask8)((1u << ((outlen - 64 + 7) / 8)) - 1u);
        for (l = 0; l < 8; l++)
            r[l] = st->s[8 + l];
        transpose8x8_u64(r);
        for (l = 0; l < 8; l++)
            _mm512_mask_storeu_epi64((void *)(out[l] + 64), m, r[l]);
    }
    (void)buf;
}

void keccakx8_hash_pfx32(uint8_t *out[8], size_t outlen, const uint8_t *pfx[8],
                         const uint8_t *data[8], size_t datalen,
                         uint8_t domain)
{
    keccakx8_state st;

    keccakx8_absorb_pfx32_gen(&st, pfx, data, datalen, domain);
    keccakx8_squeeze_le96(out, outlen, &st);
}

void keccakx8_hmac_state(keccakx8_state *st, const uint8_t *key32[8],
                         uint8_t padbyte)
{
    __m512i padv = _mm512_set1_epi64(
        (long long)(0x0101010101010101ULL * (uint64_t)padbyte));
    size_t w;

    for (w = 0; w < 25; w++)
        st->s[w] = _mm512_setzero_si512();
    /* K0 is the 32-byte key zero-extended to the 136-byte block, so only the
     * first four words carry key material; the rest is pure pad. */
    for (w = 0; w < 4; w++)
        st->s[w] = XOR(load_word8(key32, 8 * w), padv);
    for (w = 4; w < KX8_RATEW; w++)
        st->s[w] = padv;
    keccakx8_f1600(st);
}

void keccakx8_absorb_short(keccakx8_state *st, const uint8_t *d[8], size_t len,
                           uint8_t domain)
{
    absorb_tail_bytes(st, d, 0, len, 0, domain);
}

void keccakx8_hmac_absorb(keccakx8_state *dst, const keccakx8_state *src,
                          const uint8_t *d[8], size_t len, uint8_t domain)
{
    uint8_t tb[8][64];
    __m512i r[8];
    size_t nw, w;
    int l;

    /* Every Noise HKDF input is 1, 32 or 33 bytes, so the message plus its
     * domain byte never reaches past word 7. Staging just those 64 bytes per
     * lane costs one transpose group, against the generic path's 136-byte
     * memset and two full-width transposes -- and the generic path was doing
     * that work for 17 words when at most 5 carry data. */
    if (len + 1 > sizeof tb[0])
    {
        *dst = *src;
        absorb_tail_bytes(dst, d, 0, len, 0, domain);
        return;
    }

    nw = (len + 1 + 7) / 8;
    for (l = 0; l < 8; l++)
    {
        memset(tb[l], 0, sizeof tb[l]);
        if (len != 0)
            memcpy(tb[l], d[l], len);
        tb[l][len] = domain;
        r[l] = _mm512_loadu_si512((const void *)tb[l]);
    }
    transpose8x8_u64(r);

    /* Fold the state copy into the XOR: the untouched words are a plain copy,
     * so nothing is written twice. */
    for (w = 0; w < nw; w++)
        dst->s[w] = XOR(src->s[w], r[w]);
    for (; w < 25; w++)
        dst->s[w] = src->s[w];

    /* Final-block bit. Rate word 16 is never a data word here. */
    dst->s[KX8_RATEW - 1] =
        XOR(dst->s[KX8_RATEW - 1],
            _mm512_set1_epi64((long long)0x8000000000000000ULL));
}

/* Two independent x8 states, interleaved by round. a and b must not alias. */
void keccakx8_f1600x2(keccakx8_state *a, keccakx8_state *b)
{
    __m512i A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24;
    __m512i E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20, E21, E22, E23, E24;
    __m512i P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23, P24;
    __m512i Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12, Q13, Q14, Q15, Q16, Q17, Q18, Q19, Q20, Q21, Q22, Q23, Q24;
    A0 = a->s[0]; A1 = a->s[1]; A2 = a->s[2]; A3 = a->s[3]; A4 = a->s[4];
    A5 = a->s[5]; A6 = a->s[6]; A7 = a->s[7]; A8 = a->s[8]; A9 = a->s[9];
    A10 = a->s[10]; A11 = a->s[11]; A12 = a->s[12]; A13 = a->s[13]; A14 = a->s[14];
    A15 = a->s[15]; A16 = a->s[16]; A17 = a->s[17]; A18 = a->s[18]; A19 = a->s[19];
    A20 = a->s[20]; A21 = a->s[21]; A22 = a->s[22]; A23 = a->s[23]; A24 = a->s[24];
    P0 = b->s[0]; P1 = b->s[1]; P2 = b->s[2]; P3 = b->s[3]; P4 = b->s[4];
    P5 = b->s[5]; P6 = b->s[6]; P7 = b->s[7]; P8 = b->s[8]; P9 = b->s[9];
    P10 = b->s[10]; P11 = b->s[11]; P12 = b->s[12]; P13 = b->s[13]; P14 = b->s[14];
    P15 = b->s[15]; P16 = b->s[16]; P17 = b->s[17]; P18 = b->s[18]; P19 = b->s[19];
    P20 = b->s[20]; P21 = b->s[21]; P22 = b->s[22]; P23 = b->s[23]; P24 = b->s[24];
    for (int r = 0; r < 24; r += 2)
    {
        KR(A, E, RC[r]);
        KR(P, Q, RC[r]);
        KR(E, A, RC[r + 1]);
        KR(Q, P, RC[r + 1]);
    }
    a->s[0] = A0; a->s[1] = A1; a->s[2] = A2; a->s[3] = A3; a->s[4] = A4;
    a->s[5] = A5; a->s[6] = A6; a->s[7] = A7; a->s[8] = A8; a->s[9] = A9;
    a->s[10] = A10; a->s[11] = A11; a->s[12] = A12; a->s[13] = A13; a->s[14] = A14;
    a->s[15] = A15; a->s[16] = A16; a->s[17] = A17; a->s[18] = A18; a->s[19] = A19;
    a->s[20] = A20; a->s[21] = A21; a->s[22] = A22; a->s[23] = A23; a->s[24] = A24;
    b->s[0] = P0; b->s[1] = P1; b->s[2] = P2; b->s[3] = P3; b->s[4] = P4;
    b->s[5] = P5; b->s[6] = P6; b->s[7] = P7; b->s[8] = P8; b->s[9] = P9;
    b->s[10] = P10; b->s[11] = P11; b->s[12] = P12; b->s[13] = P13; b->s[14] = P14;
    b->s[15] = P15; b->s[16] = P16; b->s[17] = P17; b->s[18] = P18; b->s[19] = P19;
    b->s[20] = P20; b->s[21] = P21; b->s[22] = P22; b->s[23] = P23; b->s[24] = P24;
}

/* Pair the eight full J blocks with G and seven independent PRFs.
 * The total remains 17 x8 state permutations: eight pairs plus one tail.
 * Keep the PRF seed in transposed state words instead of round-tripping it
 * through kr. No state or branch here depends on ciphertext acceptance. */
void mlkem768x8_hashes(uint8_t *kr[8], uint8_t *nb[7][8], uint8_t *job[8],
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
        shake256x8_absorb_seed32_nonce(&other, sw, (uint8_t)n);
        keccakx8_f1600x2(&jst, &other);
        store_block(nb[n], 0, &other, SHAKE256_RATE);
    }
    for (w = 0; w < 4; w++)
        jst.s[w] = XOR(jst.s[w], load_word8(ct, 1056 + 8 * (size_t)w));
    jst.s[4] = XOR(jst.s[4], _mm512_set1_epi64(0x1f));
    jst.s[16] = XOR(jst.s[16],
        _mm512_set1_epi64((long long)0x8000000000000000ULL));
    keccakx8_f1600(&jst);
    store_block(job, 0, &jst, SHAKE256_RATE);
}

