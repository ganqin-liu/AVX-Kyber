/* Selected enc_words port; see log/joint_followup_20261102. */
#include "mlkem_enc_flow.h"
#include "align.h"
#include "rejsample.h"
#include <openssl/crypto.h>
#include <string.h>
#include <assert.h>
#define XOR _mm512_xor_si512
#include "mlkem_enc_flow_io.inc"
typedef struct {
    keccakx8_state hash, matrix;
    __m512i rho[4], coins[4], r[4];
    ALIGNED_UINT8(REJ_UNIFORM_AVX_BUFLEN) buf[8];
    unsigned ctr[8], element, blocks;
} flow;
static void output32(uint8_t *out[8], const keccakx8_state *s)
{
    __m512i r[8];
    for (int i = 0; i < 8; i++)
        r[i] = s->s[i];
    transpose8x8_u64(r);
    for (int l = 0; l < 8; l++)
        _mm256_storeu_si256((void *)out[l], _mm512_castsi512_si256(r[l]));
}
static void matrix_start(flow *q)
{
    for (int j = 0; j < 25; j++)
        q->matrix.s[j] = _mm512_setzero_si512();
    for (int j = 0; j < 4; j++)
        q->matrix.s[j] = q->rho[j];
    q->matrix.s[4] =
        _mm512_set1_epi64((q->element / 3) | ((q->element % 3) << 8) | (0x1fULL << 16));
    q->matrix.s[20] = _mm512_set1_epi64((long long)(1ULL << 63));
    q->blocks = 0;
}
static void matrix_finish(flow *q, polyvec *mat[8])
{
    uint8_t *p[8];
    for (int l = 0; l < 8; l++)
        p[l] = q->buf[l].coeffs;
    unsigned i = q->element / 3, j = q->element % 3;
    if (q->blocks < 3) {
        store_block(p, q->blocks * SHAKE128_RATE, &q->matrix, SHAKE128_RATE);
        if (++q->blocks < 3)
            return;
        for (int l = 0; l < 8; l++)
            q->ctr[l] = rej_uniform_avx(mat[l][i].vec[j].coeffs, q->buf[l].coeffs);
    } else {
        store_block(p, 0, &q->matrix, SHAKE128_RATE);
        for (int l = 0; l < 8; l++)
            if (q->ctr[l] < 256)
                q->ctr[l] += rej_uniform(mat[l][i].vec[j].coeffs + q->ctr[l], 256 - q->ctr[l],
                                         q->buf[l].coeffs, SHAKE128_RATE);
    }
    for (int l = 0; l < 8; l++)
        if (q->ctr[l] < 256)
            return;
    for (int l = 0; l < 8; l++)
        poly_nttunpack(&mat[l][i].vec[j]);
    if (++q->element < 9)
        matrix_start(q);
}
static void hash_start(flow *q, const uint8_t *pk[8], int step)
{
    if (step < 9) {
        if (step == 0)
            memset(&q->hash, 0, sizeof q->hash);
        if (step < 8)
            absorb_block(&q->hash, pk, (size_t)step * 136, 136);
        else {
            absorb_block(&q->hash, pk, 1088, 96);
            q->hash.s[12] = XOR(q->hash.s[12], _mm512_set1_epi64(6));
            q->hash.s[16] = XOR(q->hash.s[16], _mm512_set1_epi64((long long)(1ULL << 63)));
        }
    } else if (step == 9) {
        __m512i h[4];
        for (int i = 0; i < 4; i++)
            h[i] = q->hash.s[i];
        for (int i = 0; i < 25; i++)
            q->hash.s[i] = _mm512_setzero_si512();
        for (int i = 0; i < 4; i++) {
            q->hash.s[i] = q->coins[i];
            q->hash.s[4 + i] = h[i];
        }
        q->hash.s[8] = _mm512_set1_epi64((long long)0x8000000000000006ULL);
    } else
        shake256x8_absorb_seed32_nonce(&q->hash, q->r, (uint8_t)(step - 10));
}
static void hash_finish(flow *q, uint8_t *key[8], uint8_t *noise[7][8], int step)
{
    if (step == 9) {
        output32(key, &q->hash);
        for (int i = 0; i < 4; i++)
            q->r[i] = q->hash.s[i + 4];
    } else if (step >= 10)
        store_block(noise[step - 10], 0, &q->hash, SHAKE256_RATE);
}
/* H/G/PRFs retain vector words; matrix follows with the original refill
 * semantics. Exactly 44+R state permutations; no dual-state scheduling. */
void mlkem_enc_hash_flow(uint8_t *key[8], uint8_t *noise[7][8], polyvec *mat[8],
                         const uint8_t *pk[8], const uint8_t *coins[8])
{
    flow q;
    const uint8_t *rho[8];
    for (int l = 0; l < 8; l++)
        rho[l] = pk[l] + KYBER_POLYVECBYTES;
    keccakx8_seed32_transpose(q.rho, rho);
    keccakx8_seed32_transpose(q.coins, coins);
    q.element = 0;
    matrix_start(&q);
    for (int step = 0; step < 17; step++) {
        hash_start(&q, pk, step);
        keccakx8_f1600(&q.hash);
        hash_finish(&q, key, noise, step);
    }
    while (q.element < 9) {
        keccakx8_f1600(&q.matrix);
        matrix_finish(&q, mat);
    }
    OPENSSL_cleanse(&q, sizeof q);
}
