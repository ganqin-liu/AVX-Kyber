#ifndef WT_MLKEM_POLY4_H
#define WT_MLKEM_POLY4_H

#include <immintrin.h>
#include <stdint.h>

/* Four requests, eight consecutive int16 coefficients per 128-bit slot.
 * v[r] contains coefficients 8*r..8*r+7 from each of the four requests.
 * The 4x4 block transpose and final-level NTT schedule follow ML-DSA B4.
 * Coefficient order is the standard Kyber reference order. */
typedef struct { __m512i v[32]; } poly4;
typedef struct { __m512i v[32], w[32]; } fop4;

void poly4_init_tables(void); /* thread safe */
void poly4_fold(poly4 *f, const int16_t *c[4]);
void poly4_unfold(int16_t *c[4], const poly4 *f);
void fop4_from_polys(fop4 *f, const int16_t *c[4]);
void poly4_ntt(poly4 *f); /* |input| <= 3329; reduced output */
void poly4_invntt(poly4 *f); /* |input| <= 16383; |output| <= 26632 */
/* |op source coefficients| <= 4095, |b| <= 3329; |output| <= 2913. */
void poly4_basemul_acc(poly4 *r, const fop4 op[3], const poly4 b[3]);

#endif
