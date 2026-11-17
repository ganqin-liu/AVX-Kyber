/* Folded-8 polynomial kernels for ML-KEM-768 (AVX-Kyber spike).
 *
 * Layout: a poly-batch ("poly8") is 64 ZMM registers; vector v holds
 * coefficients [4v .. 4v+3] of instances 0..7, one instance per 64-bit
 * slot. Coefficient indices inside one ZMM are identical across
 * instances, so:
 *   - every NTT layer with len >= 4 is cross-vector with ONE scalar
 *     broadcast twiddle per vector pair (no vector twiddle loads at all);
 *   - only the len=2 layer works intra-register (one 32-bit shuffle +
 *     blend), with a scalar twiddle too;
 *   - basemul needs a per-vector [z,z,-z,-z] twiddle pattern (built once).
 * NTT-domain order is the REFERENCE (standard) Kyber order, not the
 * pq-crystals AVX2 internal order: fold/unfold convert to/from standard
 * coefficient order via an 8x8 u64 transpose, and NTT-domain operands
 * (s-hat, t-hat, A-hat^T) are prepared in standard order.
 * Arithmetic is mod-q equivalent to the reference; the inverse reduction
 * schedule has its own bound proof in docs/invntt-bounds.md.
 */
#ifndef POLY8_H
#define POLY8_H

#include <immintrin.h>
#include <stdint.h>

typedef struct
{
    __m512i v[64];
} poly8;

/* Folded NTT-domain operand, pre-twiddled for basemul.
 *
 * Two forms per window fold zeta and the coefficient swap into the operand:
 *
 *   v = [b0, zeta*b1, b2, -zeta*b3]   (zeta applied in the Montgomery domain)
 *   w = [b1, b0, b3, b2]
 *
 * The dot-product kernel uses v/w in two signed VPMADDWD operations per
 * module term, accumulates in int32, then reduces once per output.
 * Only v/w are stored: 8 KiB per folded operand, 1 KiB per broadcast operand.
 * The broadcast fixed-key handle occupies 15 KiB plus hpk/z. */
typedef struct
{
    __m512i v[64];
    __m512i w[64];
} fop8;

void poly8_init_tables(void); /* idempotent; builds basemul twiddles */

/* c[l] points at 256 int16 coefficients of instance l (standard order) */
void poly8_fold(poly8 *f, const int16_t *c[8]);
void poly8_unfold(int16_t *c[8], const poly8 *f);

/* Decode eight 10-bit polynomials directly into folded standard order.
 * Reads exactly 320 bytes per input, with no alignment or padding required.
 * Inputs may share storage; output must not overlap any input. Coefficients
 * are the exact integers (3329*t + 512) >> 10, in [0,3326]. */
void poly8_decompress10_fold(poly8 *out, const uint8_t *in[8]);

/* Sample eta=2 CBD directly into folded standard order. Each input points
 * to exactly 128 readable bytes; no alignment/padding is needed. Inputs may
 * repeat; output must be a valid aligned poly8 and not overlap any input.
 * Coefficients are exact signed representatives in [-2,2]. */
void poly8_cbd2_fold(poly8 *out, const uint8_t *in[8]);

/* CBD eta=2 from exactly 16 aligned vectors (1024 bytes), where each
 * qword lane holds one request's next eight PRF bytes, little endian.
 * Output must be a valid aligned poly8 disjoint from the words input.
 * Exact signed representatives [-2,2], identical to poly8_cbd2_fold. */
void poly8_cbd2_fold_words(poly8 *out, const __m512i words[16]);

/* Build a folded operand by replicating ONE instance's poly (fixed-key
 * case) into all 8 slots. Each input coefficient
 * must have magnitude <=4095; this covers all decoded 12-bit inputs. */
void fop8_from_poly(fop8 *op, const int16_t c[256]);

/* Folded operand from 8 different polys, each with |coefficient|<=4095. */
void fop8_from_polys(fop8 *op, const int16_t *c[8]);

void poly8_ntt(poly8 *f);    /* ref poly_ntt: ntt core + barrett reduce */
/* Mod-q equivalent to ref poly_invntt_tomont. Input coefficients must lie
 * in [-16383,16383]; production basemul outputs satisfy [-2913,2913].
 * Output coefficients have absolute value <=26632. */
#define POLY8_INVNTT_OUTPUT_BOUND 26632
#define POLY8_BASEMUL_OUTPUT_BOUND 2913
void poly8_invntt(poly8 *f);
void poly8_reduce(poly8 *f);
void poly8_sub(poly8 *r, const poly8 *a, const poly8 *b);

/* r = sum_{i<3} basemul(op[i], b[i])  (ref polyvec_basemul_acc_montgomery
 * semantics, K=3). b coefficients must have |b|<=3329 and
 * op must be prepared from coefficients of magnitude <=4095. */
void poly8_basemul_acc(poly8 *r, const fop8 op[3], const poly8 b[3]);

/* Compact shared-key operand: because all 8 lanes hold the SAME key, the
 * operand needs only 8 bytes per vector (one 4-coefficient window),
 * broadcast at use via vpbroadcastq. Both arrays occupy 1 KiB per poly
 * (vs 8 KiB folded); the fixed-key handle occupies 15 KiB plus hpk/z. This is the
 * instruction-level fixed-key specialization: heterogeneous-key batches
 * cannot use it. */
typedef struct
{
    uint64_t v[64];
    uint64_t w[64];
} fop8c;

void fop8c_from_poly(fop8c *op, const int16_t c[256]);
void poly8_basemul_acc_bc(poly8 *r, const fop8c op[3], const poly8 b[3]);

/* Fused output conversions. Every pointer names one of eight requests.
 * Compression: |f coefficient|<=26632, noise in [-2,2]; writes exactly
 * 320 or 128 bytes per request. Message input names 32 bytes per request. */
void poly8_unfold_add_compress10(uint8_t *out[8], const poly8 *f,
                               const int16_t *noise[8]);
void poly8_unfold_addmsg_compress4(uint8_t *out[8], const poly8 *f,
                                 const int16_t *noise[8], const uint8_t *m[8]);

/* FINV-preabsorbed fixed-key domain. Constructor: |raw coefficient|<=4095.
 * Basemul: operands from this constructor and |dynamic coefficient|<=3329;
 * output magnitude <=2199. Dedicated inverse: input magnitude <=2199,
 * output magnitude <=26632. Never mix with the ordinary basemul/inverse pair. */
typedef struct { uint64_t v[64], w[64]; } fop8c_finv;
#define POLY8_FINV_BASEMUL_BOUND 2199
void fop8c_finv_from_poly(fop8c_finv *op, const int16_t c[256]);
void poly8_basemul_acc_bc_finv(poly8 *r, const fop8c_finv op[3], const poly8 b[3]);
void poly8_invntt_finv(poly8 *f);

/* Standard-order a: 256 int16 coefficients; folded b: 256 coefficients.
 * Each exact a-b must fit int16. Writes exactly 32 bytes per request.
 * Production a is decompressed v in [0,3121], |b|<=26632. */
void poly8_std_sub_tomsg(uint8_t *m[8], const int16_t *a[8], const poly8 *b);

#endif
