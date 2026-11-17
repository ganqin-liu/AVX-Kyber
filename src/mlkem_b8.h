/* Batch-8 ML-KEM-768 decapsulation spike (AVX-Kyber).
 *
 * Variants:
 *   v2  mlkem_dec_cached      — single instance, expanded-key handle
 *                               (cached A^T/t-hat/s-hat/H(ek), no per-call
 *                               matrix expansion or key unpacking)
 *   v3  mlkem_dec_b8_serialized — 8 instances from serialized secret keys:
 *                               everything from scratch, all Keccak stages
 *                               (matrix XOF, noise PRF, G, J) 8-way
 *   v3b mlkem_dec_b8_handles  — 8 instances from 8 (possibly distinct)
 *                               expanded-key handles: no matrix expansion,
 *                               PRF/G/J 8-way
 *   v4  fixed static key      — mlkem_dec_b8_handles with all 8 handle
 *                               pointers aliasing one entry
 *
 * Polynomial arithmetic is per-instance AVX2 (pq-crystals kernels); the
 * cross-instance axis feeds the AVX-512 8-way Keccak. This intentionally
 * understates a full folded-NTT8 backend: the spike isolates the
 * Keccak-batching + fixed-key-specialization gains.
 */
#ifndef MLKEM_B8_H
#define MLKEM_B8_H

#include <stdint.h>
#include "params.h"
#include "polyvec.h"
#include "mlkem_single.h"
#include "mlkem_b4dec.h"

typedef struct
{
    mlkem_single_key single; /* worker-owned prepared B1 backend */
    polyvec at[KYBER_K]; /* A^T, NTT domain, nttunpacked (avx2 layout) */
    polyvec pkpv;        /* t-hat, unpacked */
    polyvec skpv;        /* s-hat, unpacked */
    uint8_t hpk[KYBER_SYMBYTES]; /* H(ek) */
    uint8_t z[KYBER_SYMBYTES];   /* implicit-rejection secret */
    /* Ordinary compact B4 operands are owned, rebound and wiped with h2.
     * The separate B8 handle may use the incompatible FINV domain. */
    mlkem_skh4 b4;
} mlkem_skh;

void mlkem_skh_init(mlkem_skh *h, const uint8_t sk[KYBER_SECRETKEYBYTES]);

int mlkem_dec_cached(uint8_t ss[KYBER_SSBYTES],
                     const uint8_t ct[KYBER_CIPHERTEXTBYTES],
                     const mlkem_skh *h);

void mlkem_dec_b8_handles(uint8_t ss[8][KYBER_SSBYTES],
                          const uint8_t *ct[8],
                          const mlkem_skh *h[8]);

void mlkem_dec_b8_serialized(uint8_t ss[8][KYBER_SSBYTES],
                             const uint8_t *ct[8],
                             const uint8_t *sk[8]);

/* enc_b8: batch-8 encapsulation to 8 heterogeneous public keys, all
 * Keccak stages x8, per-lane AVX2 polynomial arithmetic. Bit-exact vs
 * crypto_kem_enc_derand(ct, ss, pk[l], coins[l]). */
void mlkem_enc_b8(uint8_t ct[8][KYBER_CIPHERTEXTBYTES],
                  uint8_t ss[8][KYBER_SSBYTES],
                  const uint8_t *pk[8], const uint8_t *coins[8]);

/* Experiment: retain eight valid hash/matrix inputs, but execute polynomial
 * tails and write outputs only for lanes [0, active). active must be 1..8.
 * This is not an active-wide Keccak kernel. Same non-reentrancy as enc_b8. */
void mlkem_enc_b8_active(uint8_t ct[8][KYBER_CIPHERTEXTBYTES],
                         uint8_t ss[8][KYBER_SSBYTES],
                         const uint8_t *pk[8], const uint8_t *coins[8], int active);

#endif
