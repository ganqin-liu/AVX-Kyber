/* v5: batch-8 ML-KEM-768 decapsulation on the folded-8 AVX-512 polynomial
 * backend (shared static key). Keccak stages (G/PRF/J) x8 as in v3/v4;
 * NTT/basemul/invNTT run in folded layout with the key operands
 * (s-hat, t-hat, A-hat^T in pre-zeta and swapped forms) cached in the handle.
 * Both forms absorb FINV and require the dedicated preabsorbed inverse.
 */
#ifndef MLKEM_B8F_H
#define MLKEM_B8F_H

#include <stdint.h>
#include "params.h"
#include "poly8.h"
#include "ablation_config.h"

#if LS_OPERAND_KIND == 0
typedef fop8 ls_shared_operand;
#elif LS_OPERAND_KIND == 1
typedef fop8c ls_shared_operand;
#else
typedef fop8c_finv ls_shared_operand;
#endif

typedef struct
{
    ls_shared_operand a[3][3]; /* A^T rows, representation selected at build */
    ls_shared_operand t[3];    /* t-hat */
    ls_shared_operand s[3];    /* s-hat */
    uint8_t hpk[KYBER_SYMBYTES];
    uint8_t z[KYBER_SYMBYTES];
} mlkem_skhf;

void mlkem_skhf_init(mlkem_skhf *h, const uint8_t sk[KYBER_SECRETKEYBYTES]);

void mlkem_dec_b8_folded(uint8_t ss[8][KYBER_SSBYTES],
                         const uint8_t *ct[8], const mlkem_skhf *h);

/* Batch-8 encapsulation to 8 heterogeneous public keys with the folded
 * polynomial backend (matrix XOF x8 straight into standard order, fold,
 * per-lane fop8 operands). Bit-exact vs crypto_kem_enc_derand. */
void mlkem_enc_b8_folded(uint8_t ct[8][KYBER_CIPHERTEXTBYTES],
                         uint8_t ss[8][KYBER_SSBYTES],
                         const uint8_t *pk[8], const uint8_t *coins[8]);

#endif
