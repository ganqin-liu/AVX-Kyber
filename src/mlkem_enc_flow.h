#ifndef AVX_KYBER_MLKEM_ENC_FLOW_H
#define AVX_KYBER_MLKEM_ENC_FLOW_H
#include <stdint.h>
#include "polyvec.h"
#include "fips202x8.h"
/* All eight inputs and outputs valid, including partial Encaps cohorts.
 * key:32B, noise:7x136B, matrix:3x3 packed upstream AVX2 polynomials/lane. */
void mlkem_enc_hash_flow(uint8_t *key[8], uint8_t *noise[7][8], polyvec *matrix[8],
                         const uint8_t *pk[8], const uint8_t *coins[8]);
#endif
