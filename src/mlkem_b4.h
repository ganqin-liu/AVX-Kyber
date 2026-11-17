#ifndef WT_MLKEM_B4_H
#define WT_MLKEM_B4_H
#include <stdint.h>

/* Four independent ML-KEM-768 encapsulations. Keys must have canonical
 * 12-bit coefficients. Inputs/outputs are per-request standard encodings. */
void mlkem_enc_b4_folded(uint8_t ct[4][1088], uint8_t ss[4][32],
                         const uint8_t *pk[4], const uint8_t *coins[4]);
#endif
