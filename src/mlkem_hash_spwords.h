#ifndef SPIKE_MLKEM_HASH_SPWORDS_H
#define SPIKE_MLKEM_HASH_SPWORDS_H
#include "fips202x8.h"

/* Fixed ML-KEM-768 G/PRF/J schedule, with nonce 0..2 in state-word order.
 * sp_words is a valid 64-byte-aligned 3x16 vector array: element [n][w]
 * contains LE64(PRF_n[l]+8*w) in 64-bit lane l. Each kr[l] has 64 writable
 * bytes, each nb_tail[n][l] (nonce n+3) and job[l] has 136 writable bytes.
 * gin[l] has exactly 64 readable bytes, ct[l] 1088, z 32; no byte-buffer
 * alignment or padding is needed. Inputs may repeat. All output regions
 * must be pairwise disjoint and disjoint from every input region.
 * J is unconditional; only the unused last 8 sp PRF bytes are omitted. */
void mlkem768x8_hashes_spwords(uint8_t *kr[8], __m512i sp_words[3][16],
                               uint8_t *nb_tail[4][8], uint8_t *job[8],
                               const uint8_t *gin[8], const uint8_t z[32],
                               const uint8_t *ct[8]);
#endif
