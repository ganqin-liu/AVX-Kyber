/* 8-way vectorized Keccak/SHAKE for AVX-512.
 *
 * Lane model: state word w of all 8 instances lives in one __m512i
 * (64-bit lane l = instance l). Hash stages of the batch-16 verify run
 * as 2 passes of 8 (KECCAK_WAYS in params.h).
 *
 * Interleave contract (T5 in the plan): absorb reads 8 separate input
 * buffers of EQUAL length (mTLS fixed sizes make this sufficient;
 * asserted); squeeze de-interleaves into 8 separate output buffers.
 * Marshalling cost is part of this module and is included in its bench.
 */
#ifndef SPIKE_FIPS202X8_H
#define SPIKE_FIPS202X8_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#define SHAKE128_RATE 168
#define SHAKE256_RATE 136

typedef struct
{
    __m512i s[25];
} keccakx8_state;

/* One-shot absorb of 8 equal-length messages, domain byte 0x1F (SHAKE). */
void shake128x8_absorb_once(keccakx8_state *st, const uint8_t *in[8],
                            size_t inlen);
void shake256x8_absorb_once(keccakx8_state *st, const uint8_t *in[8],
                            size_t inlen);

/* ExpandMask-specialized one-block absorb:
 * SHAKE256(prefix64[l] || LE16(suffix[l])).  It constructs the transposed
 * state directly, avoiding the generic 8x136-byte padded tails and their
 * word-by-word restaging.  Squeezing remains the normal Keccak schedule. */
void shake256x8_absorb_prefix64_suffix2(keccakx8_state *st,
                                        const uint8_t *prefix64[8],
                                        const uint16_t suffix[8]);

/* Squeeze nblocks full rate-blocks into 8 separate buffers (each receives
 * nblocks*rate bytes), advancing the state. */
void shake128x8_squeezeblocks(uint8_t *out[8], size_t nblocks,
                              keccakx8_state *st);
void shake256x8_squeezeblocks(uint8_t *out[8], size_t nblocks,
                              keccakx8_state *st);

/* Raw permutation, exposed for the microbench (cycles/permutation). */
void keccakx8_f1600(keccakx8_state *st);
/* The two states must not alias. */
void keccakx8_f1600x2(keccakx8_state *a, keccakx8_state *b);

/* ML-KEM-768 decapsulation hashes: G(gin64), seven PRF blocks, J(z||ct1088).
 * All pointer arrays contain eight lanes; each nb[n][l] and job[l] has
 * room for 136 bytes, kr[l] for 64. Fixed lengths are part of this API.
 * Computes J unconditionally, including for accepted ciphertexts. */
void mlkem768x8_hashes(uint8_t *kr[8], uint8_t *nb[7][8], uint8_t *job[8],
                     const uint8_t *gin[8], const uint8_t z[32],
                     const uint8_t *ct[8]);

/* 8-way SHA3-512 over 8 equal-length messages (AVX-Kyber addition: the
 * ML-KEM G hash; inlen must fit rate handling of absorb_once). Each
 * out[l] receives 64 bytes. */
void sha3_512x8(uint8_t *out[8], const uint8_t *in[8], size_t inlen);

/* 8-way SHA3-256 over 8 equal-length messages (ML-KEM hash H); each
 * out[l] receives 32 bytes. */
void sha3_256x8(uint8_t *out[8], const uint8_t *in[8], size_t inlen);

/* AVX-Kyber ML-KEM-specialized entry points (see fips202x8.c) */
void keccakx8_seed32_transpose(__m512i sw[4], const uint8_t *in[8]);
void shake256x8_absorb_seed32_nonce(keccakx8_state *st, const __m512i sw[4],
                                    uint8_t nonce);
void shake256x8_absorb_z32_ct(keccakx8_state *st, const uint8_t z[32],
                              const uint8_t *ct[8], size_t ctlen);
void sha3_512x8_64B(uint8_t *out[8], const uint8_t *in[8]);

/* Noise-transcript x8: one-shot SHAKE-256 of (pfx32[l] || data[l]) with
 * per-lane 32-byte prefix; each out[l] receives 64 bytes. (32+datalen)
 * mod rate must be word-aligned. */
void shake256x8_absorb_pfx32(keccakx8_state *st, const uint8_t *pfx[8],
                             const uint8_t *data[8], size_t datalen);
void shake256x8_pfx32_hash(uint8_t *out[8], const uint8_t *pfx[8],
                           const uint8_t *data[8], size_t datalen);

/* ---- generic x8 entry points for the Noise symmetric layer ----
 *
 * Domain bytes: 0x06 = SHA3, 0x1F = SHAKE. Rate is 136 for both SHA3-256 and
 * SHAKE-256, which is why suites (b) and (c) have identical MixHash costs.
 *
 * Unlike shake256x8_absorb_pfx32 above, the tail here is byte-granular: no
 * word-alignment precondition, hence no assert. That matters beyond tidiness --
 * those asserts are live (no -DNDEBUG) and fire on attacker-influenced lengths,
 * i.e. they are a remote crash primitive once real packets arrive.
 */
#define KECCAKX8_DOMAIN_SHA3 0x06
#define KECCAKX8_DOMAIN_SHAKE 0x1F

/* Absorb (pfx32[l] || data[l]) at rate 136 and finalize, any datalen. */
void keccakx8_absorb_pfx32_gen(keccakx8_state *st, const uint8_t *pfx[8],
                               const uint8_t *data[8], size_t datalen,
                               uint8_t domain);

/* One-shot H(pfx32[l] || data[l]); outlen <= 96. */
void keccakx8_hash_pfx32(uint8_t *out[8], size_t outlen, const uint8_t *pfx[8],
                         const uint8_t *data[8], size_t datalen,
                         uint8_t domain);

/* HMAC ipad/opad state: absorb (key32[l] zero-padded to 136) XOR padbyte and
 * permute. The result is reusable across every message MACed under that key,
 * which is what makes HKDF's three output HMACs cost 2 permutations each
 * instead of 4. */
void keccakx8_hmac_state(keccakx8_state *st, const uint8_t *key32[8],
                         uint8_t padbyte);

/* Absorb a short (< 136 B) message into an already-permuted state, finalize. */
void keccakx8_absorb_short(keccakx8_state *st, const uint8_t *d[8], size_t len,
                           uint8_t domain);

/* dst = src with (d[l] || domain || pad) absorbed. Folds the state copy into
 * the XOR and stages only the words the message reaches -- the HMAC inner and
 * outer absorbs are 1, 32 or 33 bytes, so that is one transpose group instead
 * of the generic path's 136-byte memset and two full-width transposes. */
void keccakx8_hmac_absorb(keccakx8_state *dst, const keccakx8_state *src,
                          const uint8_t *d[8], size_t len, uint8_t domain);

/* Permute and extract outlen <= 96 bytes per lane. */
void keccakx8_squeeze_le96(uint8_t *out[8], size_t outlen, keccakx8_state *st);

#endif
