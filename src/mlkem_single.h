#ifndef AVX_KYBER_MLKEM_SINGLE_H
#define AVX_KYBER_MLKEM_SINGLE_H
#include <stdint.h>
#include <wolfssl/options.h>
#pragma push_macro("KYBER_N")
#undef KYBER_N
#include <wolfssl/wolfcrypt/wc_mlkem.h>
#pragma pop_macro("KYBER_N")

/* Each handle belongs to one process/worker. wolfSSL hash scratch is mutable.
 * Pinned static-key configuration has no persistent dynamic allocation. */
typedef struct { MlKemKey key; int status; } mlkem_single_key;
int mlkem_single_prepare(mlkem_single_key *, const uint8_t sk[2400]);
void mlkem_single_clear(mlkem_single_key *);
void mlkem_single_thread_clear(void);
int mlkem_single_enc(uint8_t ct[1088], uint8_t ss[32], const uint8_t pk[1184], const uint8_t coins[32]);
int mlkem_single_dec(uint8_t ss[32], const uint8_t ct[1088], mlkem_single_key *);
int mlkem_single_legacy_dec(void);
const char *mlkem_single_description(void);
#endif
