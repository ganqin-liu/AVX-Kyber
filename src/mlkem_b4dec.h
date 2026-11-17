#ifndef AVX_KYBER_MLKEM_B4DEC_H
#define AVX_KYBER_MLKEM_B4DEC_H
#include "poly4.h"
#include "poly8.h"
typedef struct {fop8c a[3][3],t[3],s[3];uint8_t hpk[32],z[32];} mlkem_skh4;
void mlkem_skh4_init(mlkem_skh4 *,const uint8_t sk[2400]);
void mlkem_dec_b4_folded(uint8_t ss[4][32],const uint8_t *ct[4],const mlkem_skh4 *);
void mlkem_b4_hashes(uint8_t kr[4][64],uint8_t noise[7][4][160],uint8_t j[4][32],const uint8_t gin[4][64],const uint8_t z[32],const uint8_t *ct[4]);
void poly4_basemul_acc_bc(poly4 *,const fop8c op[3],const poly4 b[3]);
#endif
