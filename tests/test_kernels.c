/* Finite differential checks against pristine pinned pq-crystals ML-KEM. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/crypto.h>
#include "kem.h"
#include "mlkem_b8.h"
#include "mlkem_b8f.h"
#include "mlkem_b4.h"
extern int ls_pqcrystals_pqcrystals_kyber768_avx2_keypair_derand(uint8_t *, uint8_t *, const uint8_t *);
extern int ls_pqcrystals_pqcrystals_kyber768_avx2_enc_derand(uint8_t *, uint8_t *, const uint8_t *, const uint8_t *);
extern int ls_pqcrystals_pqcrystals_kyber768_avx2_dec(uint8_t *, const uint8_t *, const uint8_t *);
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(1); } } while (0)
static void fill(uint8_t *p,size_t n,unsigned s) {
 s+=1;for(size_t i=0;i<n;i++){s^=s<<13;s^=s>>17;s^=s<<5;p[i]=(uint8_t)s;}
}
int main(void) {
 static mlkem_skh h1;
 static mlkem_skhf h8;
 uint8_t pk[8][1184],sk[8][2400],coins[8][32],seed[64];
 uint8_t ct[8][1088],ss[8][32],outct[8][1088],outss[8][32],expected[32];
 const uint8_t *pp[8],*rp[8],*cp[8];
 for(unsigned fixture=0;fixture<8;fixture++) {
  for(int l=0;l<8;l++) {
   fill(seed,sizeof seed,1000+fixture*8+l);
   CHECK(!ls_pqcrystals_pqcrystals_kyber768_avx2_keypair_derand(pk[l],sk[l],seed));
   fill(coins[l],32,2000+fixture*8+l);pp[l]=pk[l];rp[l]=coins[l];cp[l]=ct[l];
   CHECK(!ls_pqcrystals_pqcrystals_kyber768_avx2_enc_derand(ct[l],ss[l],pk[l],coins[l]));
   CHECK(!mlkem_single_enc(outct[l],outss[l],pk[l],coins[l]));
   CHECK(!memcmp(ct[l],outct[l],1088)&&!memcmp(ss[l],outss[l],32));
  }
  mlkem_enc_b8(outct,outss,pp,rp);
  CHECK(!memcmp(ct,outct,sizeof ct)&&!memcmp(ss,outss,sizeof ss));
  mlkem_enc_b8_folded(outct,outss,pp,rp);
  CHECK(!memcmp(ct,outct,sizeof ct)&&!memcmp(ss,outss,sizeof ss));
  for(int start=0;start<8;start+=4) {
   mlkem_enc_b4_folded(outct+start,outss+start,pp+start,rp+start);
   CHECK(!memcmp(ct+start,outct+start,4*1088)&&!memcmp(ss+start,outss+start,4*32));
  }
  mlkem_skh_init(&h1,sk[0]);CHECK(!h1.single.status);
  mlkem_skhf_init(&h8,sk[0]);
  for(int l=0;l<8;l++)CHECK(!ls_pqcrystals_pqcrystals_kyber768_avx2_enc_derand(ct[l],ss[l],pk[0],coins[l]));
  for(unsigned mask=0;mask<256;mask++) {
   memcpy(outct,ct,sizeof ct);
   for(int l=0;l<8;l++)if(mask&(1u<<l))outct[l][(fixture*97+l*101)%1088]^=1;
   for(int l=0;l<8;l++)cp[l]=outct[l];
   mlkem_dec_b8_folded(outss,cp,&h8);
   for(int l=0;l<8;l++) {
    CHECK(!ls_pqcrystals_pqcrystals_kyber768_avx2_dec(expected,cp[l],sk[0]));
    CHECK(!memcmp(expected,outss[l],32));
    CHECK(!mlkem_single_dec(outss[l],cp[l],&h1.single));
    CHECK(!memcmp(expected,outss[l],32));
    CHECK(!mlkem_dec_cached(outss[l],cp[l],&h1));
    CHECK(!memcmp(expected,outss[l],32));
   }
   for(int start=0;start<8;start+=4) {
    mlkem_dec_b4_folded(outss+start,cp+start,&h1.b4);
    for(int l=start;l<start+4;l++) {
     CHECK(!ls_pqcrystals_pqcrystals_kyber768_avx2_dec(expected,cp[l],sk[0]));
     CHECK(!memcmp(expected,outss[l],32));
    }
   }
  }
  mlkem_single_clear(&h1.single);
  OPENSSL_cleanse(&h1,sizeof h1);OPENSSL_cleanse(&h8,sizeof h8);
 }
 mlkem_single_thread_clear();
 puts("PASS B1/B4/B8: 64 heterogeneous Encaps, 8 key rebindings, all 256 rejection masks per key vs pristine pq-crystals");
 puts(mlkem_single_description());return 0;
}
