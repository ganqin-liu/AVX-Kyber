/* Pinned wolfSSL internal matrix-preparation API; see build_single.py.
 * Upstream GPL/other component notices are retained with fetched libraries. */
#include "mlkem_single.h"
#include <oqs/oqs.h>
#include <wolfssl/wolfcrypt/wc_port.h>
#include <cpuid.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "kem.h"
#if !defined(WOLFSSL_MLKEM_CACHE_A) || defined(WOLFSSL_MLKEM_DYNAMIC_KEYS) || defined(WOLFSSL_SMALL_STACK)
#error "single backend requires pinned static ML-KEM key/hash storage"
#endif
static pthread_once_t once=PTHREAD_ONCE_INIT;
static int enc_wolf, legacy, init_error;
static char description[128];
static _Thread_local MlKemKey peer;
static _Thread_local int peer_ready;
static void setup(void) {
    unsigned a,b,c,d;char vendor[13]={0};
    __cpuid(0,a,b,c,d);memcpy(vendor,&b,4);memcpy(vendor+4,&d,4);memcpy(vendor+8,&c,4);
    enc_wolf=!strcmp(vendor,"AuthenticAMD");
    const char *e=getenv("AVX_KYBER_SINGLE_ENC"),*dec=getenv("AVX_KYBER_SINGLE_DEC");
    if(e){if(!strcmp(e,"wolf"))enc_wolf=1;else if(!strcmp(e,"oqs"))enc_wolf=0;else if(!strcmp(e,"legacy"))enc_wolf=2;else init_error=1;}
    if(dec){if(!strcmp(dec,"legacy"))legacy=1;else if(strcmp(dec,"wolf"))init_error=1;}
    if(wolfCrypt_Init()!=0)init_error=1;
    OQS_init();
    snprintf(description,sizeof description,"enc=%s dec=%s static_key=encoded_import_prepared",
        enc_wolf==2?"pqcrystals-avx2":enc_wolf?"wolf-avx512-mixed-fused":"oqs-avx512vl",legacy?"cached-avx2":"wolf-avx512-fused");
}
static int init_peer(void) {
    pthread_once(&once,setup);if(init_error)return -1;
    if(!peer_ready){int rc=wc_MlKemKey_Init(&peer,WC_ML_KEM_768,NULL,INVALID_DEVID);if(rc)return rc;peer_ready=1;}
    return 0;
}
int mlkem_single_prepare(mlkem_single_key *s,const uint8_t sk[2400]) {
    /* Rebinding always starts from empty state: DecodePrivateKey alone does
     * not clear upstream A_SET. No stale identity survives an import error. */
    memset(s,0,sizeof *s);s->status=-1;
    if(init_peer())return -1;
    int rc=wc_MlKemKey_Init(&s->key,WC_ML_KEM_768,NULL,INVALID_DEVID);
    if(!rc)rc=wc_MlKemKey_DecodePrivateKey(&s->key,sk,2400);
    if(!rc){mlkem_prf_init(&s->key.prf);rc=mlkem_gen_matrix(&s->key.prf,s->key.a,3,s->key.pubSeed,0);}
    if(!rc)s->key.flags|=MLKEM_FLAG_A_SET;
    else wc_MlKemKey_Free(&s->key);
    s->status=rc;return rc;
}
void mlkem_single_clear(mlkem_single_key *s) {
    wc_MlKemKey_Free(&s->key);
    for(volatile unsigned char *p=(volatile unsigned char*)s;p<(volatile unsigned char*)(s+1);p++)*p=0;
    s->status=-1;
}
void mlkem_single_thread_clear(void) {
    if(peer_ready)wc_MlKemKey_Free(&peer);
    for(volatile unsigned char *p=(volatile unsigned char*)&peer;p<(volatile unsigned char*)(&peer+1);p++)*p=0;
    peer_ready=0;
}
int mlkem_single_enc(uint8_t ct[1088],uint8_t ss[32],const uint8_t pk[1184],const uint8_t coins[32]) {
    if(init_peer())return -1;
    if(enc_wolf==2)return crypto_kem_enc_derand(ct,ss,pk,coins);
    if(!enc_wolf)return OQS_KEM_ml_kem_768_encaps_derand(ct,ss,pk,coins);
    int rc=wc_MlKemKey_DecodePublicKey(&peer,pk,1184);
    return rc?rc:wc_MlKemKey_EncapsulateWithRandom(&peer,ct,ss,coins,32);
}
int mlkem_single_dec(uint8_t ss[32],const uint8_t ct[1088],mlkem_single_key *s) {
    return s->status?s->status:wc_MlKemKey_Decapsulate(&s->key,ss,ct,1088);
}
int mlkem_single_legacy_dec(void) {pthread_once(&once,setup);return legacy;}
const char *mlkem_single_description(void) {pthread_once(&once,setup);return description;}
