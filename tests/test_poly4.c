#include "poly4.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

extern void pqcrystals_kyber768_ref_ntt(int16_t *);
extern void pqcrystals_kyber768_ref_invntt(int16_t *);
extern void pqcrystals_kyber768_ref_basemul(int16_t *,const int16_t *,const int16_t *,int16_t);
extern const int16_t pqcrystals_kyber768_ref_zetas[128];
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(1); } } while (0)
static uint32_t rng = 0x861c42d3;
static uint32_t random32(void) { rng ^= rng<<13; rng ^= rng>>17; rng ^= rng<<5; return rng; }
static int canonical(int x) { int r=x%3329; return r<0 ? r+3329 : r; }

int main(void)
{
    int16_t a[4][256], out[4][256], ref[256];
    const int16_t *in[4];
    int16_t *op[4];
    poly4 f;
    poly4_init_tables();
    for (int l=0;l<4;l++) { in[l]=a[l]; op[l]=out[l]; }
    for (int l=0;l<4;l++) for(int j=0;j<256;j++) a[l][j]=1024*l+j;
    poly4_fold(&f,in);
    for (int r=0;r<32;r++) {
        int16_t slots[32];
        _mm512_storeu_si512(slots,f.v[r]);
        for (int l=0;l<4;l++) for(int j=0;j<8;j++) CHECK(slots[8*l+j]==a[l][8*r+j]);
    }
    poly4_unfold(op,&f);
    CHECK(memcmp(a,out,sizeof(a))==0);
    for (int inverse=0;inverse<2;inverse++) for (int t=0;t<256;t++) {
        int bound=inverse ? 16383 : 3329;
        for (int l=0;l<4;l++) for(int j=0;j<256;j++)
            a[l][j] = t<8 ? (((j>>(t%4))^l^(t/4))&1 ? bound : -bound)
                          : (int)(random32()%(2*bound+1))-bound;
        poly4_fold(&f,in);
        if(inverse) poly4_invntt(&f); else poly4_ntt(&f);
        poly4_unfold(op,&f);
        for (int l=0;l<4;l++) {
            memcpy(ref,a[l],sizeof(ref));
            if(inverse) pqcrystals_kyber768_ref_invntt(ref); else pqcrystals_kyber768_ref_ntt(ref);
            for(int j=0;j<256;j++) {
                CHECK(canonical(out[l][j])==canonical(ref[j]));
                CHECK(abs(out[l][j]) <= (inverse ? 26632 : 3329));
            }
        }
    }
    for(int t=0;t<256;t++) {
        int16_t av[3][4][256],bv[3][4][256];
        fop4 af[3]; poly4 bf[3];
        for(int k=0;k<3;k++) {
            for(int l=0;l<4;l++) for(int j=0;j<256;j++) {
                av[k][l][j]=t<16 ? (((j^(t>>1)^l)&1) ? 4095 : -4095) : (int)(random32()%8191)-4095;
                bv[k][l][j]=t<16 ? (((j^(t>>2)^k)&1) ? 3329 : -3329) : (int)(random32()%6659)-3329;
            }
            for(int l=0;l<4;l++) in[l]=av[k][l];
            fop4_from_polys(&af[k],in);
            for(int l=0;l<4;l++) in[l]=bv[k][l];
            poly4_fold(&bf[k],in);
        }
        poly4_basemul_acc(&f,af,bf);
        poly4_unfold(op,&f);
        for(int l=0;l<4;l++) {
            int accum[256]={0};
            for(int k=0;k<3;k++) for(int j=0;j<128;j++) {
                int16_t z=pqcrystals_kyber768_ref_zetas[64+j/2];
                int16_t pair[2];
                if(j&1) z=-z;
                pqcrystals_kyber768_ref_basemul(pair,av[k][l]+2*j,bv[k][l]+2*j,z);
                accum[2*j]+=pair[0]; accum[2*j+1]+=pair[1];
            }
            for(int j=0;j<256;j++) {
                CHECK(canonical(out[l][j])==canonical(accum[j]));
                CHECK(abs(out[l][j])<=2913);
            }
        }
    }
    puts("{\"check\":\"poly4\",\"layout\":true,\"forward_vectors\":1024,\"inverse_vectors\":1024,\"basemul_vectors\":1024,\"boundaries\":true}");
    return 0;
}
