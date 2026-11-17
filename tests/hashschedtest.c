/* Equivalence gate for interleaved Keccak and fixed ML-KEM-768 scheduling. */
#include <stdio.h>
#include <string.h>
#include "fips202x8.h"
#include "mlkem_hash_spwords.h"

static uint64_t rng = 0x9e3779b97f4a7c15ULL;
static void fill(void *buf, size_t n, int pattern)
{
    uint8_t *p = buf;
    for (size_t i = 0; i < n; i++)
    {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        p[i] = pattern == 0 ? 0 : pattern == 1 ? 255 : (uint8_t)rng;
    }
}

static int guard(const uint8_t *p,size_t n)
{
    for(unsigned i=0;i<7;i++)if(p[i]!=0xA5||p[7+n+i]!=0xA5)return 1;
    return 0;
}
static int check_spwords(const uint8_t *gp[8],const uint8_t *cp[8],const uint8_t z[32],
                         uint8_t kr[8][64],uint8_t nb[7][8][136],uint8_t job[8][136])
{
    struct { _Alignas(64) uint8_t before[64];__m512i words[3][16];uint8_t after[64]; } out;
    uint8_t k[8][78],tail[4][8][150],j[8][150];
    uint8_t savedg[8][64],savedc[8][1088],savedz[32];
    uint8_t *kp[8],*tp[4][8],*jp[8];
    memset(&out,0xA5,sizeof out);memset(k,0xA5,sizeof k);
    memset(tail,0xA5,sizeof tail);memset(j,0xA5,sizeof j);memcpy(savedz,z,32);
    for(unsigned l=0;l<8;l++) {
        kp[l]=k[l]+7;jp[l]=j[l]+7;
        memcpy(savedg[l],gp[l],64);memcpy(savedc[l],cp[l],1088);
        for(unsigned n=0;n<4;n++)tp[n][l]=tail[n][l]+7;
    }
    mlkem768x8_hashes_spwords(kp,out.words,tp,jp,gp,z,cp);
    for(unsigned l=0;l<8;l++) {
        if(memcmp(kp[l],kr[l],64)||memcmp(jp[l],job[l],136)||
           guard(k[l],64)||guard(j[l],136)||memcmp(gp[l],savedg[l],64)||
           memcmp(cp[l],savedc[l],1088))return 1;
        for(unsigned n=0;n<4;n++)
            if(memcmp(tp[n][l],nb[n+3][l],136)||guard(tail[n][l],136))return 1;
    }
    for(unsigned n=0;n<3;n++)for(unsigned w=0;w<16;w++) {
        uint64_t actual[8];memcpy(actual,&out.words[n][w],sizeof actual);
        for(unsigned l=0;l<8;l++) {
            uint64_t want=0;for(unsigned b=0;b<8;b++)want|=(uint64_t)nb[n][l][8*w+b]<<(8*b);
            if(actual[l]!=want)return 1;
        }
    }
    if(memcmp(z,savedz,32))return 1;
    for(unsigned i=0;i<64;i++)if(out.before[i]!=0xA5||out.after[i]!=0xA5)return 1;
    return 0;
}

int main(void)
{
    uint8_t gin[8][65], ct[8][1089], z[32];
    uint8_t kr[2][8][64], nb[2][7][8][136], job[2][8][136];
    const uint8_t *gp[8], *cp[8], *seed[8];
    uint8_t *kp[8], *np[7][8], *jp[8];
    keccakx8_state a, b, ra, rb, st;
    __m512i sw[4];
    for (int t = 0; t < 66; t++)
    {
        fill(&a, sizeof a, t); fill(&b, sizeof b, t);
        ra = a; rb = b;
        keccakx8_f1600(&ra); keccakx8_f1600(&rb);
        keccakx8_f1600x2(&a, &b);
        if (memcmp(&a, &ra, sizeof a) || memcmp(&b, &rb, sizeof b))
            return fprintf(stderr, "raw x2 differs at %d\n", t), 1;
        fill(gin, sizeof gin, t); fill(ct, sizeof ct, t); fill(z, sizeof z, t);
        memset(kr[0], 0xa5, sizeof kr[0]); memset(kr[1], 0x5a, sizeof kr[1]);
        memset(nb[0], 0xa5, sizeof nb[0]); memset(nb[1], 0x5a, sizeof nb[1]);
        memset(job[0], 0xa5, sizeof job[0]); memset(job[1], 0x5a, sizeof job[1]);
        for (int l = 0; l < 8; l++)
        {
            gp[l] = gin[t%5==0?0:l]+(t+l)%2; cp[l] = ct[t%7==0?0:l]+(t+l)%2;
            kp[l] = kr[0][l]; jp[l] = job[0][l];
            for (int n = 0; n < 7; n++) np[n][l] = nb[0][n][l];
        }
        mlkem768x8_hashes(kp, np, jp, gp, z, cp);
        if(check_spwords(gp,cp,z,kr[0],nb[0],job[0]))
            return fprintf(stderr,"spwords/hash/guard/input differs at %d\n",t),1;
        for (int l = 0; l < 8; l++)
        {
            kp[l] = kr[1][l]; jp[l] = job[1][l]; seed[l] = kr[1][l] + 32;
            for (int n = 0; n < 7; n++) np[n][l] = nb[1][n][l];
        }
        sha3_512x8_64B(kp, gp);
        keccakx8_seed32_transpose(sw, seed);
        for (int n = 0; n < 7; n++)
        {
            shake256x8_absorb_seed32_nonce(&st, sw, (uint8_t)n);
            shake256x8_squeezeblocks(np[n], 1, &st);
        }
        shake256x8_absorb_z32_ct(&st, z, cp, 1088);
        shake256x8_squeezeblocks(jp, 1, &st);
        if (memcmp(kr[0], kr[1], sizeof kr[0]) ||
            memcmp(nb[0], nb[1], sizeof nb[0]) ||
            memcmp(job[0], job[1], sizeof job[0]))
            return fprintf(stderr, "hash schedule differs at %d\n", t), 1;
    }
    puts("hashschedtest: PASS (66 patterns, full states, G/PRF/J and spwords/layout/guards)");
    return 0;
}
