/* Finite production Encaps-flow differential and forced-refill checks. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mlkem_enc_flow.h"
#include "fips202.h"
#include "indcpa.h"
static void fill(uint8_t *p, size_t n, uint32_t s)
{
    s = s * 2654435761u + 1;
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        p[i] = (uint8_t)s;
    }
}
static void must(int ok, const char *why)
{
    if (!ok) {
        fprintf(stderr, "FAIL %s\n", why);
        exit(1);
    }
}
static unsigned accepted(const uint8_t *b, size_t n)
{
    unsigned count = 0;
    for (size_t i = 0; i + 3 <= n; i += 3) {
        count += ((b[i] | ((b[i + 1] & 15) << 8)) < 3329);
        count += (((b[i + 1] >> 4) | (b[i + 2] << 4)) < 3329);
    }
    return count;
}
static void flow_check(void)
{
    uint8_t public[8][1184], random[8][32], key[8][32], noise[7][8][136], rare[3][32];
    uint8_t *kp[8], *np[7][8];
    const uint8_t *pp[8], *rp[8];
    polyvec matrix[8][3], ref[3], *mat[8];
    const int elements[3] = {0, 5, 8};
    for (int t = 0; t < 3; t++) {
        uint8_t ext[34], buf[504];
        unsigned attempt;
        for (attempt = 0; attempt < 100000; attempt++) {
            fill(ext, 32, 80000 + t * 100000 + attempt);
            ext[32] = elements[t] / 3;
            ext[33] = elements[t] % 3;
            shake128(buf, sizeof buf, ext, sizeof ext);
            if (accepted(buf, sizeof buf) < 256) {
                memcpy(rare[t], ext, 32);
                break;
            }
        }
        must(attempt < 100000, "find deterministic refill seed");
    }
    for (int fixture = 0; fixture < 30; fixture++) {
        for (int l = 0; l < 8; l++) {
            fill(public[l], 1184, 500 + fixture * 8 + l);
            fill(random[l], 32, 700 + fixture * 8 + l);
            if (fixture < 24 && l == fixture % 8)
                memcpy(public[l] + 1152, rare[fixture / 8], 32);
            if (fixture >= 24)
                memcpy(public[l] + 1152, rare[(fixture + l) % 3], 32);
            if (fixture == 29)
                memcpy(public[l] + 1152, rare[0], 32);
            kp[l] = key[l];
            pp[l] = public[l];
            rp[l] = random[l];
            mat[l] = matrix[l];
            for (int j = 0; j < 7; j++)
                np[j][l] = noise[j][l];
        }
        for (int arm = 0; arm < 1; arm++) {
            mlkem_enc_hash_flow(kp, np, mat, pp, rp);
            for (int l = 0; l < 8; l++) {
                uint8_t digest[32], g[64], input[64], seed[33], nb[136];
                sha3_256(digest, public[l], 1184);
                memcpy(input, random[l], 32);
                memcpy(input + 32, digest, 32);
                sha3_512(g, input, 64);
                must(!memcmp(key[l], g, 32), "G first half");
                memcpy(seed, g + 32, 32);
                for (int j = 0; j < 7; j++) {
                    seed[32] = j;
                    shake256(nb, 136, seed, 33);
                    must(!memcmp(nb, noise[j][l], 136), "seven independent PRFs");
                }
                gen_matrix(ref, public[l] + 1152, 1);
                must(!memcmp(ref, matrix[l], sizeof ref),
                     "matrix scalar SHAKE/reference packed polys including refill");
            }
        }
    }
    puts("flowcheck PASS: H/G/7 PRFs/matrix, early/middle/late refill seeds rotated across all "
         "lanes, simultaneous short lanes");
}
int main(void)
{
    flow_check();
    return 0;
}
