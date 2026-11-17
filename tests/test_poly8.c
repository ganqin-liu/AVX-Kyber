/* Unit tests: folded-8 kernels vs the ref implementation (mod-q equality;
 * internal reduction schedules differ, values may differ by multiples of q). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poly8.h"

#define Q 3329
#define N 256

void pqcrystals_kyber768_ref_ntt(int16_t r[N]);
void pqcrystals_kyber768_ref_invntt(int16_t r[N]);
extern const int16_t pqcrystals_kyber768_ref_zetas[128];

/* scalar mirrors for basemul cross-check */
static int16_t s_fqmul(int16_t a, int16_t b)
{
    int32_t t = (int32_t)a * b;
    int16_t m = (int16_t)((int16_t)t * (int16_t)-3327);
    return (int16_t)((t - (int32_t)m * Q) >> 16);
}

static void s_basemul(int16_t r[2], const int16_t a[2], const int16_t b[2],
                      int16_t zeta)
{
    r[0] = s_fqmul(a[1], b[1]);
    r[0] = s_fqmul(r[0], zeta);
    r[0] += s_fqmul(a[0], b[0]);
    r[1] = s_fqmul(a[0], b[1]);
    r[1] += s_fqmul(a[1], b[0]);
}

static void s_basemul_poly(int16_t r[N], const int16_t a[N],
                           const int16_t b[N])
{
    int i;
    for (i = 0; i < N / 4; i++)
    {
        s_basemul(r + 4 * i, a + 4 * i, b + 4 * i,
                  pqcrystals_kyber768_ref_zetas[64 + i]);
        s_basemul(r + 4 * i + 2, a + 4 * i + 2, b + 4 * i + 2,
                  (int16_t)-pqcrystals_kyber768_ref_zetas[64 + i]);
    }
}

static int modq_eq(const int16_t *a, const int16_t *b, int n,
                   const char *tag, int lane)
{
    int i;
    for (i = 0; i < n; i++)
    {
        int d = ((int)a[i] - (int)b[i]) % Q;
        if (d < 0)
            d += Q;
        if (d != 0)
        {
            printf("[FAIL] %s lane %d idx %d: %d vs %d\n", tag, lane, i,
                   a[i], b[i]);
            return 1;
        }
    }
    return 0;
}

static int16_t polys[8][N], refp[8][N], outp[8][N];
static const int16_t *cp[8];
static int16_t *op[8];

static void randomize(int bound)
{
    int l, i;
    for (l = 0; l < 8; l++)
        for (i = 0; i < N; i++)
            polys[l][i] = (int16_t)((rand() % (2 * bound + 1)) - bound);
}

/* All operand layouts and the complete basemul->invNTT chain. */
static int basemul_edges(void)
{
    static int16_t a[3][8][N], b[3][8][N], want[8][N], tmp[N];
    static fop8 full[3];
    static fop8c compact[3];
    static fop8c_finv scaled[3];
    poly8 bf[3], got;
    const int16_t *ap[8], *bp[8];
    int rc = 0;
    for (int mode = 0; mode < 4; mode++)
        for (int pattern = 0; pattern < 32; pattern++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int l = 0; l < 8; l++)
                    for (int i = 0; i < N; i++)
                    {
                        int v = pattern < 12 ? 4095 :
                                pattern < 20 ? 3329 + rand() % 767 : rand() % 4096;
                        int w = pattern < 20 ? Q - pattern % 3 : rand() % (Q + 1);
                        int sa = pattern < 2 ? pattern :
                                 pattern < 12 ? ((i >> ((pattern - 2) % 8)) ^ j ^ l) & 1 : rand() & 1;
                        int sb = pattern < 4 ? (pattern >> 1) :
                                 pattern < 12 ? ((i >> ((pattern - 2) % 8)) ^ l) & 1 : rand() & 1;
                        a[j][l][i] = (int16_t)(sa ? -v : v);
                        b[j][l][i] = (int16_t)(sb ? -w : w);
                    }
                if (mode != 2)
                    for (int l = 1; l < 8; l++)
                        memcpy(a[j][l], a[j][0], sizeof a[j][0]);
                for (int l = 0; l < 8; l++)
                {
                    ap[l] = a[j][l]; bp[l] = b[j][l];
                }
                poly8_fold(&bf[j], bp);
                if (mode == 0) fop8_from_poly(&full[j], a[j][0]);
                if (mode == 1) fop8c_from_poly(&compact[j], a[j][0]);
                if (mode == 2) fop8_from_polys(&full[j], ap);
                if (mode == 3) {
                    fop8c_finv_from_poly(&scaled[j], a[j][0]);
                    for (int v = 0; v < 64; v++) for (int k = 0; k < 4; k++)
                        if (abs((int16_t)(scaled[j].v[v] >> (16*k))) > 1754 ||
                            abs((int16_t)(scaled[j].w[v] >> (16*k))) > 1754) return 1;
                }
            }
            for (int l = 0; l < 8; l++)
            {
                memset(want[l], 0, sizeof want[l]);
                for (int j = 0; j < 3; j++)
                {
                    s_basemul_poly(tmp, a[j][l], b[j][l]);
                    for (int i = 0; i < N; i++) want[l][i] += tmp[i];
                }
            }
            if (mode == 3) poly8_basemul_acc_bc_finv(&got, scaled, bf);
            else if (mode == 1) poly8_basemul_acc_bc(&got, compact, bf);
            else poly8_basemul_acc(&got, full, bf);
            poly8_unfold(op, &got);
            for (int l = 0; l < 8; l++)
            {
                for (int i = 0; i < N; i++)
                    tmp[i] = mode == 3 ? s_fqmul(want[l][i],1441) : want[l][i];
                rc |= modq_eq(outp[l], tmp, N, "basemul edges", l);
                for (int i = 0; i < N; i++)
                {
                    if (abs(outp[l][i]) > (mode == 3 ? POLY8_FINV_BASEMUL_BOUND : POLY8_BASEMUL_OUTPUT_BOUND)) rc = 1;
                    int v = want[l][i] % Q;
                    if (v > Q / 2) v -= Q;
                    if (v < -Q / 2) v += Q;
                    want[l][i] = (int16_t)v;
                }
                pqcrystals_kyber768_ref_invntt(want[l]);
            }
            if (mode == 3) poly8_invntt_finv(&got);
            else poly8_invntt(&got);
            poly8_unfold(op, &got);
            for (int l = 0; l < 8; l++)
            {
                rc |= modq_eq(outp[l], want[l], N, "basemul->invntt", l);
                for (int i = 0; i < N; i++)
                    if (abs(outp[l][i]) > POLY8_INVNTT_OUTPUT_BOUND) rc = 1;
            }
        }
    return rc;
}

int main(void)
{
    poly8 f;
    int l, i, rc = 0, iter;

    poly8_init_tables();
    srand(12345);
    for (l = 0; l < 8; l++)
    {
        cp[l] = polys[l];
        op[l] = outp[l];
    }

    for (iter = 0; iter < 50; iter++)
    {
        /* fold/unfold roundtrip (bit-exact) */
        randomize(Q);
        poly8_fold(&f, cp);
        poly8_unfold(op, &f);
        for (l = 0; l < 8; l++)
            if (memcmp(outp[l], polys[l], sizeof polys[l]))
            {
                printf("[FAIL] fold roundtrip lane %d\n", l);
                rc = 1;
            }

        /* forward NTT (input range as in decaps: [0,q)) */
        randomize(0);
        for (l = 0; l < 8; l++)
            for (i = 0; i < N; i++)
                polys[l][i] = (int16_t)(iter & 1 ? rand() % 5 - 2 : rand() % Q);
        memcpy(refp, polys, sizeof polys);
        poly8_fold(&f, cp);
        poly8_ntt(&f);
        poly8_unfold(op, &f);
        for (l = 0; l < 8; l++)
        {
            pqcrystals_kyber768_ref_ntt(refp[l]);
            rc |= modq_eq(outp[l], refp[l], N, "ntt", l);
            for (i = 0; i < N; i++)
                if (outp[l][i] < 0 || outp[l][i] > Q) rc = 1;
        }

        /* inverse NTT (input: basemul-acc-like range) */
        randomize(3 * Q);
        memcpy(refp, polys, sizeof polys);
        poly8_fold(&f, cp);
        poly8_invntt(&f);
        poly8_unfold(op, &f);
        for (l = 0; l < 8; l++)
        {
            pqcrystals_kyber768_ref_invntt(refp[l]);
            rc |= modq_eq(outp[l], refp[l], N, "invntt", l);
        }

        /* basemul_acc: shared operand (replicated), 3-way accumulate */
        {
            static int16_t a_op[3][N], b_in[3][8][N], want[8][N], t[N];
            poly8 bf3[3];
            fop8 opf[3];
            int k;

            for (k = 0; k < 3; k++)
            {
                for (i = 0; i < N; i++)
                    a_op[k][i] = (int16_t)(rand() % Q);
                fop8_from_poly(&opf[k], a_op[k]);
                for (l = 0; l < 8; l++)
                    for (i = 0; i < N; i++)
                        b_in[k][l][i] =
                            (int16_t)((rand() % (2 * 1664 + 1)) - 1664);
                for (l = 0; l < 8; l++)
                    cp[l] = b_in[k][l];
                poly8_fold(&bf3[k], cp);
            }
            for (l = 0; l < 8; l++)
            {
                s_basemul_poly(want[l], a_op[0], b_in[0][l]);
                for (k = 1; k < 3; k++)
                {
                    s_basemul_poly(t, a_op[k], b_in[k][l]);
                    for (i = 0; i < N; i++)
                        want[l][i] += t[i];
                }
            }
            poly8_basemul_acc(&f, opf, bf3);
            poly8_unfold(op, &f);
            for (l = 0; l < 8; l++)
                rc |= modq_eq(outp[l], want[l], N, "basemul_acc", l);
            for (l = 0; l < 8; l++)
                cp[l] = polys[l];
        }
    }

    /* Cover the proven production bound and the larger documented input
     * contract. Extremal same-sign / opposite-sign butterfly pairs ensure
     * random small coefficients cannot hide a 16-bit overflow. */
    const int bounds[] = {11238, 16383};
    for (int bi = 0; bi < 2; bi++)
        for (int pattern = 0; pattern < 28; pattern++)
        {
            for (l = 0; l < 8; l++)
                for (i = 0; i < N; i++)
                {
                    int sign = pattern < 2 ? pattern :
                        pattern < 10 ? ((i >> (pattern - 2)) & 1) :
                        (rand() & 1);
                    int value = pattern < 20 ? bounds[bi] : rand() % (bounds[bi] + 1);
                    polys[l][i] = (int16_t)((sign ^ (l & 1)) ? -value : value);
                }
            memcpy(refp, polys, sizeof polys);
            poly8_fold(&f, cp); poly8_invntt(&f); poly8_unfold(op, &f);
            for (l = 0; l < 8; l++)
            {
                /* Normalize only the oracle's copy in 32-bit arithmetic.
                 * The candidate keeps the original extrema. This avoids
                 * reference Barrett's implementation-defined narrowing at
                 * large sums, without changing the expected mod-q result. */
                for (i = 0; i < N; i++)
                {
                    int v = refp[l][i] % Q;
                    if (v > Q / 2) v -= Q;
                    if (v < -Q / 2) v += Q;
                    refp[l][i] = (int16_t)v;
                }
                pqcrystals_kyber768_ref_invntt(refp[l]);
                rc |= modq_eq(outp[l], refp[l], N, "invntt extrema", l);
                for (i = 0; i < N; i++)
                    if (abs(outp[l][i]) > POLY8_INVNTT_OUTPUT_BOUND) rc = 1;
            }
        }

    rc |= basemul_edges();
    printf(rc ? "test_poly8 FAILED\n" : "test_poly8 ok (50 iters + 96 basemul edge/layout cases)\n");
    return rc;
}
