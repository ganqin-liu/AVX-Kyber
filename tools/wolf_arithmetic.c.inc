
/* ML-KEM-768 arithmetic for the pinned Linux/SysV AVX-512 build.
 * The original NTT and inverse remain in wc_mlkem_asm.S. See the range
 * derivation and differential evidence in log/arithmetic_width_20261101.
 */
#if defined(WOLFSSL_MLKEM_HAVE_INTEL_AVX512) && defined(__linux__) && defined(__x86_64__)
#include <immintrin.h>
static const sword16 avxkyber_arith_zeta[256] __attribute__((aligned(64))) = {VN_ZETAS};

static inline __m512i avxkyber_arith_mont16(__m512i a, __m512i b)
{
    __m512i lo = _mm512_mullo_epi16(a, b);
    __m512i hi = _mm512_mulhi_epi16(a, b);
    __m512i t = _mm512_mullo_epi16(lo, _mm512_set1_epi16(-3327));
    return _mm512_sub_epi16(hi,
        _mm512_mulhi_epi16(t, _mm512_set1_epi16(3329)));
}
static inline __m512i avxkyber_arith_mont32(__m512i a)
{
    /* Only the low word of each dword is consumed below. Since a and m*q
     * have equal low words, REDC(a)=high16(a)-high16(m*q), without a carry.
     * Their signed16 difference is representable under the stated bounds.
     * The upper word of each returned dword is intentionally unspecified. */
    __m512i m = _mm512_mullo_epi16(a, _mm512_set1_epi16(-3327));
    return _mm512_sub_epi16(_mm512_srli_epi32(a, 16),
        _mm512_mulhi_epi16(m, _mm512_set1_epi16(3329)));
}

/* Only the guarded k=3 assembly call may use this helper: y is an eta2 NTT
 * output, |y|<=13592; |a/pub|<=4095. The largest int32 accumulator is
 * 333955440; corrected Montgomery numerator <=443040112. Output bounds
 * 4654 even / 6761 odd satisfy the unchanged signed16 inverse contract.
 * Twiddles carry R, so cached t=M(yodd*zeta) has y's scale; output is R^-1.
 */
void avxkyber_mlkem_arithmetic(const sword16*, sword16*, sword16*,
                            const sword16*, const sword16*);
__attribute__((visibility("hidden"), noinline))
void avxkyber_mlkem_arithmetic(const sword16 *pub, sword16 *u, sword16 *v,
                            const sword16 *a, const sword16 *y)
{
    sword16 cache[3][128] __attribute__((aligned(64)));
    for (int p = 0; p < 3; ++p) for (int j = 0; j < 128; j += 16) {
        __m512i yy = _mm512_loadu_si512(y + 256*p + 2*j);
        __m512i zz = _mm512_load_si512(avxkyber_arith_zeta + 2*j);
        __m512i t = avxkyber_arith_mont16(yy, zz);
        _mm256_store_si256((__m256i*)(cache[p] + j),
            _mm512_cvtepi32_epi16(_mm512_srai_epi32(t, 16)));
    }
    /* Three matrix rows and the public-key dot product reuse the same cache. */
    for (int row = 0; row < 4; ++row) {
        const sword16 *ar = row == 3 ? pub : a + 768*row;
        sword16 *out = row == 3 ? v : u + 256*row;
        for (int j = 0; j < 128; j += 16) {
            __m512i ev = _mm512_setzero_si512(), od = ev;
            for (int p = 0; p < 3; ++p) {
                __m512i aa = _mm512_loadu_si512(ar + 256*p + 2*j);
                __m512i yy = _mm512_loadu_si512(y + 256*p + 2*j);
                __m512i t = _mm512_cvtepi16_epi32(
                    _mm256_load_si256((const __m256i*)(cache[p] + j)));
                __m512i yp = _mm512_mask_mov_epi16(yy, 0xaaaaaaaaU,
                    _mm512_slli_epi32(t, 16));
                ev = _mm512_add_epi32(ev, _mm512_madd_epi16(aa, yp));
                od = _mm512_add_epi32(od,
                    _mm512_madd_epi16(aa, _mm512_rol_epi32(yy, 16)));
            }
            ev = avxkyber_arith_mont32(ev);
            od = _mm512_slli_epi32(avxkyber_arith_mont32(od), 16);
            _mm512_storeu_si512(out + 2*j,
                _mm512_mask_blend_epi16(0xaaaaaaaaU, ev, od));
        }
    }
    ForceZero(cache, sizeof cache);
}
#endif
