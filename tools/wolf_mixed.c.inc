/* AVX-Kyber mixed SHAKE expansion, for pinned wolfSSL 5.9.4.
 * SHAKE256(r || nonce=0..6) supplies y/e1/e2; SHAKE128(rho || 2 || 2)
 * supplies matrix entry nine. Rates, padding and rejection refills differ.
 * The raw permutation, redistribution, CBD and samplers remain upstream.
 * There are no experiment switches or mutable profiling counters. */
#if !defined(WOLFSSL_MLKEM_CACHE_A) || defined(WOLFSSL_MLKEM_DYNAMIC_KEYS) || \
    defined(WOLFSSL_SMALL_STACK) || defined(WOLFSSL_MLKEM_ENCAPSULATE_SMALL_MEM)
#error "AVX-Kyber mixed SHAKE requires its pinned static-key ML-KEM configuration"
#endif

#if defined(WOLFSSL_MLKEM_HAVE_INTEL_AVX512) && defined(WOLFSSL_WC_ML_KEM_768)
static void avxkyber_scalar_permute(word64* state)
{
#ifndef WC_SHA3_NO_ASM
    if (IS_INTEL_BMI2(cpuid_flags)) sha3_block_bmi2(state);
    else if (IS_INTEL_AVX2(cpuid_flags)) sha3_block_avx2(state);
    else
#endif
        BlockSha3(state);
}
#endif

__attribute__((visibility("hidden")))
int avxkyber_mlkem_mixed_noise_matrix(MLKEM_PRF_T* prf, sword16* a,
    sword16* y, sword16* e1, sword16* e2, byte* rho, byte* r)
{
#if defined(WOLFSSL_MLKEM_HAVE_INTEL_AVX512) && defined(WOLFSSL_WC_ML_KEM_768)
    if (USE_INTEL_AVX512(cpuid_flags) && SAVE_VECTOR_REGISTERS2() == 0) {
        word64 state[25 * 8] = {0}, single[25], rw[4], pw[4];
        byte noise[8 * 128], matrix[GEN_MATRIX_SIZE + 4];
        unsigned int ctr;
        int rc = avxkyber_matrix_k3(a, rho, 1, 1);
        if (rc) {
            RESTORE_VECTOR_REGISTERS();
            return rc;
        }
        readUnalignedWords64(rw, r, 4);
        readUnalignedWords64(pw, rho, 4);
        for (int w = 0; w < 4; w++)
            for (int lane = 0; lane < 8; lane++)
                state[w * 8 + lane] = lane == 7 ? pw[w] : rw[w];
        for (int lane = 0; lane < 7; lane++) {
            state[4 * 8 + lane] = 0x1f00 + lane;
            state[16 * 8 + lane] = W64LIT(0x8000000000000000);
        }
        state[4 * 8 + 7] = 0x1f0202;
        state[20 * 8 + 7] = W64LIT(0x8000000000000000);
        sha3_blocksx8_avx512(state);
        mlkem_redistribute_16_rand_x8_avx512(state, noise, 128);
        for (int i = 0; i < 3; i++) {
            mlkem_cbd_eta2_ins(y + i * MLKEM_N, noise + i * 128);
            mlkem_cbd_eta2_ins(e1 + i * MLKEM_N, noise + (i + 3) * 128);
        }
        mlkem_cbd_eta2_ins(e2, noise + 6 * 128);

        /* Continue all 25 words, not only the first output block. */
        for (int w = 0; w < 25; w++) single[w] = state[w * 8 + 7];
        XMEMCPY(matrix, single, SHA3_128_BYTES);
        for (int off = SHA3_128_BYTES; off < GEN_MATRIX_SIZE; off += SHA3_128_BYTES) {
            avxkyber_scalar_permute(single);
            XMEMCPY(matrix + off, single, SHA3_128_BYTES);
        }
        /* Preserve the upstream sampler's readable overrun bytes. */
        XMEMSET(matrix + GEN_MATRIX_SIZE, 0xff, 4);
        a += 8 * MLKEM_N;
        ctr = mlkem_rej_uniform_n_ins(a, MLKEM_N, matrix, GEN_MATRIX_SIZE);
        while (ctr < MLKEM_N) {
            avxkyber_scalar_permute(single);
            XMEMCPY(matrix, single, SHA3_128_BYTES);
            ctr += mlkem_rej_uniform_ins(a + ctr, MLKEM_N - ctr, matrix, SHA3_128_BYTES);
        }
        ForceZero(state, sizeof state);
        ForceZero(noise, sizeof noise);
        ForceZero(rw, sizeof rw);
        /* single, matrix and pw contain only the public rho expansion. */
        RESTORE_VECTOR_REGISTERS();
        return 0;
    }
#endif
    /* Keep upstream behavior if the SIMD guard or register save declines. */
    int rc = mlkem_get_noise(prf, 3, y, e1, e2, r);
    return rc ? rc : mlkem_gen_matrix(prf, a, 3, rho, 1);
}
