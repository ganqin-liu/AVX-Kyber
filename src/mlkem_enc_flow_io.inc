/* Private Keccak IO helpers from fips202x8.c and rejection sampler from mlkem_b8.c.
 * Ported unchanged from the measured joint_followup_20261102 words candidate. */
static void transpose8x8_u64(__m512i r[8])
{
    __m512i t[8];

    t[0] = _mm512_unpacklo_epi64(r[0], r[1]);
    t[1] = _mm512_unpackhi_epi64(r[0], r[1]);
    t[2] = _mm512_unpacklo_epi64(r[2], r[3]);
    t[3] = _mm512_unpackhi_epi64(r[2], r[3]);
    t[4] = _mm512_unpacklo_epi64(r[4], r[5]);
    t[5] = _mm512_unpackhi_epi64(r[4], r[5]);
    t[6] = _mm512_unpacklo_epi64(r[6], r[7]);
    t[7] = _mm512_unpackhi_epi64(r[6], r[7]);

    r[0] = _mm512_shuffle_i64x2(t[0], t[2], 0x88); /* 128-bit lanes 0,2 */
    r[1] = _mm512_shuffle_i64x2(t[1], t[3], 0x88);
    r[2] = _mm512_shuffle_i64x2(t[0], t[2], 0xdd); /* 128-bit lanes 1,3 */
    r[3] = _mm512_shuffle_i64x2(t[1], t[3], 0xdd);
    r[4] = _mm512_shuffle_i64x2(t[4], t[6], 0x88);
    r[5] = _mm512_shuffle_i64x2(t[5], t[7], 0x88);
    r[6] = _mm512_shuffle_i64x2(t[4], t[6], 0xdd);
    r[7] = _mm512_shuffle_i64x2(t[5], t[7], 0xdd);

    t[0] = _mm512_shuffle_i64x2(r[0], r[4], 0x88);
    t[1] = _mm512_shuffle_i64x2(r[1], r[5], 0x88);
    t[2] = _mm512_shuffle_i64x2(r[2], r[6], 0x88);
    t[3] = _mm512_shuffle_i64x2(r[3], r[7], 0x88);
    t[4] = _mm512_shuffle_i64x2(r[0], r[4], 0xdd);
    t[5] = _mm512_shuffle_i64x2(r[1], r[5], 0xdd);
    t[6] = _mm512_shuffle_i64x2(r[2], r[6], 0xdd);
    t[7] = _mm512_shuffle_i64x2(r[3], r[7], 0xdd);

    for (int i = 0; i < 8; i++)
        r[i] = t[i];
}

static __m512i load_word8(const uint8_t *in[8], size_t off)
{
    uint64_t w[8];
    int l;
    for (l = 0; l < 8; l++)
        memcpy(&w[l], in[l] + off, 8);
    return _mm512_loadu_si512((const void *)w);
}

static void absorb_block(keccakx8_state *st, const uint8_t *in[8], size_t off,
                         size_t rate)
{
    size_t nwords = rate / 8;
    size_t w = 0;

    while (w + 8 <= nwords)
    {
        __m512i r[8];
        int l;
        for (l = 0; l < 8; l++)
            r[l] = _mm512_loadu_si512((const void *)(in[l] + off + 8 * w));
        transpose8x8_u64(r);
        for (l = 0; l < 8; l++)
            st->s[w + (size_t)l] = XOR(st->s[w + (size_t)l], r[l]);
        w += 8;
    }
    for (; w < nwords; w++)
        st->s[w] = XOR(st->s[w], load_word8(in, off + 8 * w));
}

static void store_block(uint8_t *out[8], size_t off, const keccakx8_state *st,
                        size_t rate)
{
    size_t nwords = rate / 8;
    size_t w = 0;

    while (w + 8 <= nwords)
    {
        __m512i r[8];
        int l;
        for (l = 0; l < 8; l++)
            r[l] = st->s[w + (size_t)l];
        transpose8x8_u64(r);
        /* after transpose, r[l] = 8 consecutive words of stream l */
        for (l = 0; l < 8; l++)
            _mm512_storeu_si512((void *)(out[l] + off + 8 * w), r[l]);
        w += 8;
    }
    for (; w < nwords; w++)
    {
        uint64_t words[8];
        int l;
        _mm512_storeu_si512((void *)words, st->s[w]);
        for (l = 0; l < 8; l++)
            memcpy(out[l] + off + 8 * w, &words[l], 8);
    }
}
static unsigned int rej_uniform(int16_t *r, unsigned int len,
                                const uint8_t *buf, unsigned int buflen)
{
    unsigned int ctr, pos;
    uint16_t val0, val1;

    ctr = pos = 0;
    while (ctr < len && pos <= buflen - 3)
    {
        val0 = ((buf[pos + 0] >> 0) | ((uint16_t)buf[pos + 1] << 8)) & 0xFFF;
        val1 = ((buf[pos + 1] >> 4) | ((uint16_t)buf[pos + 2] << 4)) & 0xFFF;
        pos += 3;

        if (val0 < KYBER_Q)
            r[ctr++] = (int16_t)val0;
        if (ctr < len && val1 < KYBER_Q)
            r[ctr++] = (int16_t)val1;
    }
    return ctr;
}
