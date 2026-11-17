#ifndef AVX_KYBER_ABLATION_CONFIG_H
#define AVX_KYBER_ABLATION_CONFIG_H

/* Build every translation unit with the same settings: the operand choice
 * changes the key-handle ABI. All choices retain FIPS 203 semantics and key
 * caching. Kind 0 disables shared-operand compression, not key reuse itself. */
#ifndef LS_OPERAND_KIND
#define LS_OPERAND_KIND 2
#endif
#ifndef LS_PRF_WORDS
#define LS_PRF_WORDS 1
#endif
#ifndef LS_ENCAP_FOLDED
#define LS_ENCAP_FOLDED 0
#endif
#if LS_OPERAND_KIND < 0 || LS_OPERAND_KIND > 2
#error "LS_OPERAND_KIND must be full=0, compact=1, or compact_FINV=2"
#endif
#if (LS_PRF_WORDS != 0 && LS_PRF_WORDS != 1) || (LS_ENCAP_FOLDED != 0 && LS_ENCAP_FOLDED != 1)
#error "ablation switches must be zero or one"
#endif

#endif
