# Validation — 2026-11-17

PASS on Linux x86-64, Intel Core i3-1005G1 with AVX-512.
Compiler: `cc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0`.

Command: `make -j2 check JOBS=2` (exit 0).

- B1 dependencies rebuilt from pinned sources with AVX-Kyber patch symbols.
- Pristine upstream ML-KEM-768 finite test passed on oracle rebuild.
- B1/B4/B8 Encaps ciphertexts and secrets match the pristine oracle for 64
  heterogeneous key/coin inputs, including both B8 Encaps implementations.
- Eight prepared identities, each tested across all 256 ciphertext-corruption
  masks: wolfSSL and cached AVX2 B1 Decaps and B4/B8 folded Decaps match
  pristine accepted and implicit-rejection secrets.
- Full KEM checks passed for default, wolfSSL, liboqs and legacy Encaps policies.
- Original polynomial, hash schedule/guard and forced matrix-refill tests passed.
- Vendor source identities and contained symlink targets were preserved.
- All 233 inventory entries match `provenance.json`. No benchmarks were run.

Environment variables use `AVX_KYBER_SINGLE_*`; patch symbols use `avxkyber_*`.
The oracle builder supports rebuilding its own generated directory when the
recipe changes. Final console output is in `validation/check.txt`.

Current build SHA-256 identities:

- `build/libavxkyber.a`: `0d35a7888d6d777b1d0230da779d9de600b706c6c1b8e8f0d5b5d1e23c1fbcec`
- `build/test_kernels`: `7fda1dcb72af43e1152c278669b8dd147743301000b519ef56459fca5a57d8b1`
- `build/pqcrystals/pqcrystals.o`: `723e6b21398aefd923e1edc115827e6b38c5ec0a3c5e5683c8843fd7fd866a1e`
- `build/single/wolf-arithmetic/src/.libs/libwolfssl.a`: `98b9196c9fe22f75e46efd3f01b16e2041c78905036d813a5060a8978236b484`
- `build/single/oqs-private.o`: `61b9e21592acf654e1b08c0d354e93037e8355d4a5ec989991efde4b787dea2f`
