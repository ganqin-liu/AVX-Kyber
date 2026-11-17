# AVX-Kyber

AVX optimized **ML-KEM-768** B1, B4 and B8 kernels.

## Build and check

Linux x86-64 with AVX2, AVX-512F/BW/DQ/VL, BMI2 and POPCNT is required.
Install GCC, Make, binutils, OpenSSL development headers, Python 3, Git,
CMake >= 3.18, Autoconf, Automake, Libtool and Perl.

```sh
make -j2 JOBS=2
make check JOBS=2
```

The first build fetches pinned wolfSSL and liboqs sources into ignored
`build/single/`, applies the included B1 patches to separate source copies,
and builds static dependencies. It needs network access. `make deps` verifies
or rebuilds these dependencies. `AVX_KYBER_SINGLE_SOURCES=/path/to/sources`
can supply verified local donor trees named `wolfssl/` and `liboqs/`.
Builds use `-march=native`; rebuild on a different CPU or after changing flags.
Serialize independent make invocations in one build directory.

Output: `build/libavxkyber.a`. Link it with
`build/single/wolf-arithmetic/src/.libs/libwolfssl.a`,
`build/single/oqs-private.o`, `-lcrypto -pthread`.
Include `src/`, `vendor/internal-kyber/avx2`,
`build/single/wolf-arithmetic`, `build/single/wolf-source`, and
`build/single/oqs/include`; define `KYBER_K=3`.
The test link command in `Makefile` is a complete example.
`make clean` removes generated build outputs and downloaded dependencies.

## Kernel entry points

| Width | Encapsulation | Decapsulation / preparation |
| --- | --- | --- |
| B1 | `mlkem_single_enc` | `mlkem_single_dec` / `mlkem_single_prepare` |
| B4 | `mlkem_enc_b4_folded` | `mlkem_dec_b4_folded` / `mlkem_skh4_init` |
| B8 | `mlkem_enc_b8` | `mlkem_dec_b8_folded` / `mlkem_skhf_init` |

B1 uses the extracted CPU-vendor policy: wolfSSL Encaps on AMD, liboqs
Encaps otherwise, and prepared wolfSSL Decaps. The wolfSSL source patches
combine mixed SHAKE streams and specialize polynomial arithmetic.
`AVX_KYBER_SINGLE_ENC=wolf|oqs|legacy` selects a backend explicitly.
`AVX_KYBER_SINGLE_DEC=wolf|legacy` exposes a policy flag through
`mlkem_single_legacy_dec()` for callers. The direct `mlkem_single_dec` API
always calls wolfSSL; `mlkem_dec_cached` always executes the legacy cached
AVX2 implementation. `mlkem_single_description()` reports the policy. Clear B1 handles with `mlkem_single_clear`
and thread scratch with `mlkem_single_thread_clear`.

B4 uses four-request folded polynomial arithmetic with x8 hashes that pair
streams. B8 Encaps uses x8 hash flow and per-request AVX2 polynomial arithmetic;
B8 Decaps uses folded AVX-512 arithmetic and compact FINV prepared operands.
`mlkem_enc_b8_folded` is also retained as an optional control, not the original
production default. `mlkem_b8.h` retains cached and serialized research APIs.

Key sizes: public key 1184, private key 2400, ciphertext 1088, shared secret
32, Encaps coins 32 bytes. Key generation takes 64 deterministic seed bytes
through the internal `crypto_kem_keypair_derand` API.

Raw batch APIs require all four/eight input and output lanes initialized.
Encaps public keys must be canonical; validate untrusted keys before calling.
Pad partial batches explicitly and publish only real lanes. Folded Decaps
shares one prepared secret key across all lanes. B4 ordinary and B8 FINV key
representations are incompatible. Preparation expects a valid encoded key;
raw batch preparation does not validate arbitrary private-key encodings.
Clear prepared B4/B8 key storage with `OPENSSL_cleanse` when finished.
Implicit rejection is implemented inside decapsulation.

Use these kernels serially within each process. Several B8 paths retain
process-global scratch and polynomial initialization is not synchronized;
separate handles alone do not make concurrent calls safe. B1 handles also
contain mutable scratch. There is no whole-library fallback for CPUs without
AVX-512. Keep all translation units on the same `ablation_config.h` settings,
since operand settings change handle layout.

## Layout and validation

- `src/`: extracted kernels, polynomial arithmetic and x8 Keccak.
- `tools/`: pinned B1 dependency builders/patches and pristine oracle builder.
- `vendor/internal-kyber/`: original internal dependency, including its notices.
- `vendor/pq-crystals-kyber/`: pristine pinned differential oracle, with symlinks
  and their targets preserved; its AVX2 inventory is separately verified.
- `tests/`: deterministic full-KEM differential checks plus original polynomial,
  hash schedule and forced matrix-refill checks.

`make check` compares B1/B4/B8 ciphertexts and secrets with the pristine oracle,
checks all 256 accepted/rejected lane masks for eight prepared identities,
and checks B1 default and forced wolfSSL, liboqs and legacy Encaps policies,
with both wolfSSL and cached AVX2 Decaps in each run. The pristine
upstream finite test also runs on first oracle build. No timing benchmark is
part of `check`. See `VALIDATION.md` for the extraction check record.

## Provenance and licenses

Per-file source hashes are in `provenance.json`. The internal
Kyber dependency is distinct from the pristine pinned upstream oracle.
B1 pins are embedded in `tools/build_single.py`: wolfSSL
`3c5eead44904df64e6a5a1f4ebdce377d35a849a` and liboqs
`5a1a854b0dc9f2141bdc771c555ee60c37950183`.
Original donor trees remain pristine; generated patched trees, inventories,
licenses and dependency manifests remain under `build/single/`.
See [LICENSES.md](LICENSES.md); this extraction assigns no new umbrella license.
