.DEFAULT_GOAL := all
CC = cc
AR = ar
JOBS ?= 2
KYB = vendor/internal-kyber
CPPFLAGS += -DKYBER_K=3 -Isrc -I$(KYB)/avx2 -Ibuild/single/wolf-arithmetic -Ibuild/single/wolf-source -Ibuild/single/oqs/include
CFLAGS = -O3 -march=native -mtune=native -fomit-frame-pointer -Wall -Wextra -pthread
DEPFLAGS = -MMD -MP
LDFLAGS += -pthread -Wl,-z,noexecstack
SINGLE_LIBS = build/single/wolf-arithmetic/src/.libs/libwolfssl.a build/single/oqs-private.o
KERNELS = mlkem_single mlkem_b8 mlkem_b8f mlkem_b4 mlkem_b4dec poly4 poly8 fips202x8 mlkem_hash_spwords mlkem_enc_flow
AVX_C = kem indcpa polyvec poly consts rejsample cbd verify fips202 fips202x4 symmetric-shake randombytes
AVX_S = fq shuffle ntt invntt basemul
OBJECTS = $(addprefix build/,$(addsuffix .o,$(KERNELS))) $(addprefix build/avx2_,$(addsuffix .o,$(AVX_C) $(AVX_S))) build/keccak4x.o build/ref_ntt.o build/ref_reduce.o
TESTS = test_kernels test_poly4 test_poly8 encflowtest hashschedtest
.PHONY: all check deps clean
all: build/libavxkyber.a
# Source builder verifies pins and applies the recorded B1 patches.
build/single.stamp: $(wildcard tools/*.py tools/*.inc) | build
	python3 tools/build_single.py --jobs $(JOBS)
	touch $@
deps:
	python3 tools/build_single.py --jobs $(JOBS)
build/mlkem_single.o build/mlkem_b8.o: build/single.stamp
build/libavxkyber.a: $(OBJECTS)
	$(AR) rcs $@ $^
build/%.o: src/%.c | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c $< -o $@
build/avx2_%.o: $(KYB)/avx2/%.c | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c $< -o $@
build/avx2_%.o: $(KYB)/avx2/%.S | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c $< -o $@
build/keccak4x.o: $(KYB)/avx2/keccak4x/KeccakP-1600-times4-SIMD256.c | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c $< -o $@
build/ref_%.o: $(KYB)/ref/%.c | build
	$(CC) $(CFLAGS) $(DEPFLAGS) -DKYBER_K=3 -I$(KYB)/ref -c $< -o $@
build/pqcrystals/pqcrystals.o: tools/build_upstream.py vendor/pq-crystals-provenance.json | build
	python3 tools/build_upstream.py
build/test_kernels: tests/test_kernels.c build/libavxkyber.a build/pqcrystals/pqcrystals.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $< build/libavxkyber.a build/pqcrystals/pqcrystals.o $(SINGLE_LIBS) $(LDFLAGS) -lcrypto -o $@
build/test_poly4: tests/test_poly4.c build/poly4.o build/ref_ntt.o build/ref_reduce.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ $(LDFLAGS) -o $@
build/test_poly8: tests/test_poly8.c build/poly8.o build/ref_ntt.o build/ref_reduce.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ $(LDFLAGS) -o $@
build/encflowtest: tests/encflowtest.c build/libavxkyber.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< build/libavxkyber.a $(LDFLAGS) -lcrypto -o $@
build/hashschedtest: tests/hashschedtest.c build/fips202x8.o build/mlkem_hash_spwords.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ $(LDFLAGS) -lcrypto -o $@
check: $(addprefix build/,$(TESTS))
	./build/test_kernels
	AVX_KYBER_SINGLE_ENC=wolf ./build/test_kernels
	AVX_KYBER_SINGLE_ENC=oqs ./build/test_kernels
	AVX_KYBER_SINGLE_ENC=legacy AVX_KYBER_SINGLE_DEC=legacy ./build/test_kernels
	./build/test_poly4
	./build/test_poly8
	./build/encflowtest
	./build/hashschedtest
build:
	mkdir -p $@
clean:
	rm -rf build
-include $(wildcard build/*.d)
