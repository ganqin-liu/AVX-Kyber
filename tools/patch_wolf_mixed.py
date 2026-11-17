#!/usr/bin/env python3
"""Materialize the pinned mixed-SHAKE patch without changing its donor tree."""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = {
    'wolfcrypt/src/wc_mlkem.c': '1f2ea3c64c9c6f3a1f5611f60d89c3c9478e4f6691df54b72fb6f62bb9dd2155',
    'wolfcrypt/src/wc_mlkem_poly.c': '805eaa937c8f0fc895e23d06156628ae5c1a79091882383eb7fd9e23ef1afe2e',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def replace_once(text, old, new):
    assert text.count(old) == 1, (old[:100], text.count(old))
    return text.replace(old, new, 1)

def prepare(donor, destination):
    donor, destination = Path(donor), Path(destination)
    assert all(sha(donor / p) == h for p, h in UPSTREAM.items())
    recipe = {p.name: sha(p) for p in (Path(__file__), HERE / 'wolf_mixed.c.inc')}
    stamp = destination / 'avxkyber-mixed.json'
    if stamp.exists():
        old = json.loads(stamp.read_text())
        if old['record']['recipe'] == recipe and all(
                (destination / p).is_file() and sha(destination / p) == h
                for p, h in old['files'].items()):
            return old['record']
    # This is our disposable copy; never replace the pristine wolf-source link.
    if destination.exists():
        assert not destination.is_symlink()
        shutil.rmtree(destination)
    shutil.copytree(donor, destination, symlinks=True,
                    ignore=shutil.ignore_patterns('.git', 'autom4te.cache'))
    cpath = destination / 'wolfcrypt/src/wc_mlkem.c'
    c = cpath.read_text()
    signature = 'static int mlkemkey_encapsulate(MlKemKey* key, const byte* m, byte* r, byte* c)'
    c = replace_once(c, signature,
        'extern int avxkyber_mlkem_mixed_noise_matrix(MLKEM_PRF_T*, sword16*, '
        'sword16*, sword16*, sword16*, byte*, byte*);\n' + signature)
    c = replace_once(c, signature + '\n{\n    int ret = 0;',
                    signature + '\n{\n    int ret = 0;\n    int mixed = 0;')
    c = replace_once(c, '        ret = mlkem_get_noise(&key->prf, (int)k, y, e1, e2, r);',
        '''        /* Fresh ML-KEM-768 peers: seven SHAKE256 noise streams share
         * one permutation with the ninth SHAKE128 matrix stream. Cached A
         * (including prepared FO reencryption) keeps the upstream path. */
        mixed = k == 3 && !(key->flags & MLKEM_FLAG_A_SET);
        if (mixed)
            ret = avxkyber_mlkem_mixed_noise_matrix(&key->prf, a, y, e1, e2,
                                                  key->pubSeed, r);
        else
            ret = mlkem_get_noise(&key->prf, (int)k, y, e1, e2, r);''')
    c = replace_once(c,
        '        ret = mlkem_gen_matrix(&key->prf, a, (int)k, key->pubSeed, 1);',
        '        if (!mixed)\n            ret = mlkem_gen_matrix(&key->prf, a, (int)k, key->pubSeed, 1);')
    cpath.write_text(c)
    ppath = destination / 'wolfcrypt/src/wc_mlkem_poly.c'
    p = ppath.read_text()
    start = p.index('static int mlkem_gen_matrix_k3_avx512(')
    end = p.index('\n#endif /* WOLFSSL_MLKEM_HAVE_INTEL_AVX512 */', start)
    block = p[start:end]
    block = replace_once(block,
        'static int mlkem_gen_matrix_k3_avx512(sword16* a, byte* seed, int transposed)',
        'static int avxkyber_matrix_k3(sword16* a, byte* seed, int transposed, int first8)')
    block = replace_once(block, '    /* Ninth polynomial (row 2, column 2)',
                          '    if (!first8) {\n    /* Ninth polynomial (row 2, column 2)')
    block = replace_once(block, '    WC_FREE_VAR_EX(rand, NULL, DYNAMIC_TYPE_TMP_BUFFER);',
                          '    }\n    WC_FREE_VAR_EX(rand, NULL, DYNAMIC_TYPE_TMP_BUFFER);')
    block += '\nstatic int mlkem_gen_matrix_k3_avx512(sword16* a, byte* seed, int transposed)\n{ return avxkyber_matrix_k3(a, seed, transposed, 0); }\n'
    p = p[:start] + block + p[end:]
    at = p.rfind('#endif')
    p = p[:at] + (HERE / 'wolf_mixed.c.inc').read_text() + '\n' + p[at:]
    ppath.write_text(p)
    files = {str(p.relative_to(destination)): sha(p)
             for p in sorted(destination.rglob('*')) if p.is_file()}
    record = dict(recipe=recipe, upstream=UPSTREAM,
                  patched={p: files[p] for p in UPSTREAM},
                  source_tree_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
    stamp.write_text(json.dumps(dict(record=record, files=files), indent=2) + '\n')
    return record
