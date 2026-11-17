#!/usr/bin/env python3
"""Compose the fixed-k3 arithmetic patch on a verified mixed-SHAKE copy."""
import hashlib
import json
import re
import shutil
from pathlib import Path
from patch_wolf_mixed import replace_once

HERE = Path(__file__).resolve().parent
ASM = 'wolfcrypt/src/wc_mlkem_asm.S'
ASM_SHA = '592dc04ca7451538742fd327c67e9d5acae5fbb8ea16ebbf811bf8c3b8c6f8d0'
POLY = 'wolfcrypt/src/wc_mlkem_poly.c'
GUARD = '#if defined(__linux__) && defined(__x86_64__)\n'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(donor, destination):
    donor, destination = Path(donor), Path(destination)
    mixed = json.loads((donor/'avxkyber-mixed.json').read_text())
    assert all(sha(donor/p) == h for p,h in mixed['files'].items())
    assert sha(donor/ASM) == ASM_SHA
    recipe = {p.name:sha(p) for p in (Path(__file__), HERE/'wolf_arithmetic.c.inc')}
    stamp = destination/'avxkyber-arithmetic.json'
    if stamp.exists():
        old = json.loads(stamp.read_text())
        if old['record']['recipe'] == recipe and old['record']['mixed'] == mixed['record'] and all(
                (destination/p).is_file() and sha(destination/p) == h
                for p,h in old['files'].items()):
            return old['record']
    if destination.exists():
        assert not destination.is_symlink()
        shutil.rmtree(destination)
    shutil.copytree(donor, destination, symlinks=True)
    poly = (destination/POLY).read_text()
    table = re.search(r'const sword16 zetas\[.*?\] = \{(.*?)\};', poly, re.S)
    assert table
    values = list(map(int, re.findall(r'-?\d+', table.group(1))))
    assert len(values) == 128
    z = [values[64+j//2] * (-1 if j % 2 else 1) for j in range(128)]
    helper = (HERE/'wolf_arithmetic.c.inc').read_text().replace('VN_ZETAS',
        ','.join(map(str, [t for v in z for t in (v,v)])))
    at = poly.rfind('#endif')
    (destination/POLY).write_text(poly[:at] + helper + '\n' + poly[at:])
    asm = (destination/ASM).read_text()
    start = asm.index('\nmlkem_encapsulate_avx512:')
    end = asm.index('\n.size\tmlkem_encapsulate_avx512', start)
    part = asm[start:end]
    registers = ['rdi','rsi','rdx','rcx','r8','r9','rax','r10','r11']
    call = GUARD + '''        cmpl $3, %r11d
        jne L_avxkyber_arith_original
        subq $80, %rsp
'''
    call += ''.join(f'        movq %{r}, {i*8}(%rsp)\n' for i,r in enumerate(registers))
    call += '        call avxkyber_mlkem_arithmetic\n'
    call += ''.join(f'        movq {i*8}(%rsp), %{r}\n' for i,r in enumerate(registers))
    call += '''        addq $80, %rsp
        movq $0xd01, %r13
        vmovd %r13d, %xmm14
        vpbroadcastw %xmm14, %zmm14
        movq $0x4ebf, %r13
        vmovd %r13d, %xmm15
        vpbroadcastw %xmm15, %zmm15
        movl $0xaaaa, %r13d
        kmovw %r13d, %k1
        movl $0x5555, %r13d
        kmovw %r13d, %k2
        movl $0xaaaaaaaa, %r13d
        kmovd %r13d, %k3
        movl $0x55555555, %r13d
        kmovd %r13d, %k4
        vmovdqu64 L_mlkem_avx512_perm20(%rip), %zmm17
        vmovdqu64 L_mlkem_avx512_perm31(%rip), %zmm18
        leaq 512(%rsi), %r15
        movq $3, %r12
        jmp L_avxkyber_arith_inv01
L_avxkyber_arith_original:
#endif
'''
    part = replace_once(part, 'L_encap_avx512_sp_done:\n', 'L_encap_avx512_sp_done:\n'+call)
    inv01 = '        leaq\tL_mlkem_avx512_zetas_inv(%rip), %rbp\n        vmovdqu64\t(%rsi), %ymm0\n        vinserti64x4\t$0x01, (%r15), %zmm0, %zmm0'
    part = replace_once(part, inv01, 'L_avxkyber_arith_inv01:\n'+inv01)
    inv23 = '        leaq\tL_mlkem_avx512_zetas_inv(%rip), %rbp\n        vmovdqu64\t(%rdx), %ymm0\n        vinserti64x4\t$0x01, (%rsi), %zmm0, %zmm0'
    part = replace_once(part, inv23, 'L_avxkyber_arith_inv23:\n'+inv23)
    tail = '        subq\t$2, %r12\n        jmp\tL_encap_avx512_calc'
    part = replace_once(part, tail, '        subq\t$2, %r12\n'+GUARD+'''
        cmpl $3, %r11d
        je L_avxkyber_arith_inv23
#endif
        jmp L_encap_avx512_calc''')
    (destination/ASM).write_text(asm[:start]+part+asm[end:])
    files = {str(p.relative_to(destination)):sha(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    record = dict(recipe=recipe, mixed=mixed['record'], upstream_asm=ASM_SHA,
        patched={p:files[p] for p in (ASM,POLY)},
        source_tree_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())
    stamp.write_text(json.dumps(dict(record=record,files=files),indent=2)+'\n')
    return record
