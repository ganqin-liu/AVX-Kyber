#!/usr/bin/env python3
"""Build pristine pinned pq-crystals AVX2 sources; namespace objects, not source."""
import hashlib,json,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
COMMIT='3edd5af5991927164edd4aacebfcbee00b8064e7'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 repo=ROOT/'vendor/pq-crystals-kyber';src=repo/'avx2';out=ROOT/'build/pqcrystals'
 # Rebuild only the dedicated generated oracle directory when its recipe changes.
 if out.exists():
  assert not out.is_symlink()
  shutil.rmtree(out)
 out.mkdir(exist_ok=False)
 sources=['kem.c','indcpa.c','polyvec.c','poly.c','fq.S','shuffle.S','ntt.S','invntt.S','basemul.S','consts.c','rejsample.c','cbd.c','verify.c','fips202.c','fips202x4.c','symmetric-shake.c','randombytes.c','keccak4x/KeccakP-1600-times4-SIMD256.c']
 # Exact flags from the pinned upstream Makefile, with explicit K=3.
 flags=['-Wall','-Wextra','-Wpedantic','-Wmissing-prototypes','-Wredundant-decls','-Wshadow','-Wpointer-arith','-mavx2','-mbmi2','-mpopcnt','-march=native','-mtune=native','-O3','-fomit-frame-pointer','-z','noexecstack','-DKYBER_K=3']
 if (repo/'.git').exists():
  assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==COMMIT
  assert not subprocess.check_output(['git','diff','--name-only',COMMIT,'--','avx2'],cwd=repo,text=True).strip()
 hashes={str(p.relative_to(repo)):sha(p) for p in src.rglob('*') if p.is_file()}
 expected=json.loads((ROOT/'vendor/pq-crystals-provenance.json').read_text())
 assert expected['commit']==COMMIT and hashes==expected['sources_sha256']
 objects=[];commands=[]
 with (out/'build.txt').open('x') as log:
  for name in sources:
   target=out/(name.replace('/','_')+'.o');cmd=['cc',*flags,'-c',str(src/name),'-o',str(target)];commands.append(cmd)
   subprocess.run(cmd,cwd=src,stdout=log,stderr=subprocess.STDOUT,check=True);objects.append(target)
  combined=out/'combined-unrenamed.o';cmd=['ld','-r','-o',str(combined),*map(str,objects)];commands.append(cmd);subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
  syms=subprocess.check_output(['nm','-g','--defined-only',str(combined)],text=True)
  mapping=[]
  for line in syms.splitlines():
   parts=line.split()
   if len(parts)==3:mapping.append((parts[2],'ls_pqcrystals_'+parts[2]))
  m=out/'symbols.map';m.write_text(''.join(f'{a} {b}\n' for a,b in mapping))
  linked=out/'pqcrystals.o';cmd=['objcopy','--redefine-syms',str(m),str(combined),str(linked)];commands.append(cmd);subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
  # Original upstream functional test uses the unrenamed combined object.
  test=out/'test_kyber768';cmd=['cc',*flags,'-I',str(src),str(src/'test/test_kyber.c'),str(combined),'-o',str(test)];commands.append(cmd);subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 with (out/'upstream-test.txt').open('x') as log:subprocess.run([str(test)],stdout=log,stderr=subprocess.STDOUT,check=True)
 assert hashes=={str(p.relative_to(repo)):sha(p) for p in src.rglob('*') if p.is_file()}
 manifest=dict(status='PASS',upstream='https://github.com/pq-crystals/kyber',commit=COMMIT,sources_sha256=hashes,flags=flags,commands=commands,compiler=subprocess.check_output(['cc','--version'],text=True),object_sha256=sha(linked),symbols_renamed=len(mapping),scope='No source changes or cached-key wrapper; upstream default flags include -march=native, so AVX2 names the implementation, not an enforced ISA ceiling.')
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(status='PASS',object_sha256=manifest['object_sha256'],symbols_renamed=len(mapping))))
if __name__=='__main__':main()
