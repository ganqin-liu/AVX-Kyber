#!/usr/bin/env python3
"""Build pinned single-request backends; source and licenses stay in build/."""
import argparse, hashlib, json, os, shutil, subprocess
from pathlib import Path
from patch_wolf_mixed import prepare as prepare_mixed
from patch_wolf_arithmetic import prepare as prepare_arithmetic

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'wolfssl': ('https://github.com/wolfSSL/wolfssl.git', '3c5eead44904df64e6a5a1f4ebdce377d35a849a'),
    'liboqs': ('https://github.com/open-quantum-safe/liboqs.git', '5a1a854b0dc9f2141bdc771c555ee60c37950183')}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=int,default=2);ap.add_argument('--sources',type=Path,default=os.environ.get('AVX_KYBER_SINGLE_SOURCES'));a=ap.parse_args()
    w=ROOT/'build/single';w.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,LC_ALL='C',CC='cc',CXX='c++',CPPFLAGS='',LDFLAGS='');commands=[]
    def run(cmd,cwd=w,extra=None):
        commands.append(dict(argv=list(map(str,cmd)),cwd=str(cwd),extra=extra or {}))
        subprocess.run(list(map(str,cmd)),cwd=cwd,env=dict(env,**(extra or {})),check=True)
    sources={}
    previous=json.loads((w/'manifest.json').read_text()) if (w/'manifest.json').exists() else {}
    for name,(url,commit) in PINS.items():
        p=(a.sources/name).resolve() if a.sources else Path(previous.get('identity',{}).get('sources',{}).get(name,w/name))
        if not a.sources and name=='wolfssl' and (w/'wolf-source').is_symlink():p=(w/'wolf-source').resolve()
        if not p.exists():
            run(['git','init',p]);run(['git','remote','add','origin',url],p)
            run(['git','fetch','--depth','1','origin',commit],p);run(['git','checkout','--detach',commit],p)
        if (p/'.git').exists():
            assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip()==commit
            assert not subprocess.check_output(['git','diff','HEAD','--'],cwd=p)
        else:
            snapshot=json.loads((p/'snapshot.json').read_text());assert snapshot['commit']==commit
            assert all(sha(p/f)==h for f,h in snapshot['files'].items())
        sources[name]=p
    link=w/'wolf-source'
    if link.is_symlink(): assert link.resolve()==sources['wolfssl']
    else: link.symlink_to(sources['wolfssl'],target_is_directory=True)
    if not (sources['wolfssl']/'configure').exists():run(['./autogen.sh'],sources['wolfssl'])
    mixed_source=w/'wolf-mixed-source'
    patch=prepare_mixed(sources['wolfssl'],mixed_source)
    arithmetic_source=w/'wolf-arithmetic-source'
    arithmetic_patch=prepare_arithmetic(mixed_source,arithmetic_source)
    flags='-O3 -march=native -mtune=native'
    cpu={k.strip():v.strip() for line in Path('/proc/cpuinfo').read_text().split('\n\n')[0].splitlines() for k,_,v in [line.partition(':')] if k.strip() in ('vendor_id','cpu family','model','model name','stepping','flags')}
    identity=dict(pins=PINS,sources={k:str(v) for k,v in sources.items()},flags=flags,
                  compiler=subprocess.check_output(['cc','--version'],text=True),cpu=cpu,recipe=sha(__file__),mixed_patch=patch,arithmetic_patch=arithmetic_patch)
    identity=json.loads(json.dumps(identity))
    archives=[w/'wolf-arithmetic/src/.libs/libwolfssl.a',w/'oqs/lib/liboqs.a']
    manifest=w/'manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['identity']==identity and all(Path(p).exists() and sha(p)==h for p,h in old['outputs'].items()):
            print('single-request dependencies verified');return
    # Only disposable outputs are reset; sources and licenses stay intact.
    for directory in (w/'wolf-arithmetic',w/'oqs'):
        if directory.exists():shutil.rmtree(directory)
    wolf=w/'wolf-arithmetic';wolf.mkdir()
    run([arithmetic_source/'configure','--enable-cryptonly','--enable-static','--disable-shared','--disable-examples','--disable-crypttests','--enable-intelasm','--enable-mlkem=768,make,enc,dec,cache-a','--with-pic'],wolf,dict(CFLAGS=flags))
    run(['make',f'-j{a.jobs}'],wolf)
    run(['cmake','-S',sources['liboqs'],'-B',w/'oqs','-DOQS_MINIMAL_BUILD=KEM_ml_kem_768','-DOQS_BUILD_ONLY_LIB=ON','-DBUILD_SHARED_LIBS=OFF','-DCMAKE_POSITION_INDEPENDENT_CODE=ON','-DOQS_USE_OPENSSL=OFF','-DOQS_DIST_BUILD=ON','-DOQS_USE_SHA3_AVX512VL=ON','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_C_FLAGS='+flags])
    run(['cmake','--build',w/'oqs',f'-j{a.jobs}'])
    # Resolve OQS-internal Keccak globals before making them local: IPsec-MB
    # contains independently built symbols with the same upstream names.
    merged=w/'oqs-private.o'
    run(['ld','-r','--whole-archive',archives[1],'--no-whole-archive','-o',merged])
    symbols=subprocess.check_output(['nm','-g','--defined-only',merged],text=True)
    local=w/'oqs-local-symbols.txt';local.write_text(''.join(line.split()[-1]+'\n' for line in symbols.splitlines() if len(line.split())>=3 and not line.split()[-1].startswith('OQS_')))
    run(['objcopy','--localize-symbols='+str(local),merged])
    outputs=archives+[merged,local,wolf/'wolfssl/options.h',w/'oqs/include/oqs/oqsconfig.h']
    licenses={k:{str(f.relative_to(p)):sha(f) for f in p.iterdir() if f.is_file() and f.name.startswith(('LICENSE','COPYING'))} for k,p in sources.items()}
    manifest.write_text(json.dumps(dict(identity=identity,commands=commands,outputs={str(p):sha(p) for p in outputs},licenses=licenses),indent=2)+'\n')
if __name__=='__main__': main()
