from pathlib import Path
import argparse,ast,hashlib,json,shutil,subprocess,sys
H=Path(__file__).resolve().parent;B=Path('/private/yp/jiuge-training');E=B/'checks/20261113-eight-gpu-corex5'
for rel,digest in json.loads((H/'FILES_SHA256.json').read_text()).items():
 f=H/rel;assert hashlib.sha256(f.read_bytes()).hexdigest()==digest,rel
 if f.suffix=='.py':ast.parse(f.read_text(encoding='utf-8-sig'))
p=argparse.ArgumentParser();p.add_argument('--install',action='store_true');a=p.parse_args()
if not a.install:print('Use --install in a fresh Linux/CoreX5 project');sys.exit()
assert sys.platform.startswith('linux') and not (B/'tools').exists() and not E.exists(),'Fresh workspace required'
for path,sha in [(B/'source','9fabd71df4897f08267df5431aa480ab9ef70a75'),(B/'Megatron-LM','5be9626709af2722333bf54797c954c09edeada3')]:
 assert subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()==sha
 subprocess.run(['git','-C',str(path),'diff','--quiet',sha],check=True)
shutil.copytree(H/'golden',B/'reproduction-golden-20261116');shutil.copytree(H/'snapshot/tools',B/'tools');shutil.copytree(H/'configs',E/'prepared-v4')
(E/'numeric-v6').mkdir();shutil.copyfile(H/'numerical-gate.json',E/'numeric-v6/protocol.original.json')
(E/'overlay/megatron').mkdir(parents=True);(E/'overlay/megatron/__init__.py').write_text('__path__ = ["/private/yp/jiuge-training/Megatron-LM/megatron"]\n')
for n in ['deps','lib','tmp','cache']:(E/n).mkdir()
lib=next((p for p in [Path('/usr/local/iluvatar/lib64/libcuda.so.1'),Path('/usr/local/iluvatar/lib64/libcuda.so'),Path('/usr/local/corex/lib64/libcuda.so.1'),Path('/usr/local/corex/lib64/libcuda.so'),Path('/usr/lib/x86_64-linux-gnu/libcuda.so.1'),Path('/usr/lib/x86_64-linux-gnu/libcuda.so')] if p.exists()),None);assert lib,'Provide vendor libcuda.so.1'
(E/'lib/libcuda.so').symlink_to(lib);shutil.copyfile(H/'environment.json',E/'environment.json');shutil.copyfile(H/'run_gpu.py',B/'tools/run_public_e2e_20261116.py')
print('Installed frozen tools, configs, overlay and fresh runner; prepare dependencies and inputs next.')
