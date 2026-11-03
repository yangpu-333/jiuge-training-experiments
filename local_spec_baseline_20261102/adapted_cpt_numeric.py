"""Full-model local-operator baseline with nine-grid semantic adapters."""
import hashlib,json,runpy,shlex,subprocess,sys
from pathlib import Path
SCRIPT=Path(__file__).resolve();BASE=Path('/private/yp/jiuge-training')
def driver():
 original=subprocess.call
 def dispatch(command,*args,**kwargs):
  command=list(command)
  for i in range(len(command)-1):
   if command[i:i+2]==['-m','jiuge_training.train']:
    actual=command[:i]+[str(SCRIPT),'--compact-worker']+command[i+2:]
    idx=actual.index('--save');del actual[idx:idx+2]
    cfg=Path(actual[actual.index('--jiuge-config')+1]);assert cfg.is_relative_to(BASE)
    files=['adapted_cpt_numeric.py','local_spec_ablation_patch.py','local_spec_patch.py','numeric_local_audit.py','compact_numeric_training.py','compact_tensor_pack.py','compact_xor_v2.py']
    (cfg.parent/'numeric-command.txt').write_text(shlex.join(actual)+'\n')
    (cfg.parent/'numeric-launcher.json').write_text(json.dumps({'candidate':'MCore local DotProductAttention with nine-grid reference RMSNorm and explicit semantic adapters','actual_backend':'native local operators; CLI reference','storage':'lossless XOR/zlib to immutable reference, per-tensor content hash, free-space guard','no_optimizer_checkpoint_duplicated':True,'dependency_sha256':{n:hashlib.sha256((SCRIPT.parent/n).read_bytes()).hexdigest() for n in files}},indent=2)+'\n')
    return original(actual,*args,**kwargs)
  return original(command,*args,**kwargs)
 subprocess.call=dispatch
 try:runpy.run_module('jiuge_training.cli',run_name='__main__')
 finally:subprocess.call=original
def worker():
 import importlib
 sys.modules.setdefault('tensorflow',None)
 baseline=importlib.import_module('jiuge_training.train');baseline.configure_worker_caches();baseline.configure_worker_caches=lambda:None
 import local_spec_ablation_patch,reduction_patch
 local_spec_ablation_patch.install();reduction_patch.install=lambda:None
 import compact_tensor_pack,compact_xor_v2
 compact_tensor_pack.make_pack=compact_xor_v2.make_pack
 # Same original audit except suppress redundant initial master tensor file;
 # the initial masters and every subsequent full capture remain hashed.
 sys.modules['numeric_training_split']=importlib.import_module('numeric_local_audit')
 import compact_numeric_training
 compact_numeric_training.worker()
if __name__=='__main__':
 if '--compact-worker' in sys.argv:worker()
 else:driver()
