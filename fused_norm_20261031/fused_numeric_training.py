import hashlib
import json
from pathlib import Path
import runpy
import shlex
import subprocess
import sys

SCRIPT=Path(__file__).resolve()

def driver():
    original=subprocess.call
    def dispatch(command,*args,**kwargs):
        command=list(command)
        for i in range(len(command)-1):
            if command[i:i+2]==['-m','jiuge_training.train']:
                actual=command[:i]+[str(SCRIPT),'--compact-worker']+command[i+2:]
                index=actual.index('--save');del actual[index:index+2]
                config=Path(actual[actual.index('--jiuge-config')+1]);assert config.is_relative_to(Path('/private/yp'))
                dependencies=['fused_numeric_training.py','fused_split_norm.py','compact_numeric_training.py','compact_tensor_pack.py','numeric_training_split.py','native_order_reductions.py','split_graph_patch.py','reduction_patch.py','oracle_rmsnorm_forward.py']
                (config.parent/'numeric-command.txt').write_text(shlex.join(actual)+'\n')
                (config.parent/'numeric-launcher.json').write_text(json.dumps({'candidate':'fused-split-v1','two_input_edges':True,'native_optimizer_checkpoint':False,'storage':'Existing lossless reference reuse; full-element reconstruction and original numerical review','dependency_sha256':{n:hashlib.sha256((SCRIPT.parent/n).read_bytes()).hexdigest() for n in dependencies}},indent=2)+'\n')
                return original(actual,*args,**kwargs)
        return original(command,*args,**kwargs)
    subprocess.call=dispatch
    runpy.run_module('jiuge_training.cli',run_name='__main__')

if __name__=='__main__':
    if '--compact-worker' in sys.argv:
        import importlib
        baseline=importlib.import_module('jiuge_training.train')
        baseline.configure_worker_caches()
        baseline.configure_worker_caches=lambda:None
        import reduction_patch
        import fused_split_norm
        import compact_numeric_training
        reduction_patch.install=fused_split_norm.install
        compact_numeric_training.worker()
    else:driver()
