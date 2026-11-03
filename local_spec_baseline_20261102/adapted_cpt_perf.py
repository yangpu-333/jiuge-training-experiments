"""Measurement calibration: identical post-warmup full GC in both arms."""
import functools
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import performance_training as perf

def worker():
    baseline=importlib.import_module('jiuge_training.train');baseline.configure_worker_caches();baseline.configure_worker_caches=lambda:None
    import torch
    import torch.distributed as dist
    import lora_linear_cpp,lora_tp_linear
    extension=lora_linear_cpp.extension();tp_extension=lora_tp_linear.extension()
    arm=os.environ['PERF_LOCAL_ARM'];assert arm in ['native','optimized']
    task=json.loads(Path(sys.argv[sys.argv.index('--jiuge-config')+1]).read_text())['task']['name']
    assert task=='cpt'
    import reduction_patch,fused_split_norm,fused_norm_dw
    if arm=='native':
        import local_spec_ablation_patch
        os.environ['LOCAL_SPEC_ABLATION']='attention-only'
        local_spec_ablation_patch.install();norm_module=local_spec_ablation_patch
    else:
        lora_linear_cpp.install()
        norm_module=fused_norm_dw;reduction_patch.install=norm_module.install
    import megatron.training.training as mcore
    original=mcore.train_step
    root=Path(os.environ['PHASE_OUTPUT']);rank=os.environ['RANK']
    active={};events=[];state={'iteration':None,'calls':0};setup={}
    def cb(phase,info):
        g=info['generation']
        if phase=='start':active[g]=(time.perf_counter(),state['iteration'])
        elif g in active:
            t,i=active.pop(g);events.append({'generation':g,'iteration':i,'seconds':time.perf_counter()-t,'collected':info['collected'],'uncollectable':info['uncollectable']})
    gc.callbacks.append(cb)
    @functools.wraps(original)
    def step(*args,**kwargs):
        iteration=state['calls'];state['calls']+=1;state['iteration']=iteration
        try:
            result=original(*args,**kwargs)
            if iteration==4:
                torch.cuda.synchronize();dist.barrier()
                before=time.perf_counter();collected=gc.collect();elapsed=time.perf_counter()-before
                dist.barrier();torch.cuda.synchronize()
                setup.update(full_gc_seconds=elapsed,collected=collected,auto_gc_enabled_after=gc.isenabled(),threshold=gc.get_threshold())
            return result
        finally:state['iteration']=None
    mcore.train_step=step
    (root/f'normalized-install-rank{rank}.json').write_text(json.dumps({'arm':arm,'norm_or_patch_sha256':hashlib.sha256(Path(norm_module.__file__).read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),'tp_binary_sha256':hashlib.sha256(Path(tp_extension.__file__).read_bytes()).hexdigest(),'protocol':'same full GC at end of last warmup step, automatic GC remains enabled','measurement_not_runtime_optimization':True}))
    try:
        import phase_training
        phase_training.worker()
    finally:
        gc.callbacks.remove(cb)
        (root/f'normalized-gc-rank{rank}.json').write_text(json.dumps({'setup':setup,'events':events,'auto_gc_policy_disabled':False,'measurement_scope':'All steady samples retained; setup GC is separately recorded warmup cost'},indent=2)+'\n')

if __name__=='__main__':
    if '--performance-worker' in sys.argv:worker()
    else:perf.SCRIPT=Path(__file__).resolve();perf.driver()
