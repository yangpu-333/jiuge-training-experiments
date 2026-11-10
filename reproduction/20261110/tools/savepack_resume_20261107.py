"""Native checkpoint step9 -> step10, with streaming full-state fingerprints."""
import functools
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys

BASE=Path('/private/yp/jiuge-training')

def fingerprint(value):
    import numpy as np
    import torch
    if isinstance(value,torch.Tensor):
        x=value.detach().reshape(-1);h=hashlib.sha256()
        for start in range(0,x.numel(),1048576):
            h.update(x[start:start+1048576].contiguous().cpu().view(torch.uint8).numpy().tobytes())
        return {'tensor':True,'shape':list(value.shape),'dtype':str(value.dtype),'sha256':h.hexdigest()}
    if isinstance(value,np.ndarray):return {'ndarray':True,'dtype':str(value.dtype),'shape':list(value.shape),'sha256':hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value,dict):return {'dict':[[fingerprint(k),fingerprint(v)] for k,v in sorted(value.items(),key=lambda x:repr(x[0]))]}
    if isinstance(value,(list,tuple)):return {type(value).__name__:[fingerprint(x) for x in value]}
    if isinstance(value,np.generic):return value.item()
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError(f'Unsupported state object {type(value)}')

def worker():
    baseline=importlib.import_module('jiuge_training.train');baseline.configure_worker_caches();baseline.configure_worker_caches=lambda:None
    import torch
    import lora_linear_cpp,fused_split_norm,fused_norm_dw
    cfgpath=Path(sys.argv[sys.argv.index('--jiuge-config')+1]);cfg=json.loads(cfgpath.read_text())
    task=cfg['task']['name'];norm=fused_norm_dw if task=='cpt' else fused_split_norm
    lora_linear_cpp.install();norm.install()
    import swiglu_fast_patch_20261106
    swiglu_fast_patch_20261106.install()
    import megatron.training.training as mcore
    import audit_training
    assert cfg['task']['train_iters']==51 and cfg['task']['recompute'] is False
    assert cfg['task'].get('load') and cfg['task']['save_interval']==50
    assert cfg['parallel']['tp']==cfg['parallel']['pp']==2 and cfg['task']['seq_length']==512
    original_step=mcore.train_step;original_save=mcore.save_checkpoint;progress={}
    from jiuge_ops import ReferenceRMSNorm
    import jiuge_ops._triton as norm_implementation
    def optimizer_fingerprint(opt):
        if hasattr(opt,'chained_optimizers'):
            return {'children':[optimizer_fingerprint(x) for x in opt.chained_optimizers]}
        # DistributedOptimizer.state_dict deliberately omits parameter moments.
        # Hash actual inner optimizer tensors AND its FP32 parameter groups.
        inner=opt.optimizer
        return {'class':type(opt).__name__,'wrapper_metadata':fingerprint(opt.state_dict()),
                'inner_state':fingerprint(inner.state_dict()),
                'master_parameters':fingerprint([[p for p in g['params']] for g in inner.param_groups])}
    def snapshot(bound):
        return {'model':[fingerprint(x.state_dict()) for x in bound['model']],
                'optimizer':optimizer_fingerprint(bound['optimizer']),
                'scheduler':fingerprint(bound['opt_param_scheduler'].state_dict())}
    @functools.wraps(original_step)
    def step(*args,**kwargs):
        bound=inspect.signature(original_step).bind(*args,**kwargs).arguments
        if not progress:
            norms=[m for model in bound['model'] for m in model.modules() if isinstance(m,ReferenceRMSNorm)]
            assert norms and all(m.backend=='triton' for m in norms)
            assert norm_implementation.triton_rms_norm is norm.rms_norm
            extension=lora_linear_cpp.extension()
            receipt={'task':task,'norm_dw_active':task=='cpt','norm_module_count':len(norms),'norm_binding_verified':True,
                     'norm_sha256':hashlib.sha256(Path(norm.__file__).read_bytes()).hexdigest(),
                     'lora_binary_sha256':hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest()}
            (cfgpath.parent/f'candidate-install-rank{os.environ["RANK"]}.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if bound['iteration']==50 and cfg['task'].get('load'):
            torch.cuda.synchronize()
            (cfgpath.parent/f'loaded-step50-state-rank{os.environ["RANK"]}.json').write_text(json.dumps(snapshot(bound),indent=2)+'\n')
        result=original_step(*args,**kwargs)
        progress['completed_iteration']=bound['iteration']+1
        if bound['iteration'] in [49,50]:
            torch.cuda.synchronize()
            state=snapshot(bound)
            name='step50-state' if bound['iteration']==49 else 'final-state'
            (cfgpath.parent/f'{name}-rank{os.environ["RANK"]}.json').write_text(json.dumps(state,indent=2)+'\n')
        return result
    @functools.wraps(original_save)
    def save(*args,**kwargs):
        bound=inspect.signature(original_save).bind(*args,**kwargs).arguments
        iteration=bound['iteration']
        # Existing step9 checkpoint is read only; suppress all new saves.
        print('RESTORE_AUDIT_SAVE_SKIPPED',iteration,flush=True)
    mcore.train_step=step;mcore.save_checkpoint=save
    try:
        audit_training.worker()
        settings=mcore.get_args()
        (cfgpath.parent/f'final-counters-rank{os.environ["RANK"]}.json').write_text(json.dumps({
            'iteration':progress['completed_iteration'],'consumed_train_samples':settings.consumed_train_samples,
            'consumed_valid_samples':settings.consumed_valid_samples},indent=2)+'\n')
    finally:mcore.train_step=original_step;mcore.save_checkpoint=original_save

if __name__=='__main__':
    if '--audit-worker' in sys.argv:worker()
    else:
        import audit_training
        audit_training.SCRIPT=Path(__file__).resolve();audit_training.driver()
