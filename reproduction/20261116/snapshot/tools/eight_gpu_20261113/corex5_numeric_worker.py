"""Full-element fingerprint audit and explicitly RAM-only loader diagnostic.

Hashes establish a zero-error fast path. A mismatch never establishes failure of
the original tolerance gate; it requires subsequent full numerical comparison.
No tensor checkpoint or gradient bodies are persisted by this worker.
"""
from pathlib import Path
import functools,hashlib,inspect,io,json,os,pickle,sys
B=Path('/private/yp/jiuge-training')

def fingerprint(value):
    import torch
    if isinstance(value,torch.Tensor):
        flat=value.detach().reshape(-1);h=hashlib.sha256()
        for start in range(0,flat.numel(),1048576):
            h.update(flat[start:start+1048576].contiguous().cpu().view(torch.uint8).numpy().tobytes())
        return {'tensor':True,'shape':list(value.shape),'dtype':str(value.dtype),'sha256':h.hexdigest()}
    if isinstance(value,dict):return {'dict':[[fingerprint(k),fingerprint(v)] for k,v in sorted(value.items(),key=lambda x:repr(x[0]))]}
    if isinstance(value,(list,tuple)):return {type(value).__name__:[fingerprint(x) for x in value]}
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError(type(value).__name__)

def children(opt):return getattr(opt,'chained_optimizers',[opt])

def full_state(bound):
    def optimizer_state(opt):
        return {'class':type(opt).__name__,'metadata':fingerprint(opt.state_dict()),
                'actual_inner':fingerprint(opt.optimizer.state_dict()),
                'masters':fingerprint([[p for p in group['params']] for group in opt.optimizer.param_groups])}
    return {'models':[fingerprint(m.state_dict()) for m in bound['model']],
            'optimizers':[optimizer_state(o) for o in children(bound['optimizer'])],
            'scheduler':fingerprint(bound['opt_param_scheduler'].state_dict())}

def rng_hash():
    import random,numpy as np,torch
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    return {'python':hashlib.sha256(pickle.dumps(random.getstate(),protocol=4)).hexdigest(),
            'numpy':hashlib.sha256(pickle.dumps(np.random.get_state(),protocol=4)).hexdigest(),
            'cpu':fingerprint(torch.get_rng_state()),'cuda':fingerprint(torch.cuda.get_rng_state()),
            'tracker':fingerprint(get_cuda_rng_tracker().get_states())}

def ram_roundtrip(bound,root,rank):
    import torch,torch.distributed as dist
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    opts=children(bound['optimizer'])
    assert len(opts)==1 and isinstance(opts[0],DistributedOptimizer)
    opt=opts[0];assert opt.data_parallel_group.size()==1
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    assert available>256*2**30,'Keep at least 256GiB host RAM available before diagnostic'
    before=full_state(bound);rng_before=rng_hash()
    native=opt.get_parameter_state_dp_zero()
    native_fp=fingerprint(native)
    payload={'model':[m.state_dict() for m in bound['model']],
             'optimizer_native_dp_zero':native,
             'scheduler':bound['opt_param_scheduler'].state_dict()}
    with io.BytesIO() as stream:
        assert torch.serialization.get_crc32_options()
        torch.save(payload,stream);serialized_bytes=stream.tell();stream.seek(0)
        restored=torch.load(stream,map_location='cpu',weights_only=False)
    del payload,native
    assert fingerprint(restored['optimizer_native_dp_zero'])==native_fp
    for model,state in zip(bound['model'],restored['model']):model.load_state_dict(state,strict=True)
    opt.load_parameter_state_from_dp_zero(restored['optimizer_native_dp_zero'])
    bound['opt_param_scheduler'].load_state_dict(restored['scheduler'])
    del restored
    torch.cuda.synchronize()
    assert full_state(bound)==before,'RAM roundtrip changed actual model/master/moments/scheduler'
    assert rng_hash()==rng_before,'RAM roundtrip changed RNG'
    repacked=opt.get_parameter_state_dp_zero();assert fingerprint(repacked)==native_fp;del repacked
    row={'passed':True,'rank':rank,'after_completed_step':9,'serialized_bytes':serialized_bytes,
         'crc32':True,'full_state_exact':True,'rng_exact':True,'native_repack_exact':True,
         'scope':'RAM-only serialization + strict model load + native DP1 optimizer load; same process; NOT durable checkpoint or fresh-process resume'}
    (root/f'ram-roundtrip-rank{rank}.json').write_text(json.dumps(row,indent=2)+'\n')
    dist.barrier()

def main():
    import megatron.training.training as mcore
    cfgpath=Path(sys.argv[sys.argv.index('--jiuge-config')+1]);cfg=json.loads(cfgpath.read_text())
    root=cfgpath.parent;rank=int(os.environ['RANK']);arm=os.environ['COREX5_AUDIT_ARM']
    assert arm in ['reference','ram-restore','fused-norm']
    if arm=='fused-norm':
        import fused_split_norm,fused_norm_dw
        candidate=fused_norm_dw if cfg['task']['name']=='cpt' else fused_split_norm
        candidate.install()
        (root/f'candidate-install-rank{rank}.json').write_text(json.dumps({'module':candidate.__name__,
          'sha256':hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
          'scope':'RMSNorm fusion only; no TP2-only LoRA extension; no other optimized stack'})+'\n')
    original=mcore.train_step
    @functools.wraps(original)
    def step(*args,**kwargs):
        bound=inspect.signature(original).bind(*args,**kwargs).arguments
        iteration=bound['iteration']
        if iteration==0:(root/f'full-initial-rank{rank}.json').write_text(json.dumps(full_state(bound))+'\n')
        result=original(*args,**kwargs)
        if iteration==8 and arm=='ram-restore':ram_roundtrip(bound,root,rank)
        if iteration==9:(root/f'full-final-rank{rank}.json').write_text(json.dumps(full_state(bound))+'\n')
        return result
    mcore.train_step=step
    original_file=B/'tools/numeric_training.py';source=original_file.read_text()
    needle='        values, schema = [], []'
    assert source.count(needle)==1
    source=source.replace(needle,'        save = False  # full-element hash only: no persisted tensor bodies\n'+needle)
    namespace={'__name__':'corex5_fingerprint_capture','__file__':str(original_file)}
    exec(compile(source,str(original_file),'exec'),namespace)
    (root/f'audit-storage-rank{rank}.json').write_text(json.dumps({'original_numeric_source_sha256':hashlib.sha256(original_file.read_bytes()).hexdigest(),
       'original_capture_positions_unchanged':True,'tensor_storage':'full-element SHA256 only; no tensor-body .pt files',
       'hash_mismatch_policy':'inconclusive under original tolerance gates; requires full elementwise metrics, never relax gates'})+'\n')
    namespace['worker']()

if __name__=='__main__':main()
