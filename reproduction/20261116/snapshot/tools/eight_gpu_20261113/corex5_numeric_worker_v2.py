"""RAM roundtrip for both native LoRA FP16-wrapper and CPT DP1 optimizers."""
from pathlib import Path
import io,json
import corex5_numeric_worker as base

def roundtrip(bound,root,rank):
    import torch,torch.distributed as dist
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params
    opts=base.children(bound['optimizer'])
    assert opts and all(isinstance(o,(DistributedOptimizer,Float16OptimizerWithFloat16Params)) for o in opts)
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    assert available>256*2**30
    before=base.full_state(bound);rng_before=base.rng_hash()
    native=[]
    for opt in opts:
        if isinstance(opt,DistributedOptimizer):
            assert opt.data_parallel_group.size()==1
            native.append({'kind':'distributed-dp-zero','state':opt.get_parameter_state_dp_zero()})
        else:native.append({'kind':'float16-wrapper-state-dict','state':opt.state_dict()})
    native_fp=base.fingerprint(native)
    payload={'model':[m.state_dict() for m in bound['model']], 'optimizer_native':native,
             'scheduler':bound['opt_param_scheduler'].state_dict()}
    with io.BytesIO() as stream:
        assert torch.serialization.get_crc32_options()
        torch.save(payload,stream);size=stream.tell();stream.seek(0)
        restored=torch.load(stream,map_location='cpu',weights_only=False)
    del payload,native
    assert base.fingerprint(restored['optimizer_native'])==native_fp
    for model,state in zip(bound['model'],restored['model']):model.load_state_dict(state,strict=True)
    for opt,item in zip(opts,restored['optimizer_native']):
        if isinstance(opt,DistributedOptimizer):opt.load_parameter_state_from_dp_zero(item['state'])
        else:opt.load_state_dict(item['state'])
    bound['opt_param_scheduler'].load_state_dict(restored['scheduler'])
    del restored
    torch.cuda.synchronize()
    assert base.full_state(bound)==before,'Actual model/master/Adam/scheduler changed during RAM roundtrip'
    assert base.rng_hash()==rng_before,'RNG changed during RAM roundtrip'
    repacked=[]
    for opt in opts:
        if isinstance(opt,DistributedOptimizer):repacked.append({'kind':'distributed-dp-zero','state':opt.get_parameter_state_dp_zero()})
        else:repacked.append({'kind':'float16-wrapper-state-dict','state':opt.state_dict()})
    assert base.fingerprint(repacked)==native_fp;del repacked
    (root/f'ram-roundtrip-rank{rank}.json').write_text(json.dumps({'passed':True,'rank':rank,
      'after_completed_step':9,'serialized_bytes':size,'optimizer_classes':[type(o).__name__ for o in opts],
      'crc32':True,'full_state_exact':True,'rng_exact':True,'native_repack_exact':True,
      'scope':'RAM-only serialization and native optimizer/strict model loaders; same process; NOT durable or fresh-process resume'},indent=2)+'\n')
    dist.barrier()

base.ram_roundtrip=roundtrip
if __name__=='__main__':base.main()
