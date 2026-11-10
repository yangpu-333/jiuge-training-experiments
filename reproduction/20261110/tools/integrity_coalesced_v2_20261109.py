"""Real checkpoint save N, continue N+1, then native resume N -> N+1."""
from pathlib import Path
import functools,inspect,json,os,sys
import e2e_training_20261106 as base
base.SCRIPT=Path(__file__).resolve()

def worker():
    sys.argv.remove('--e2e-worker')
    root=Path(sys.argv[sys.argv.index('--jiuge-config')+1]).parent
    cfg=json.loads((root/'config.json').read_text());target=int(os.environ['AUDIT_SAVE_ITER']);mode=os.environ['BENCH_ARM']
    assert cfg['task']['train_iters']==target+1 and cfg['task']['save_interval']==target
    assert cfg['task']['name']=='cpt' and cfg['task']['recompute'] is False and cfg['task']['seq_length']==512
    assert cfg['parallel']['tp']==cfg['parallel']['pp']==2
    import jiuge_training.train as baseline
    baseline.configure_worker_caches();baseline.configure_worker_caches=lambda:None
    import torch
    assert torch.cuda.device_count()==4
    if mode=='baseline':
        import local_spec_ablation_patch
        os.environ['LOCAL_SPEC_ABLATION']='attention-only';local_spec_ablation_patch.install()
    else:
        import lora_linear_cpp,fused_norm_dw,swiglu_fast_patch_20261106
        lora_linear_cpp.install();fused_norm_dw.install();swiglu_fast_patch_20261106.install()
    from checkpoint_round_common_v2_20261109 import install
    from savepack_resume_20261107 import fingerprint as fp
    from checkpoint_diagnostic_20261107 import fingerprint as cpu_fp
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    import megatron.training.training as mcore
    import audit_training
    finish=install(root,mode,audit=True);original_step=mcore.train_step;original_save=mcore.save_checkpoint;progress={}
    def optimizer_fp(opt):
        if hasattr(opt,'chained_optimizers'):return {'children':[optimizer_fp(x) for x in opt.chained_optimizers]}
        return {'class':type(opt).__name__,'wrapper_metadata':fp(opt.state_dict()),'inner_state':fp(opt.optimizer.state_dict()),'master_parameters':fp([[p for p in g['params']] for g in opt.optimizer.param_groups])}
    def snapshot(bound):return {'model':[fp(x.state_dict()) for x in bound['model']],'optimizer':optimizer_fp(bound['optimizer']),'scheduler':fp(bound['opt_param_scheduler'].state_dict())}
    @functools.wraps(original_step)
    def step(*a,**kw):
        bound=inspect.signature(original_step).bind(*a,**kw).arguments;iteration=bound['iteration']
        if iteration==target and cfg['task'].get('load'):
            torch.cuda.synchronize()
            (root/f'loaded-state-rank{os.environ["RANK"]}.json').write_text(json.dumps(snapshot(bound),indent=2)+'\n')
            opt=bound['optimizer'];opts=getattr(opt,'chained_optimizers',[opt]);assert len(opts)==1
            state=DistributedOptimizer.get_parameter_state_dp_zero(opts[0])
            (root/f'loaded-optimizer-cpu-rank{os.environ["RANK"]}.json').write_text(json.dumps(cpu_fp(state),indent=2)+'\n');del state
        result=original_step(*a,**kw);progress['iteration']=int(iteration+1)
        if iteration in [target-1,target]:
            torch.cuda.synchronize();name='saved-state' if iteration==target-1 else 'final-state'
            (root/f'{name}-rank{os.environ["RANK"]}.json').write_text(json.dumps(snapshot(bound),indent=2)+'\n')
        return result
    @functools.wraps(original_save)
    def save(*a,**kw):
        bound=inspect.signature(original_save).bind(*a,**kw).arguments
        if not cfg['task'].get('load') and bound['iteration']==target:return original_save(*a,**kw)
        print('INTEGRITY_ONLY_EXTRA_SAVE_SKIPPED',bound['iteration'],flush=True)
    mcore.train_step=step;mcore.save_checkpoint=save
    sys.argv.append('--audit-worker')
    try:
        audit_training.worker()
        args=mcore.get_args();assert progress['iteration']==target+1
        (root/f'final-counters-rank{os.environ["RANK"]}.json').write_text(json.dumps({'iteration':progress['iteration'],'consumed_train_samples':args.consumed_train_samples,'consumed_valid_samples':args.consumed_valid_samples},indent=2)+'\n')
    finally:mcore.train_step=original_step;mcore.save_checkpoint=original_save;finish()

if __name__=='__main__':worker() if '--e2e-worker' in sys.argv else base.driver()
