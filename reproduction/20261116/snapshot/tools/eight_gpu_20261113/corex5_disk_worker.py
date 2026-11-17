"""Authorized local-disk native save/resume and uninstrumented E2E worker."""
from pathlib import Path
import functools,inspect,json,os,sys,hashlib
import corex5_numeric_worker as state
B=Path('/private/yp/jiuge-training')

def main():
    cfgpath=Path(sys.argv[sys.argv.index('--jiuge-config')+1]);cfg=json.loads(cfgpath.read_text());root=cfgpath.parent
    rank=int(os.environ['RANK']);mode=os.environ['COREX5_DISK_MODE']
    if cfg['backend']['rmsnorm']=='triton':
        import corex5_norm_order,corex5_norm_dw_native,fused_split_norm,fused_norm_dw
        corex5_norm_order.install_order()
        if cfg['task']['name']=='cpt':corex5_norm_dw_native.install();fused_norm_dw.install()
        else:fused_split_norm.install()
    import megatron.training.training as mcore
    original_step=mcore.train_step;original_save=mcore.save_checkpoint;progress={};saved=set()
    @functools.wraps(original_step)
    def step(*args,**kwargs):
        bound=inspect.signature(original_step).bind(*args,**kwargs).arguments
        if bound['iteration']==9 and cfg['task'].get('load'):
            (root/f'loaded-step9-rank{rank}.json').write_text(json.dumps(state.full_state(bound))+'\n')
        result=original_step(*args,**kwargs);progress['iteration']=bound['iteration']+1
        if bound['iteration'] in [8,9]:
            label='step9' if bound['iteration']==8 else 'final'
            (root/f'{label}-state-rank{rank}.json').write_text(json.dumps(state.full_state(bound))+'\n')
        return result
    @functools.wraps(original_save)
    def save(*args,**kwargs):
        bound=inspect.signature(original_save).bind(*args,**kwargs).arguments;i=bound['iteration']
        target=9 if mode=='restore' else 50
        if i==target and i not in saved and not cfg['task'].get('load'):
            saved.add(i);return original_save(*args,**kwargs)
    mcore.save_checkpoint=save
    if mode=='restore':
        mcore.train_step=step
        import audit_training
        audit_training.worker()
        settings=mcore.get_args()
        (root/f'counters-rank{rank}.json').write_text(json.dumps({'iteration':progress['iteration'],'samples':settings.consumed_train_samples,'valid_samples':settings.consumed_valid_samples})+'\n')
    else:
        sys.argv.remove('--audit-worker')
        import jiuge_training.train as baseline
        baseline.main()

if __name__=='__main__':main()
