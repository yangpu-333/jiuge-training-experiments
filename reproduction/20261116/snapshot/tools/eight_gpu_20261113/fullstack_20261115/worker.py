"""Layered numerical and native disk workers, never skipping audit gates."""
from pathlib import Path
import json,os,runpy,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(1,str(HERE.parent))
import stack
cfgpath=Path(sys.argv[sys.argv.index('--jiuge-config')+1]);cfg=json.loads(cfgpath.read_text());out=cfgpath.parent
stage=os.environ['FULLSTACK_STAGE'];mode=os.environ['FULLSTACK_MODE']
if mode=='numeric':
    stack.install(stage,cfg,out)
    runpy.run_path(str(HERE.parent/'corex5_numeric_worker_v6.py'),run_name='__main__')
else:
    if stage!='reference':stack.install(stage,cfg,out)
    if stage in ['reference','adapted-reference']:assert cfg['backend']['rmsnorm']=='reference'
    stack.install_persistence(out,cfg['task']['save'],stage=='packed',mode=='restore')
    if mode=='performance':
        import atexit,functools,inspect,math
        import megatron.training.training as mcore
        rows=[];original=mcore.train_step
        @functools.wraps(original)
        def step(*args,**kwargs):
            bound=inspect.signature(original).bind(*args,**kwargs).arguments
            result=original(*args,**kwargs)
            row={'iteration':int(bound['iteration']),'skipped':int(result[1]),
                 'grad_norm':float(result[5]),'losses':{k:float(v.detach().item()) for k,v in result[0].items()}}
            assert row['skipped']==0 and math.isfinite(row['grad_norm']) and all(math.isfinite(x) for x in row['losses'].values())
            rows.append(row);return result
        mcore.train_step=step
        atexit.register(lambda:(out/f'performance-steps-rank{os.environ["RANK"]}.json').write_text(json.dumps(rows)+'\n'))
    import corex5_disk_worker
    corex5_disk_worker.main()
