"""Match native scheduler loader's fresh-counter lifecycle in the RAM audit."""
from pathlib import Path
import hashlib
import corex5_numeric_worker as base

original=Path(__file__).with_name('corex5_numeric_worker_v2.py')
source=original.read_text()
needle="    bound['opt_param_scheduler'].load_state_dict(restored['scheduler'])"
assert source.count(needle)==1
source=source.replace(needle,"""    # Native load_state_dict adds saved num_steps; a fresh scheduler starts at 0.
    # Emulate that lifecycle, then demand the original full-state equality.
    scheduler=bound['opt_param_scheduler']
    import copy
    fresh_scheduler=copy.copy(scheduler);fresh_scheduler.num_steps=0
    fresh_scheduler.load_state_dict(restored['scheduler'])
    scheduler.__dict__.update(fresh_scheduler.__dict__)""")
needle="    assert base.full_state(bound)==before,'Actual model/master/Adam/scheduler changed during RAM roundtrip'"
assert source.count(needle)==1
source=source.replace(needle,"""    after=base.full_state(bound)
    if after!=before:
        differences=[]
        def visit(a,b,path=''):
            if type(a) is not type(b):differences.append({'path':path,'before_type':str(type(a)),'after_type':str(type(b))});return
            if isinstance(a,dict):
                if a.keys()!=b.keys():differences.append({'path':path,'keys_changed':True});return
                for k in a:visit(a[k],b[k],path+'/'+str(k))
            elif isinstance(a,list):
                if len(a)!=len(b):differences.append({'path':path,'length_changed':True});return
                for i,(x,y) in enumerate(zip(a,b)):visit(x,y,path+'/'+str(i))
            elif a!=b:differences.append({'path':path,'before':a,'after':b})
        visit(before,after)
        (root/f'ram-state-differences-rank{rank}.json').write_text(json.dumps({'count':len(differences),'differences':differences[:100]},indent=2)+'\\n')
    assert after==before,'Actual model/master/Adam/scheduler changed during RAM roundtrip'""")
namespace={'__name__':'corex5_ram_v3','__file__':str(original)}
exec(compile(source,str(original),'exec'),namespace)
if __name__=='__main__':base.main()
