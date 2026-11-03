import json
from pathlib import Path
import torch
from reduction_patch import FullReductionRMSNorm, rms_norm as previous
from fused_split_norm import FusedSplitRMSNorm, rms_norm as fused
from probe_gradient_graph import metric

BASE=Path('/private/yp/jiuge-training')
ROOT=BASE/'checks/20261031-fused-norm'

def main():
    rows=[]
    cases=[]
    for rank in [0,2]:
        records=torch.load(BASE/f'runs/4b-cpt-tp2-pp2-reference-no-recompute-s512-gbs16-backward-capture-v1/norm-capture-rank{rank}.pt',weights_only=True,map_location='cpu',mmap=True)
        for name,row in records.items():cases.append((f'rank{rank}/{name}',row))
    generator=torch.Generator().manual_seed(20261031)
    for seed in [0,1]:
        for scale in [0.,1e-4,1.,100.]:
            cases.append((f'synthetic-{seed}-{scale}',{'x':(torch.randn(512,1,2560,generator=generator)*scale).bfloat16(),'w':torch.randn(2560,generator=generator).bfloat16(),'g':torch.randn(512,1,2560,generator=generator).bfloat16(),'eps':1e-6}))
    for index,(name,row) in enumerate(cases):
        x,w,g=[row[k].cuda() for k in ['x','w','g']]
        outputs=[]
        for cls in [FullReductionRMSNorm,FusedSplitRMSNorm]:
            xx=x.detach().requires_grad_();xf=x.float().detach().requires_grad_();ww=w.detach().requires_grad_()
            y=cls.apply(xx,xf,ww,row['eps'])
            dx,dxf,dw=torch.autograd.grad(y,(xx,xf,ww),g)
            outputs.append([y,dx,dxf,dw])
        record={'name':name,**{key:metric(a,b) for key,a,b in zip(['output','direct','variance_fp32','dw'],*outputs)}}
        residual=[]
        for fn in [lambda a,b,e:previous(a,b,e,'full'),fused]:
            xx=x.detach().requires_grad_();ww=w.detach().requires_grad_()
            y=fn(xx,ww,row['eps'])
            assert y.grad_fn.next_functions[0][0] is not None
            assert 'ToCopyBackward' in type(y.grad_fn.next_functions[1][0]).__name__
            residual.append(torch.autograd.grad((y,xx),xx,(g,g.flip(-1).contiguous()))[0])
        record['residual_dx']=metric(*residual)
        # Frozen norm weight is the real LoRA path; direct/variance still separate.
        frozen=[]
        for fn in [lambda a,b,e:previous(a,b,e,'full'),fused]:
            xx=x.detach().requires_grad_()
            frozen.append(torch.autograd.grad(fn(xx,w.detach(),row['eps']),xx,g)[0])
        record['frozen_weight_dx']=metric(*frozen)
        rows.append(record)
        if index%20==0:print('FUSED_PARTIAL',index,name,{k:record[k]['unequal'] for k in ['output','direct','variance_fp32','dw','residual_dx','frozen_weight_dx']},flush=True)
    keys=['output','direct','variance_fp32','dw','residual_dx','frozen_weight_dx']
    summary={k:{'unequal':sum(r[k]['unequal'] for r in rows),'finite':all(r[k]['finite'] for r in rows),'max_relative_l2':max(r[k]['relative_l2'] or 0 for r in rows)} for k in keys}
    result={'cases':len(cases),'captured':125,'synthetic':8,'scope':'Exact parity against validated reductions-full-v1, including independent input edges, residual graph and frozen weight','summary':summary,'rows':rows}
    path=ROOT/'local-parity.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
    print('FUSED_LOCAL_PARITY',json.dumps(summary),flush=True)
    assert all(v['unequal']==0 and v['finite'] for v in summary.values())

if __name__=='__main__':main()
