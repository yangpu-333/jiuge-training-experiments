import json
from pathlib import Path
import statistics
import time
import torch
import torch.nn.functional as F
from lora_linear_cpp import linear

BASE=Path('/private/yp/jiuge-training');ROOT=BASE/'checks/20261101-lora-linear'

def main():
    results=[];timings=[];torch.manual_seed(20261101)
    shapes=[(2560,8),(1280,8),(3200,8),(8,1536),(8,2560),(8,6400)]
    for k,n in shapes:
        for scale in [0.,1e-3,1.,10.]:
            x=(torch.randn(512,1,k,device='cuda')*scale).bfloat16()
            w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16);g=torch.randn(512,1,n,device='cuda',dtype=torch.bfloat16)
            for requires_x in [False,True]:
                out=[]
                for fn in [F.linear,linear]:
                    xx=x.detach().requires_grad_(requires_x);ww=w.detach().requires_grad_()
                    y=fn(xx,ww);grads=torch.autograd.grad(y,(xx,ww) if requires_x else (ww,),g)
                    out.append((y,grads))
                unequal=[int((out[0][0]!=out[1][0]).sum())]+[int((a!=b).sum()) for a,b in zip(out[0][1],out[1][1])]
                results.append({'k':k,'n':n,'scale':scale,'requires_x':requires_x,'unequal':unequal})
        x=torch.randn(512,1,k,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        g=torch.randn(512,1,n,device='cuda',dtype=torch.bfloat16)
        def run(fn):
            y=fn(x,w);torch.autograd.grad(y,(x,w),g)
        for rep,order in enumerate([[F.linear,linear],[linear,F.linear],[F.linear,linear],[linear,F.linear]]):
            for fn in order:
                for _ in range(10):run(fn)
                torch.cuda.synchronize();start=time.perf_counter();cpu=time.process_time()
                for _ in range(100):run(fn)
                torch.cuda.synchronize()
                timings.append({'k':k,'n':n,'repeat':rep,'arm':'native' if fn==F.linear else 'manual','wall_us_per_call':(time.perf_counter()-start)*1e4,'process_cpu_us_per_call':(time.process_time()-cpu)*1e4})
        print('LINEAR_PROBE',k,n,{a:statistics.mean(x['wall_us_per_call'] for x in timings if x['k']==k and x['n']==n and x['arm']==a) for a in ['native','manual']},flush=True)
    result={'numerical':results,'timings':timings,'scope':'Isolated BF16 native GEMMs, forward+first-order VJP. Not whole-model numerical/performance acceptance.'}
    (ROOT/'local-linear-cpp-probe.json').write_text(json.dumps(result,indent=2)+'\n')
    assert all(all(x==0 for x in r['unequal']) for r in results)
    print('LORA_LINEAR_LOCAL_EXACT',len(results),flush=True)

if __name__=='__main__':main()
