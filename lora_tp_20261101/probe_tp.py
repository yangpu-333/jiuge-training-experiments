"""Four-rank TP2 exact output/gradient probe before any model integration."""
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist

BASE=Path('/private/yp/jiuge-training')
OUT=BASE/'checks/20261102-lora-tp/probe'

def fp(t):
    x=t.detach().contiguous().cpu()
    return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()}

def main():
    rank=int(os.environ['RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    groups=[dist.new_group([0,1]),dist.new_group([2,3])]
    group=groups[rank//2]
    import lora_linear_cpp,lora_tp_linear
    from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region,reduce_from_tensor_model_parallel_region
    old=lora_linear_cpp.extension();new=lora_tp_linear.extension()
    assert group.size()==2
    rows=[]
    for kind in ['column','row']:
      for width in [1280,2560,3200]:
       for grad_x in [False,True]:
        for strided in [False,True]:
         for zero in [False,True]:
          torch.manual_seed(1000+rank*31+width)
          raw=torch.randn((32,1,width*(2 if strided else 1)),device='cuda',dtype=torch.bfloat16)
          x0=raw[...,::2] if strided else raw
          w0=torch.randn((8,width),device='cuda',dtype=torch.bfloat16)
          if zero:w0.zero_()
          # Independent leaves; column backward all-reduce mutates contiguous g.
          x1=x0.detach().requires_grad_(grad_x)
          x2=(raw.detach().clone()[...,::2] if strided else raw.detach().clone()).detach().requires_grad_(grad_x)
          w1=w0.detach().clone().requires_grad_();w2=w0.detach().clone().requires_grad_()
          y1=old.linear(x1,w1)
          if kind=='column':y1=copy_to_tensor_model_parallel_region(y1,group=group)
          else:y1=reduce_from_tensor_model_parallel_region(y1,group=group)
          y2=new.linear_tp(x2,w2,group,kind=='column')
          torch.manual_seed(2000+rank*29+width)
          g=torch.randn_like(y1)
          inputs1=(x1,w1) if grad_x else (w1,)
          inputs2=(x2,w2) if grad_x else (w2,)
          grad1=torch.autograd.grad(y1,inputs1,g.clone())
          grad2=torch.autograd.grad(y2,inputs2,g.clone())
          exact=torch.equal(y1,y2) and all(torch.equal(a,b) for a,b in zip(grad1,grad2))
          finite=bool(torch.isfinite(y1).all() and all(torch.isfinite(a).all() for a in grad1+grad2))
          row={'kind':kind,'width':width,'grad_x':grad_x,'strided':strided,'zero_weight':zero,'exact':exact,'finite':finite,
               'output_ref':fp(y1),'output_candidate':fp(y2),'grads_ref':[fp(a) for a in grad1],'grads_candidate':[fp(a) for a in grad2]}
          rows.append(row)
          if not(exact and finite):
              (OUT/f'rank-{rank}.json').write_text(json.dumps(rows,indent=2)+'\n')
              raise AssertionError(row)
    (OUT/f'rank-{rank}.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('LORA_TP_PROBE_PASS',rank,len(rows),flush=True)
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
