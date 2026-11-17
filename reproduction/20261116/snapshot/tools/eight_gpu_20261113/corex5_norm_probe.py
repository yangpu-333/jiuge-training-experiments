"""Localize CoreX5 differences: statistics/output and independent backward edges."""
from pathlib import Path
import json,torch
import fused_split_norm,native_order_reductions
from jiuge_ops import rms_norm
R=Path('/private/yp/jiuge-training/checks/20261113-eight-gpu-corex5')

def metrics(a,b):
    a=a.detach().double().cpu().flatten();b=b.detach().double().cpu().flatten();d=b-a
    norm=float(a.norm());other=float(b.norm())
    return {'elements':a.numel(),'unequal':int((a!=b).sum()),'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
      'max_abs':float(d.abs().max()),'relative_l2':float(d.norm())/norm if norm else float(d.norm()),
      'cosine':float(a.dot(b))/(norm*other) if norm*other else None}

def main():
    torch.manual_seed(1234);rows=[]
    for scale in [0.01,1.0,8.0]:
        x=(torch.randn((512,1,2560),device='cuda')*scale).bfloat16()
        w=(1+torch.randn(2560,device='cuda')*0.1).bfloat16()
        g=torch.randn_like(x);xf=x.float();eps=1e-6
        mean_ref=xf.square().mean(-1).reshape(512);r_ref=torch.rsqrt(mean_ref+eps)
        mean_kernel,r_kernel=native_order_reductions.statistics(xf,eps)
        r_fused=torch.empty_like(r_ref);y_fused=torch.empty_like(x)
        fused_split_norm.fused_statistics_output[(512,)](x,xf,w,r_fused,y_fused,eps,4096,num_warps=1,enable_fp_fusion=False)
        row={'scale':scale,'mean':metrics(mean_ref,mean_kernel),'r_native_reduce':metrics(r_ref,r_kernel),'r_fused':metrics(r_ref,r_fused)}
        xd=x.detach().requires_grad_();xf_leaf=xf.detach().requires_grad_();wd=w.detach().requires_grad_()
        ref=((xd*torch.rsqrt(xf_leaf.square().mean(-1,keepdim=True)+eps)).to(x.dtype)*wd).to(x.dtype)
        direct,variance,dw=torch.autograd.grad(ref,[xd,xf_leaf,wd],g)
        xc=x.detach().requires_grad_();xfc=xf.detach().requires_grad_();wc=w.detach().requires_grad_()
        cand=fused_split_norm.FusedSplitRMSNorm.apply(xc,xfc,wc,eps)
        c_direct,c_variance,c_dw=torch.autograd.grad(cand,[xc,xfc,wc],g)
        row.update(output=metrics(ref,cand),direct=metrics(direct,c_direct),variance=metrics(variance,c_variance),dw=metrics(dw,c_dw),
           total_dx=metrics((direct+variance.to(x.dtype)).to(x.dtype),(c_direct+c_variance.to(x.dtype)).to(x.dtype)))
        gw,dot,coefficient=native_order_reductions.row_vjp(x,w,g,r_ref.reshape(512,1,1))
        row['dot_with_torch_r']=metrics(((g*w).float()*xf).sum(-1).reshape(512),dot)
        rows.append(row);print(json.dumps(row),flush=True)
    with (R/'norm-probe-v1.json').open('x') as f:json.dump({'cases':rows,'scope':'isolation only; no numerical gate or speed acceptance'},f,indent=2)

if __name__=='__main__':main()
