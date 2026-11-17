"""Candidate: retain fused row path, use current Torch for weight-column sum.

Not promoted until isolated and original full-model audits pass.
"""
import torch
from torch.autograd.function import once_differentiable
import fused_split_norm,fused_norm_dw

class NativeColumnNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,xf,w,eps):return fused_split_norm.FusedSplitRMSNorm.forward(ctx,x,xf,w,eps)
    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        import jiuge_ops._triton as original
        x,w,r=ctx.saved_tensors;g=g.contiguous()
        direct=variance=None;products=torch.empty(x.shape,device=x.device,dtype=w.dtype)
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            direct=torch.empty_like(x);variance=torch.empty(x.shape,device=x.device,dtype=torch.float32)
            fused_norm_dw.fused_row_dual_dx_products[(512,)](x,w,g,r,direct,variance,products,4096,num_warps=1,enable_fp_fusion=False)
        else:original._weight_products[(16,40)](x,g,r,products,512,2560,32,64,num_warps=4)
        dw=products.sum(dim=(0,1))
        return direct,variance,dw,None

def rms_norm(x,w,eps):
    if not w.requires_grad:return fused_split_norm.rms_norm(x,w,eps)
    return NativeColumnNorm.apply(x,x.float(),w,eps)

def install():fused_norm_dw.rms_norm=rms_norm
