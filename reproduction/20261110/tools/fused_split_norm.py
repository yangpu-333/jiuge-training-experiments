"""Fused row arithmetic with the two validated autograd edges preserved."""
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from torch.autograd.function import once_differentiable
from native_order_reductions import combine_native_row, rounded_bf16


@triton.jit
def fused_statistics_output(X, XF, W, R, Y, EPS:tl.constexpr, B:tl.constexpr):
    row=tl.program_id(0)
    lane=tl.arange(0,64)
    a0=tl.full((64,),0.,tl.float32);a1=tl.full((64,),0.,tl.float32)
    a2=tl.full((64,),0.,tl.float32);a3=tl.full((64,),0.,tl.float32)
    for group in tl.static_range(0,10):
        i=row*2560+lane*4+group*256
        x0=tl.load(XF+i);x1=tl.load(XF+i+1);x2=tl.load(XF+i+2);x3=tl.load(XF+i+3)
        a0=a0+x0*x0;a1=a1+x1*x1;a2=a2+x2*x2;a3=a3+x3*x3
    total=combine_native_row(a0,a1,a2,a3)
    mean=total*(1./2560.)
    # Materialize the FP32 boundary in the existing 512-element R buffer.
    # Volatile reload prevents fusing its reciprocal multiplication with eps.
    tl.store(R+row,mean)
    tl.debug_barrier()
    mean=tl.load(R+row,volatile=True)
    r=tl.rsqrt(mean+EPS)
    tl.store(R+row,r)
    # Use the same 64-lane domain as the validated native-order reduction.
    for group in tl.static_range(0,10):
        for element in tl.static_range(0,4):
            c=lane*4+group*256+element
            x=tl.load(X+row*2560+c).to(tl.float32)
            w=tl.load(W+c).to(tl.float32)
            normalized=(x*r).to(tl.bfloat16).to(tl.float32)
            tl.store(Y+row*2560+c,normalized*w)


@triton.jit
def fused_row_dual_dx(X,W,G,R,DIRECT,VARIANCE,B:tl.constexpr):
    row=tl.program_id(0)
    lane=tl.arange(0,64)
    a0=tl.full((64,),0.,tl.float32);a1=tl.full((64,),0.,tl.float32)
    a2=tl.full((64,),0.,tl.float32);a3=tl.full((64,),0.,tl.float32)
    for group in tl.static_range(0,10):
        c=lane*4+group*256;i=row*2560+c
        w0=tl.load(W+c).to(tl.float32);w1=tl.load(W+c+1).to(tl.float32)
        w2=tl.load(W+c+2).to(tl.float32);w3=tl.load(W+c+3).to(tl.float32)
        g0=tl.load(G+i).to(tl.float32);g1=tl.load(G+i+1).to(tl.float32)
        g2=tl.load(G+i+2).to(tl.float32);g3=tl.load(G+i+3).to(tl.float32)
        gw0=rounded_bf16(g0*w0);gw1=rounded_bf16(g1*w1)
        gw2=rounded_bf16(g2*w2);gw3=rounded_bf16(g3*w3)
        x0=tl.load(X+i).to(tl.float32);x1=tl.load(X+i+1).to(tl.float32)
        x2=tl.load(X+i+2).to(tl.float32);x3=tl.load(X+i+3).to(tl.float32)
        a0=a0+gw0*x0;a1=a1+gw1*x1;a2=a2+gw2*x2;a3=a3+gw3*x3
    dot=combine_native_row(a0,a1,a2,a3)
    r=tl.load(R+row)
    cube=libdevice.mul_rn(libdevice.mul_rn(r,r),r)
    unscaled=libdevice.mul_rn(libdevice.mul_rn(dot,-0.5),cube)
    coefficient=libdevice.mul_rn(unscaled,1./2560.)
    for group in tl.static_range(0,10):
        for element in tl.static_range(0,4):
            c=lane*4+group*256+element;i=row*2560+c
            w=tl.load(W+c).to(tl.float32)
            g=tl.load(G+i).to(tl.float32)
            x=tl.load(X+i).to(tl.float32)
            gw=rounded_bf16(g*w)
            tl.store(DIRECT+i,gw*r)
            tl.store(VARIANCE+i,libdevice.mul_rn(coefficient,libdevice.mul_rn(2.,x)))


class FusedSplitRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,xf,weight,eps):
        assert tuple(x.shape)==(512,1,2560) and tuple(weight.shape)==(2560,)
        assert x.dtype==weight.dtype==torch.bfloat16 and xf.dtype==torch.float32
        assert x.is_contiguous() and xf.is_contiguous() and weight.is_contiguous()
        r=torch.empty(512,device=x.device,dtype=torch.float32)
        y=torch.empty_like(x)
        fused_statistics_output[(512,)](x,xf,weight,r,y,eps,4096,num_warps=1,enable_fp_fusion=False)
        ctx.save_for_backward(x,weight,r)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx,grad_output):
        import jiuge_ops._triton as original
        x,weight,r=ctx.saved_tensors
        g=grad_output.contiguous()
        direct=variance=dw=None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            direct=torch.empty_like(x)
            variance=torch.empty(x.shape,device=x.device,dtype=torch.float32)
            fused_row_dual_dx[(512,)](x,weight,g,r,direct,variance,4096,num_warps=1,enable_fp_fusion=False)
        if ctx.needs_input_grad[2]:
            products=torch.empty(x.shape,device=x.device,dtype=weight.dtype)
            partial=torch.empty((16,2560),device=x.device,dtype=torch.float32)
            dw=torch.empty_like(weight)
            original._weight_products[(16,40)](x,g,r,products,512,2560,32,64,num_warps=4)
            original._backward_weight_partial[(16,40)](products,partial,512,2560,32,64,num_warps=4)
            original._backward_weight_reduce[(80,)](partial,dw,16,2560,16,32,num_warps=4)
        return direct,variance,dw,None


def rms_norm(x,weight,eps):
    # Deliberately outside Function: the FP32 cast has its own backward edge.
    return FusedSplitRMSNorm.apply(x,x.float(),weight,eps)


def install():
    import jiuge_ops._triton as implementation
    implementation.triton_rms_norm=rms_norm
