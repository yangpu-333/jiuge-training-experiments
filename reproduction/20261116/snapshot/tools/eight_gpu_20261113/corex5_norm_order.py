"""Match installed Torch2.10 Reduce.cuh decreasing shuffle offsets (32..1).

Only runtime bindings change; original 4.4 candidate files remain untouched.
"""
import triton
import triton.language as tl

@triton.jit
def combine_corex5_row(a0,a1,a2,a3):
    value=((a0+a1)+a2)+a3
    value=tl.sum(tl.reshape(value,(2,32)),axis=0)
    value=tl.sum(tl.reshape(value,(2,16)),axis=0)
    value=tl.sum(tl.reshape(value,(2,8)),axis=0)
    value=tl.sum(tl.reshape(value,(2,4)),axis=0)
    value=tl.sum(tl.reshape(value,(2,2)),axis=0)
    value=tl.sum(tl.reshape(value,(2,1)),axis=0)
    return tl.sum(value,axis=0)

def install_order():
    import native_order_reductions,fused_split_norm,fused_norm_dw
    for module in [native_order_reductions,fused_split_norm,fused_norm_dw]:
        module.combine_native_row=combine_corex5_row

def install(task):
    install_order()
    import fused_split_norm,fused_norm_dw
    (fused_norm_dw if task=='cpt' else fused_split_norm).install()
