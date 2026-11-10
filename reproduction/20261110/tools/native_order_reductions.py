#!/usr/bin/env python3
"""BI-V150 H2560 row reductions preserving the observed native FP32 order.

Specialized for contiguous [512,1,2560], warp64, four accumulators/lane.
The native layout and combine order were inspected in the installed Torch
Reduce.cuh and verified by a short actual-device PyTorch profiler trace.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def combine_native_row(a0, a1, a2, a3):
    value = ((a0 + a1) + a2) + a3
    # Explicit adjacent pairs match native shuffle offsets 1,2,4,8,16,32.
    # This CoreX language build has no tl.gather.
    value = tl.sum(tl.reshape(value, (32, 2)), axis=1)
    value = tl.sum(tl.reshape(value, (16, 2)), axis=1)
    value = tl.sum(tl.reshape(value, (8, 2)), axis=1)
    value = tl.sum(tl.reshape(value, (4, 2)), axis=1)
    value = tl.sum(tl.reshape(value, (2, 2)), axis=1)
    value = tl.sum(tl.reshape(value, (1, 2)), axis=1)
    return tl.sum(value, axis=0)


@triton.jit
def forward_statistics(XF, MEAN, R, EPS: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, 64)
    a0 = tl.full((64,), 0.0, tl.float32)
    a1 = tl.full((64,), 0.0, tl.float32)
    a2 = tl.full((64,), 0.0, tl.float32)
    a3 = tl.full((64,), 0.0, tl.float32)
    for group in tl.static_range(0, 10):
        index = row * 2560 + lane * 4 + group * 256
        x0 = tl.load(XF + index + 0)
        x1 = tl.load(XF + index + 1)
        x2 = tl.load(XF + index + 2)
        x3 = tl.load(XF + index + 3)
        a0 = a0 + x0 * x0
        a1 = a1 + x1 * x1
        a2 = a2 + x2 * x2
        a3 = a3 + x3 * x3
    total = combine_native_row(a0, a1, a2, a3)
    mean = total * (1.0 / 2560.0)
    tl.store(MEAN + row, mean)
    tl.store(R + row, tl.rsqrt(mean + EPS))


@triton.jit
def rounded_bf16(v):
    bits = v.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    result = rounded.to(tl.float32, bitcast=True)
    return tl.where((bits & 0x7F800000) == 0x7F800000, v, result)


@triton.jit
def backward_row(X, W, G, R, GW, DOT, COEFFICIENT, DIV_RN: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, 64)
    a0 = tl.full((64,), 0.0, tl.float32)
    a1 = tl.full((64,), 0.0, tl.float32)
    a2 = tl.full((64,), 0.0, tl.float32)
    a3 = tl.full((64,), 0.0, tl.float32)
    for group in tl.static_range(0, 10):
        col = lane * 4 + group * 256
        index = row * 2560 + col
        w0 = tl.load(W + col + 0).to(tl.float32)
        w1 = tl.load(W + col + 1).to(tl.float32)
        w2 = tl.load(W + col + 2).to(tl.float32)
        w3 = tl.load(W + col + 3).to(tl.float32)
        g0 = tl.load(G + index + 0).to(tl.float32)
        g1 = tl.load(G + index + 1).to(tl.float32)
        g2 = tl.load(G + index + 2).to(tl.float32)
        g3 = tl.load(G + index + 3).to(tl.float32)
        gw0 = rounded_bf16(g0 * w0)
        gw1 = rounded_bf16(g1 * w1)
        gw2 = rounded_bf16(g2 * w2)
        gw3 = rounded_bf16(g3 * w3)
        tl.store(GW + index + 0, gw0)
        tl.store(GW + index + 1, gw1)
        tl.store(GW + index + 2, gw2)
        tl.store(GW + index + 3, gw3)
        x0 = tl.load(X + index + 0).to(tl.float32)
        x1 = tl.load(X + index + 1).to(tl.float32)
        x2 = tl.load(X + index + 2).to(tl.float32)
        x3 = tl.load(X + index + 3).to(tl.float32)
        a0 = a0 + gw0 * x0
        a1 = a1 + gw1 * x1
        a2 = a2 + gw2 * x2
        a3 = a3 + gw3 * x3
    dot = combine_native_row(a0, a1, a2, a3)
    r = tl.load(R + row)
    cube = (r * r) * r
    unscaled = (dot * -0.5) * cube
    if DIV_RN:
        coefficient = tl.div_rn(unscaled, 2560.0)
    else:
        # Native Torch scalar division here multiplies an FP32 reciprocal.
        # An explicit division has a different (more exact) rounding path.
        coefficient = unscaled * (1.0 / 2560.0)
    tl.store(DOT + row, dot)
    tl.store(COEFFICIENT + row, coefficient)


def statistics(xf, eps):
    assert tuple(xf.shape) == (512, 1, 2560) and xf.dtype == torch.float32 and xf.is_contiguous()
    mean = torch.empty(512, dtype=torch.float32, device=xf.device)
    r = torch.empty_like(mean)
    forward_statistics[(512,)](xf, mean, r, eps, num_warps=1, enable_fp_fusion=False)
    return mean, r


def row_vjp(x, w, g, r, div_rn=False):
    assert tuple(x.shape) == tuple(g.shape) == (512, 1, 2560) and tuple(w.shape) == (2560,)
    assert x.dtype == w.dtype == g.dtype == torch.bfloat16
    assert x.is_contiguous() and w.is_contiguous() and g.is_contiguous()
    gw = torch.empty_like(x)
    dot = torch.empty(512, dtype=torch.float32, device=x.device)
    coefficient = torch.empty_like(dot)
    backward_row[(512,)](x, w, g, r, gw, dot, coefficient, div_rn, num_warps=1, enable_fp_fusion=False)
    return gw, dot, coefficient
