#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <stdint.h>
__device__ __forceinline__ float f(uint16_t x){return __uint_as_float((uint32_t)x<<16);}
__device__ __forceinline__ uint16_t b(float x){uint32_t u=__float_as_uint(x);if ((u&0x7fffffff)>0x7f800000)return 0x7fc0;return (u+0x7fff+((u>>16)&1))>>16;}
__global__ void forward(const uint16_t* x,uint16_t* y,int64_t n,int h){int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int64_t a=(i/h)*(2*h)+i%h;float v=f(x[a]);float s=1.f/(1.f+expf(-v));y[i]=b(f(b(v*s))*f(x[a+h]));}}
__global__ void backward(const uint16_t* x,const uint16_t* g,uint16_t* dx,int64_t n,int h){int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int64_t a=(i/h)*(2*h)+i%h;float v=f(x[a]),z=f(x[a+h]),q=f(g[i]);float s=1.f/(1.f+expf(-v));float gs=f(b(q*z));dx[a]=b((gs*s)*(1.f+v*(1.f-s)));dx[a+h]=b(q*f(b(v*s)));}}
at::Tensor swiglu_forward_cuda(at::Tensor x){c10::cuda::CUDAGuard guard(x.device());auto sizes=x.sizes().vec();sizes.back()/=2;auto y=at::empty(sizes,x.options());int h=sizes.back();int64_t n=y.numel();forward<<<(n+255)/256,256,0,c10::cuda::getCurrentCUDAStream().stream()>>>((uint16_t*)x.data_ptr(),(uint16_t*)y.data_ptr(),n,h);C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}
at::Tensor swiglu_backward_cuda(at::Tensor x,at::Tensor g){c10::cuda::CUDAGuard guard(x.device());auto y=at::empty_like(x);int h=x.size(-1)/2;int64_t n=g.numel();backward<<<(n+255)/256,256,0,c10::cuda::getCurrentCUDAStream().stream()>>>((uint16_t*)x.data_ptr(),(uint16_t*)g.data_ptr(),(uint16_t*)y.data_ptr(),n,h);C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}
