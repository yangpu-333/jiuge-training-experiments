// Prepared candidate only: no change to SUM collectives or native BF16 GEMMs.
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

at::Tensor reduce_native(at::Tensor input,c10d::ProcessGroup* group) {
  auto output=input.contiguous();
  std::vector<at::Tensor> tensors{output};
  c10d::AllreduceOptions options;options.reduceOp=c10d::ReduceOp::SUM;
  auto work=group->allreduce(tensors,options);
  work->wait(); // Match async_op=False Python interface; do not synchronize device.
  return output;
}

class LoRATPLinear final : public torch::autograd::Function<LoRATPLinear> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
      torch::Tensor x,torch::Tensor weight,
      c10::intrusive_ptr<c10d::ProcessGroup> group,bool column) {
    TORCH_CHECK(group && group->getSize()==2,"TP2 only");
    TORCH_CHECK(x.scalar_type()==at::kBFloat16 && weight.scalar_type()==at::kBFloat16,
                "BF16 only");
    TORCH_CHECK(weight.dim()==2 && weight.size(0)==8 && x.size(-1)==weight.size(1),
                "LoRA A rank8 shape required");
    ctx->save_for_backward({x,weight});ctx->saved_data["column"]=column;
    if (column)ctx->saved_data["group"]=c10::IValue::make_capsule(group);
    auto output=at::linear(x,weight);
    return column?output:reduce_native(output,group.get());
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list gradients) {
    auto g=gradients[0];
    if (ctx->saved_data["column"].toBool()) {
      // Capsule retains precisely this PG until the backward context is freed.
      auto holder=ctx->saved_data["group"].toCapsule();
      g=reduce_native(g,static_cast<c10d::ProcessGroup*>(holder.get()));
    }
    auto saved=ctx->get_saved_variables();const auto& x=saved[0];const auto& w=saved[1];
    auto gg=g.reshape({-1,w.size(0)});auto xx=x.reshape({-1,w.size(1)});
    torch::Tensor dw,dx;
    if(ctx->needs_input_grad(1))dw=at::mm(gg.t(),xx);
    if(ctx->needs_input_grad(0))dx=at::mm(gg,w).reshape(x.sizes());
    return {dx,dw,torch::Tensor(),torch::Tensor()};
  }
};

torch::Tensor linear_tp(torch::Tensor x,torch::Tensor weight,
    c10::intrusive_ptr<c10d::ProcessGroup> group,bool column) {
  return LoRATPLinear::apply(x,weight,group,column);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("linear_tp",&linear_tp,pybind11::call_guard<pybind11::gil_scoped_release>());
}
