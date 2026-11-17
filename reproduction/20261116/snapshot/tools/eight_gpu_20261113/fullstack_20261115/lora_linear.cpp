#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>

class LoRALinear final : public torch::autograd::Function<LoRALinear> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor x, torch::Tensor weight) {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
                "LoRA candidate supports BF16 only");
    TORCH_CHECK(weight.dim() == 2 && x.size(-1) == weight.size(1), "Shape mismatch");
    ctx->save_for_backward({x, weight});
    return at::linear(x, weight);
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list gradients) {
    const auto saved = ctx->get_saved_variables();
    const auto& x = saved[0]; const auto& weight = saved[1];
    const auto g = gradients[0].reshape({-1, weight.size(0)});
    const auto xx = x.reshape({-1, weight.size(1)});
    torch::Tensor dw, dx;
    // Match captured native BF16 GEMM operands, strides and launch order.
    if (ctx->needs_input_grad(1)) dw = at::mm(g.t(), xx);
    if (ctx->needs_input_grad(0)) dx = at::mm(g, weight).reshape(x.sizes());
    return {dx, dw};
  }
};

torch::Tensor lora_linear(torch::Tensor x, torch::Tensor weight) {
  return LoRALinear::apply(x, weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear", &lora_linear, pybind11::call_guard<pybind11::gil_scoped_release>(),
        "Native BF16 LoRA linear with compact C++ autograd");
}
