#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
at::Tensor swiglu_forward_cuda(at::Tensor x);
at::Tensor swiglu_backward_cuda(at::Tensor x,at::Tensor g);
class Swi final : public torch::autograd::Function<Swi> {
 public:
 static torch::Tensor forward(torch::autograd::AutogradContext* c,torch::Tensor x) {
  TORCH_CHECK(x.scalar_type()==at::kBFloat16 && x.is_contiguous() && x.size(-1)==6400,"BF16 contiguous H=6400 only");
  c->save_for_backward({x}); return swiglu_forward_cuda(x);
 }
 static torch::autograd::variable_list backward(torch::autograd::AutogradContext* c,torch::autograd::variable_list g) {
  return {swiglu_backward_cuda(c->get_saved_variables()[0],g[0].contiguous())};
 }
};
torch::Tensor swiglu_cpp(torch::Tensor x) { return Swi::apply(x); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("swiglu",&swiglu_cpp,pybind11::call_guard<pybind11::gil_scoped_release>());}

