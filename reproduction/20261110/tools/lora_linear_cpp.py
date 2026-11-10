from pathlib import Path
from torch.utils.cpp_extension import load

BASE=Path('/private/yp/jiuge-training')
_extension=None

def extension():
    global _extension
    if _extension is None:
        build=BASE/'cache/torch-extensions/lora-linear-cpp-v2';build.mkdir(parents=True,exist_ok=True)
        _extension=load(name='jiuge_lora_linear_v2',sources=[str(BASE/'tools/lora_linear_cpp.cpp')],extra_cflags=['-O2'],build_directory=str(build),with_cuda=False,verbose=True)
    return _extension

def linear(x,weight):return extension().linear(x,weight)

def install():
    extension()
    import lora_linear_patch
    lora_linear_patch.linear=linear
    lora_linear_patch.install()
