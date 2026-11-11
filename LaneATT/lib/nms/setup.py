import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

if torch.cuda.is_available():
    print("NVIDIA GPU detected. CUDA extension is compiling...")
    ext_modules = [
        CUDAExtension('nms.details', [
            'src/nms.cpp',
            'src/nms_kernel.cu',
        ])
    ]
else:
    print("CPU with integrated graphics detected. Standard C++ extension is compiling...")
    ext_modules = [
        CppExtension('nms.details', [
            'src/nms.cpp',
        ])
    ]

setup(
    name='nms',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)