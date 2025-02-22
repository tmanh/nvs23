from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='adaptive_conv_cuda',
    ext_modules=[
        CUDAExtension(
            'adaptive_conv_cuda_impl',
            [
                './adaptive_conv_cuda.cpp',
                './adaptive_conv_kernel.cu',
            ]),
        CppExtension(
            'adaptive_conv_cpp_impl',
            ['./adaptive_conv.cpp'],
            undef_macros=["NDEBUG"]),
    ],
    cmdclass={'build_ext': BuildExtension}
)
