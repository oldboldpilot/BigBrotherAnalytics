"""
Setup script for building C++ feature extractor Python bindings
"""
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import pybind11

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""
    def __str__(self):
        return pybind11.get_include()

ext_modules = [
    Extension(
        'feature_extractor_cpp',
        sources=['bindings/feature_extractor_binding.cpp'],
        include_dirs=[
            get_pybind_include(),
            'src',
        ],
        language='c++',
        extra_compile_args=[
            '-std=c++23',
            '-O3',
            '-march=native',
            '-mavx2',
            '-mfma',
        ],
    ),
]

setup(
    name='feature_extractor_cpp',
    version='1.0.0',
    author='Olumuyiwa Oluwasanmi',
    description='C++23 feature extractor with Python bindings',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.10.0'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    python_requires='>=3.10',
)
