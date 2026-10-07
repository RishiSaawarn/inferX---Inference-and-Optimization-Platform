import pybind11
from setuptools import Extension, setup

ext_modules = [
    Extension(
        "inferx_preprocess",
        ["preprocess.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
    ),
]

setup(
    name="inferx_preprocess",
    version="0.1.0",
    ext_modules=ext_modules,
)
