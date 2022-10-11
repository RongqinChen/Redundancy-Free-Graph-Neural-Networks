from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("make_tpf.pyx", compiler_directives={'language_level': "3"}))
