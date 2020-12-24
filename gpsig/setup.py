from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize, build_ext
import numpy as np

exts = [Extension(name='sigKer_fast',
                  sources=['sigKer_fast.pyx']#,
                  #extra_compile_args=['-fopenmp'],
		      #extra_link_args=['-fopenmp']
                  )]

setup(name = 'sigKer_fast',
      ext_modules=cythonize(exts))
