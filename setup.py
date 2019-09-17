from setuptools import setup, find_packages, Extension

import numpy as np
import os

fastmp = Extension('gspen.fastmp.fastmp', sources = ['gspen/fastmp/fastmpmodule.cpp'],
                    include_dirs = [np.get_include()],
                    libraries = ['pthread'],
                    extra_compile_args = ['-W', '-fopenmp', '-O3', '-pedantic', '-std=c++0x', '-DWHICH_FUNC=1'],
                    extra_link_args=['-lgomp', '-lirc'])

if "GUROBI_HOME" in os.environ:
    ilpinf = Extension('gspen.fastmp.ilpinf', sources = ['gspen/fastmp/ilpinf.cpp'],
                        include_dirs = [np.get_include(), os.path.join(os.environ['GUROBI_HOME'],'include')],
                        library_dirs = [os.path.join(os.environ['GUROBI_HOME'], 'lib')],
                        libraries = ['gurobi80', 'gurobi_c++'],
                        extra_compile_args = ['-Wall', '-W', '-fopenmp', '-O3', '-pedantic', '-std=c++0x', '-DWHICH_FUNC=1', '-fPIC'],
                        extra_link_args=['-lgomp', '-lgurobi80', '-lgurobi_c++', '-fPIC', '-lirc'])

    setup(
        name='GSPEN',
        version='0.1',
        author='Colin Graber',
        author_email='cgraber2@illinois.edu',
        description='Implementation of GSPEN model',
        packages=find_packages(),
        python_requires='>=3.5',
        #install_requires
        ext_modules=[fastmp, ilpinf],
    )
else:
    print("No Gurobi detected. ILP and LP inference engines disabled.")
    setup(
        name='GSPEN',
        version='0.1',
        author='Colin Graber',
        author_email='cgraber2@illinois.edu',
        description='Implementation of GSPEN model',
        packages=find_packages(),
        python_requires='>=3.5',
        #install_requires
        ext_modules=[fastmp, ilpinf, fwlpinf],
    )
