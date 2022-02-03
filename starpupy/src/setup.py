# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020       Universit'e de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#
from distutils.core import setup, Extension

numpy_dir = ''
if numpy_dir != '':
    numpy_include_dir = [numpy_dir]
else:
    numpy_include_dir = []
starpupy = Extension('starpu.starpupy',
                     include_dirs = ['/home/gonthier/starpu/./include', '/home/gonthier/starpu/include'] + numpy_include_dir,
                     libraries = ['starpu-1.3'],
                     library_dirs = ['/home/gonthier/starpu/src/.libs'],
	             sources = ['starpu/starpu_task_wrapper.c'])

setup(
    name = 'starpupy',
    version = '0.5',
    description = 'Python bindings for StarPU',
    author = 'StarPU team',
    author_email = 'starpu-devel@lists.gforge.inria.fr',
    url = 'https://starpu.gitlabpages.inria.fr/',
    license = 'GPL',
    platforms = 'posix',
    ext_modules = [starpupy],
    packages = ['starpu'],
    )