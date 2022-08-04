# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

include $(top_srcdir)/starpu-tests.mk
include $(top_srcdir)/starpu-loader.mk

SUBDIRS =

CLEANFILES = *.gcno *.gcda *.linkinfo

if STARPU_USE_MPI_MASTER_SLAVE
TESTS_ENVIRONMENT	+=	LOADER_ARGS="--mpirun"
endif

TESTS	=

if !STARPU_SIMGRID
TESTS	+=	starpu_py.sh

if STARPU_STARPUPY_MULTI_INTERPRETER
TESTS	+=	starpu_py_parallel.sh
endif

if STARPU_STARPUPY_NUMPY
TESTS	+=	starpu_py_parallel.sh
TESTS	+=	starpu_py_np.sh
TESTS	+=	starpu_py_handle.sh
TESTS	+=	starpu_py_partition.sh
endif
endif

EXTRA_DIST	=		\
	starpu_py_parallel.py	\
	starpu_py_parallel.sh	\
	starpu_py.py		\
	starpu_py.sh		\
	starpu_py_handle.py		\
	starpu_py_handle.sh		\
	starpu_py_np.py		\
	starpu_py_np.sh     \
	starpu_py_partition.py		\
	starpu_py_partition.sh

python_sourcesdir = $(libdir)/starpu/python
dist_python_sources_DATA	=	\
	starpu_py_parallel.py	\
	starpu_py.py   \
	starpu_py_handle.py	   \
	starpu_py_np.py   \
	starpu_py_partition.py
