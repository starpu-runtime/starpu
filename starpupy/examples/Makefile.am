# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

include $(top_srcdir)/make/starpu-tests.mk
include $(top_srcdir)/make/starpu-loader.mk

SUBDIRS =

CLEANFILES = *.gcno *.gcda *.linkinfo \
	starpu_py.concurrent.py         \
	starpu_py_handle.concurrent.py  \
	starpu_py_np.concurrent.py      \
	starpu_py_partition.concurrent.py \
	starpu_py_numpy.concurrent.py

if STARPU_USE_MPI_MASTER_SLAVE
TESTS_ENVIRONMENT	+=	LOADER_ARGS="--mpirun"
endif

%.concurrent.py: %.py
	sed -e 's/async //g' -e 's/\<await(/await_fut(/g' -e 's/asyncio.run(main())/main()/g' < $< > $@

all-local: \
	starpu_py.concurrent.py         \
	starpu_py_handle.concurrent.py  \
	starpu_py_np.concurrent.py      \
	starpu_py_partition.concurrent.py \
	starpu_py_numpy.concurrent.py

TESTS	=

if !STARPU_SIMGRID
TESTS	+=	starpu_py.sh
TESTS	+=	starpu_py.concurrent.sh

TESTS	+=	starpu_py_parallel.sh
TESTS	+=	starpu_py_handle.sh
TESTS	+=	starpu_py_handle.concurrent.sh

if STARPU_STARPUPY_NUMPY
TESTS	+=	starpu_py_numpy.sh
TESTS	+=	starpu_py_numpy.concurrent.sh
TESTS	+=	starpu_py_np.sh
TESTS	+=	starpu_py_np.concurrent.sh
TESTS	+=	starpu_py_partition.sh
TESTS	+=	starpu_py_partition.concurrent.sh
endif
endif

EXTRA_DIST	=		\
	starpu_py_parallel.py	\
	starpu_py_parallel.sh	\
	starpu_py.py		\
	starpu_py.sh		\
	starpu_py.concurrent.sh		\
	starpu_py_handle.py	\
	starpu_py_handle.sh	\
	starpu_py_handle.concurrent.sh	\
	starpu_py_np.py		\
	starpu_py_np.sh     	\
	starpu_py_np.concurrent.sh     	\
	starpu_py_partition.py	\
	starpu_py_partition.sh	\
	starpu_py_partition.concurrent.sh	\
	starpu_py_numpy.py	\
	starpu_py_numpy.sh	\
	starpu_py_numpy.concurrent.sh

python_sourcesdir = $(libdir)/starpu/python
dist_python_sources_DATA	=	\
	starpu_py_parallel.py		\
	starpu_py.py   			\
	starpu_py_handle.py	  	\
	starpu_py_np.py   		\
	starpu_py_partition.py		\
	starpu_py_numpy.py
