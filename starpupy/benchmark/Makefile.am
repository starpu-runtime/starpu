# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2021-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

SUBDIRS =

CLEANFILES = *.gcno *.gcda *.linkinfo

EXTRA_DIST	=		\
	tasks_size_overhead.py     \
	tasks_size_overhead.sh     \
	tasks_size_overhead.gp     \
	test_handle_perf.py		\
	test_handle_perf.sh     \
	test_handle_perf_pickle.py		\
	test_handle_perf_pickle.sh	\
	test_handle_bench.py		\
	handle_perf_plot_pickle.py	\
	handle_perf_plot.py

python_sourcesdir = $(libdir)/starpu/python
dist_python_sources_DATA	=	\
	tasks_size_overhead.py     \
	test_handle_perf.py		\
	test_handle_perf_pickle.py

TESTS	=
