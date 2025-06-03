# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

SUBDIRS	=

AM_CFLAGS += $(APP_CFLAGS)
AM_CXXFLAGS += $(APP_CXXFLAGS)
AM_FFLAGS += $(APP_FFLAGS)
AM_FCFLAGS += $(APP_FCFLAGS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_builddir)/src -I$(top_srcdir)/src/ $(STARPU_H_CPPFLAGS)
AM_LDFLAGS = @STARPU_EXPORT_DYNAMIC@
LIBS += $(top_builddir)/src/@LIBSTARPU_LINK@ $(STARPU_EXPORTED_LIBS)
LIBS += $(HWLOC_LIBS)
LIBS += $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS) $(STARPU_HIP_LDFLAGS)
LIBS += $(STARPU_MAX_FPGA_LDFLAGS)

EXTRA_DIST =					\
	tasks_data_overhead.sh	\
	sync_tasks_data_overhead.sh	\
	async_tasks_data_overhead.sh	\
	tasks_size_overhead.sh	\
	tasks_size_overhead_sched.sh	\
	tasks_size_overhead_scheds.sh	\
	tasks_size_overhead.gp	\
	parallel_dependent_homogeneous_tasks_data.sh	\
	parallel_independent_heterogeneous_tasks_data.sh	\
	parallel_independent_heterogeneous_tasks.sh	\
	parallel_independent_homogeneous_tasks_data.sh	\
	parallel_independent_homogeneous_tasks.sh	\
	bandwidth_scheds.sh		\
	starpu_check.sh		\
	$(MICROBENCHS:=.sh) \
	microbench.sh

CLEANFILES = 					\
	*.gcno *.gcda *.linkinfo core starpu_idle_microsec.log *.mod *.png *.output tasks.rec perfs.rec */perfs.rec */*/perfs.rec perfs2.rec bandwidth-*.dat bandwidth.gp bandwidth.eps bandwidth.svg *.csv *.md *.Rmd *.pdf *.html

clean-local:

BUILT_SOURCES =

examplebindir = $(libdir)/starpu/examples

if STARPU_USE_OPENCL
nobase_STARPU_OPENCL_DATA_DATA =
endif

testbindir = $(libdir)/starpu/tests

#####################################
# What to install and what to check #
#####################################

TESTS = $(SHELL_TESTS) $(myPROGRAMS)

SHELL_TESTS =

myPROGRAMS =

myPROGRAMS +=					\
	display_structures_size	\
	local_pingpong

if !STARPU_SIMGRID
myPROGRAMS +=				\
	async_tasks_overhead	\
	sync_tasks_overhead		\
	tasks_overhead		\
	tasks_size_overhead		\
	prefetch_data_on_node 	\
	redundant_buffer		\
	matrix_as_vector		\
	bandwidth
endif

MICROBENCHS = \
	parallel_independent_homogeneous_tasks      \
	parallel_independent_heterogeneous_tasks    \
	parallel_independent_homogeneous_tasks_data	\
	parallel_independent_heterogeneous_tasks_data \
	parallel_redux_homogeneous_tasks_data \
	parallel_redux_heterogeneous_tasks_data \
	parallel_dependent_homogeneous_tasks_data

examplebin_PROGRAMS = \
	async_tasks_overhead	\
	sync_tasks_overhead		\
	tasks_overhead		\
	tasks_size_overhead		\
	local_pingpong
examplebin_SCRIPTS = \
	tasks_data_overhead.sh \
	sync_tasks_data_overhead.sh \
	async_tasks_data_overhead.sh \
	tasks_size_overhead.gp \
	tasks_size_overhead.sh
if !STARPU_SIMGRID
if !STARPU_USE_MPI_SERVER_CLIENT
examplebin_PROGRAMS += \
	bandwidth
SHELL_TESTS += \
	tasks_data_overhead.sh \
	sync_tasks_data_overhead.sh \
	async_tasks_data_overhead.sh \
	tasks_size_overhead_scheds.sh
endif
endif

if STARPU_HAVE_WINDOWS
check_PROGRAMS	=	$(myPROGRAMS)
else
check_PROGRAMS	=	$(LOADER) $(myPROGRAMS)
endif
noinst_PROGRAMS += $(myPROGRAMS)
noinst_PROGRAMS += $(MICROBENCHS)

if STARPU_SIMGRID
SHELL_TESTS += $(MICROBENCHS:=.sh)
endif

SHELL_TESTS += \
	bandwidth_scheds.sh


