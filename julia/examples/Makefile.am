# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
include $(top_srcdir)/starpu.mk

if STARPU_HAVE_WINDOWS
LOADER_BIN		=
else
loader_CPPFLAGS 	= 	$(AM_CFLAGS) $(AM_CPPFLAGS) -I$(top_builddir)/src/
if !STARPU_SIMGRID
LOADER			=	loader
LOADER_BIN		=	$(abs_top_builddir)/julia/examples/$(LOADER)
endif
loader_SOURCES		=	../../tests/loader.c
endif

if STARPU_HAVE_AM111
TESTS_ENVIRONMENT	=	top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)"
else
TESTS_ENVIRONMENT 	=	top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)" $(LOADER_BIN)
endif

BUILT_SOURCES =

CLEANFILES = *.gcno *.gcda *.linkinfo starpu_idle_microsec.log

EXTRA_DIST =			\
	mult/mult_starpu.sh

examplebindir = $(libdir)/starpu/julia

examplebin_PROGRAMS =

if STARPU_USE_CUDA
if STARPU_COVERITY
include $(top_srcdir)/starpu-mynvcc.mk
else
NVCCFLAGS += --compiler-options -fno-strict-aliasing  -I$(top_srcdir)/include/ -I$(top_builddir)/include/ $(HWLOC_CFLAGS)

.cu.cubin:
	$(V_nvcc) $(NVCC) -cubin $< -o $@ $(NVCCFLAGS)

.cu.o:
	$(V_nvcc) $(NVCC) $< -c -o $@ $(NVCCFLAGS)
endif
endif

AM_CFLAGS = -Wall $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS) $(FXT_CFLAGS) $(MAGMA_CFLAGS) $(HWLOC_CFLAGS) $(GLOBAL_AM_CFLAGS) -Wno-unused
LIBS = $(top_builddir)/src/@LIBSTARPU_LINK@ ../src/libstarpujulia-@STARPU_EFFECTIVE_VERSION@.la -lm @LIBS@ $(FXT_LIBS) $(MAGMA_LIBS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/examples/ -I$(top_builddir)/include
AM_LDFLAGS = $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS) $(FXT_LDFLAGS) $(STARPU_COI_LDFLAGS) $(STARPU_SCIF_LDFLAGS)

check_PROGRAMS = $(LOADER) $(starpu_julia_EXAMPLES)
SHELL_TESTS	=
STARPU_JULIA_EXAMPLES	=

if BUILD_EXAMPLES
examplebin_PROGRAMS 	+=	$(STARPU_JULIA_EXAMPLES)

TESTS			=	$(SHELL_TESTS) $(STARPU_JULIA_EXAMPLES)
endif

######################
#      Examples      #
######################

SHELL_TESTS	+=	check_deps/check_deps.sh

STARPU_JULIA_EXAMPLES	+=	mult/mult
mult_mult_SOURCES	=	mult/mult.c mult/cpu_mult.c
SHELL_TESTS		+=	mult/mult_starpu.sh

STARPU_JULIA_EXAMPLES				+=	task_insert_color/task_insert_color
task_insert_color_task_insert_color_SOURCES	=	task_insert_color/task_insert_color.c
SHELL_TESTS					+=	task_insert_color/task_insert_color.sh

SHELL_TESTS	+=	variable/variable.sh
SHELL_TESTS	+=	vector_scal/vector_scal.sh

STARPU_JULIA_EXAMPLES		+=	mandelbrot/mandelbrot
mandelbrot_mandelbrot_SOURCES	=	mandelbrot/mandelbrot.c mandelbrot/cpu_mandelbrot.c mandelbrot/cpu_mandelbrot.h
SHELL_TESTS			+=	mandelbrot/mandelbrot.sh

STARPU_JULIA_EXAMPLES		+= callback/callback
callback_callback_SOURCES	=	callback/callback.c
SHELL_TESTS			+=	callback/callback.sh

SHELL_TESTS			+=	dependency/tag_dep.sh
SHELL_TESTS			+=	dependency/task_dep.sh
SHELL_TESTS			+=	dependency/end_dep.sh

if !NO_BLAS_LIB
SHELL_TESTS			+= axpy/axpy.sh
endif
