# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

AM_CFLAGS = $(MAGMA_CFLAGS) $(HWLOC_CFLAGS) -Wall $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS) $(GLOBAL_AM_CFLAGS)
LIBS = $(top_builddir)/socl/src/libsocl-@STARPU_EFFECTIVE_VERSION@.la $(top_builddir)/src/libstarpu-@STARPU_EFFECTIVE_VERSION@.la
AM_LDFLAGS = $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS)


SOCL_EXAMPLES	=
if !STARPU_SIMGRID
TESTS		=	$(SOCL_EXAMPLES)
endif

noinst_PROGRAMS	=
check_PROGRAMS	=	$(SOCL_EXAMPLES)

if !STARPU_HAVE_WINDOWS
## test loader program
LOADER			=	loader
loader_CPPFLAGS 	=	$(AM_CFLAGS) $(AM_CPPFLAGS) -I$(top_builddir)/src/
LOADER_BIN		=	$(abs_top_builddir)/socl/examples/$(LOADER)
loader_SOURCES		=	../../tests/loader.c
noinst_PROGRAMS		+=	loader

if STARPU_HAVE_AM111
TESTS_ENVIRONMENT	=	LD_LIBRARY_PATH="@SOCL_OCL_LIB_OPENCL_DIR@:$(LD_LIBRARY_PATH)" OCL_ICD_VENDORS="$(abs_top_builddir)/socl/vendors/" top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)"
LOG_COMPILER		=	$(LOADER_BIN)
else
TESTS_ENVIRONMENT	=	LD_LIBRARY_PATH="@SOCL_OCL_LIB_OPENCL_DIR@:$(LD_LIBRARY_PATH)" OCL_ICD_VENDORS="$(abs_top_builddir)/socl/vendors/" top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)" $(LOADER_BIN)
endif

endif

CLEANFILES = *.gcno *.gcda starpu_idle_microsec.log

examplebindir = $(libdir)/starpu/examples/socl/
examplebin_PROGRAMS =

examplebin_PROGRAMS +=		\
	basic/basic		\
	testmap/testmap		\
	clinfo/clinfo 		\
	matmul/matmul 		\
	mansched/mansched


SOCL_EXAMPLES +=		\
	basic/basic		\
	testmap/testmap		\
	clinfo/clinfo		\
	matmul/matmul		 \
	mansched/mansched

basic_basic_SOURCES = basic/basic.c
testmap_testmap_SOURCES = testmap/testmap.c
clinfo_clinfo_SOURCES = clinfo/clinfo.c
matmul_matmul_SOURCES = matmul/matmul.c
matmul_matmul_LDADD = -lm
mansched_mansched_SOURCES = mansched/mansched.c

#mandelbrot_mandelbrot_CPPFLAGS = $(AM_CPPFLAGS) $(AM_CFLAGS)
#if STARPU_HAVE_X11
#mandelbrot_mandelbrot_CPPFLAGS += $(X_CFLAGS)
#mandelbrot_mandelbrot_LDADD = $(X_PRE_LIBS) $(X_LIBS) -lX11 $(X_EXTRA_LIBS)
#endif
