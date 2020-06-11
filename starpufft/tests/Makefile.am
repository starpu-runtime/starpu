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

CLEANFILES = starpu_idle_microsec.log
examplebindir = $(libdir)/starpu/examples/starpufft

EXTRA_DIST =		\
	testx.c		\
	testx_threads.c	\
	testf_threads.c	\
	test_threads.c

noinst_PROGRAMS	=
check_PROGRAMS	=	$(STARPU_FFT_EXAMPLES)

if !STARPU_HAVE_WINDOWS
## test loader program
LOADER			=	loader
loader_CPPFLAGS 	=  	$(AM_CFLAGS) $(AM_CPPFLAGS) -I$(top_builddir)/src/
LOADER_BIN		=	$(abs_top_builddir)/starpufft/tests/$(LOADER)
loader_SOURCES		=	../../tests/loader.c
noinst_PROGRAMS		+=	loader

if STARPU_HAVE_AM111
TESTS_ENVIRONMENT	=	top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)"
LOG_COMPILER		=	$(LOADER_BIN)
else
TESTS_ENVIRONMENT	=	top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)" $(LOADER_BIN)
endif

endif

AM_CFLAGS = -Wall $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS) $(FXT_CFLAGS) $(MAGMA_CFLAGS) $(HWLOC_CFLAGS) $(GLOBAL_AM_CFLAGS) -Wno-unused
LIBS = $(top_builddir)/src/@LIBSTARPU_LINK@ @LIBS@ $(FXT_LIBS) $(MAGMA_LIBS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_builddir)/include -I$(top_srcdir)/starpufft/include -I$(top_srcdir)/starpufft/src
AM_LDFLAGS = $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS) $(FXT_LDFLAGS) $(STARPU_COI_LDFLAGS) $(STARPU_SCIF_LDFLAGS)

examplebin_PROGRAMS =
examplebin_PROGRAMS +=	\
	testf 		\
	test
STARPU_FFT_EXAMPLES = testf
testf_LDADD = ../src/libstarpufft-@STARPU_EFFECTIVE_VERSION@.la $(top_builddir)/src/libstarpu-@STARPU_EFFECTIVE_VERSION@.la $(FFTWF_LIBS)

# If we don't have CUDA, we assume that we have fftw available in double
# precision anyway, we just want to make sure that if CUFFT is used, it also
# supports double precision.
if !STARPU_USE_CUDA
STARPU_FFT_EXAMPLES += test
else
if STARPU_HAVE_CUFFTDOUBLECOMPLEX
STARPU_FFT_EXAMPLES += test
endif
endif
test_LDADD = ../src/libstarpufft-@STARPU_EFFECTIVE_VERSION@.la $(top_builddir)/src/libstarpu-@STARPU_EFFECTIVE_VERSION@.la $(FFTW_LIBS)

TESTS = $(STARPU_FFT_EXAMPLES)


#check_PROGRAMS += examples/test_threads examples/testf_threads
#examples_test_threads_LDADD = libstarpufft-@STARPU_EFFECTIVE_VERSION@.la $(top_builddir)/src/libstarpu.la -lfftw3_threads
#examples_testf_threads_LDADD = libstarpufft-@STARPU_EFFECTIVE_VERSION@.la $(top_builddir)/src/libstarpu.la -lfftw3f_threads
