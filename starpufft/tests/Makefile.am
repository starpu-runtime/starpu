# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

CLEANFILES = starpu_idle_microsec.log
examplebindir = $(libdir)/starpu/examples/starpufft

EXTRA_DIST =		\
	testx.c		\
	testx_threads.c	\
	testf_threads.c	\
	test_threads.c

check_PROGRAMS	=	$(STARPU_FFT_EXAMPLES)

AM_CFLAGS += $(APP_CFLAGS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_builddir)/include -I$(top_srcdir)/starpufft/include -I$(top_srcdir)/starpufft/src $(STARPU_H_CPPFLAGS)
AM_LDFLAGS = @STARPU_EXPORT_DYNAMIC@
LIBS += $(top_builddir)/src/@LIBSTARPU_LINK@ ../src/libstarpufft-@STARPU_EFFECTIVE_VERSION@.la $(STARPU_EXPORTED_LIBS)
LIBS += $(STARPU_CUDA_LDFLAGS)

examplebin_PROGRAMS =
examplebin_PROGRAMS +=	\
	testf 		\
	test
STARPU_FFT_EXAMPLES = testf
testf_LDADD = $(FFTWF_LIBS)

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
test_LDADD = $(FFTW_LIBS)

TESTS = $(STARPU_FFT_EXAMPLES)


#check_PROGRAMS += examples/test_threads examples/testf_threads
#examples_test_threads_LDADD = -lfftw3_threads
#examples_testf_threads_LDADD = -lfftw3f_threads
