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

include $(top_srcdir)/starpu-notests.mk

AM_CFLAGS += $(FFTWF_CFLAGS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/starpufft/include/ -I$(top_builddir)/include -I$(top_builddir)/src -I$(top_srcdir)/src/ $(STARPU_H_CPPFLAGS)
LIBS += $(top_builddir)/src/@LIBSTARPU_LINK@ $(STARPU_EXPORTED_LIBS)
LIBS += $(FFTW_LIBS) $(FFTWF_LIBS)
LIBS += $(STARPU_CUDA_LDFLAGS) $(STARPU_CUFFT_LDFLAGS)

lib_LTLIBRARIES = libstarpufft-@STARPU_EFFECTIVE_VERSION@.la

EXTRA_DIST =			\
	starpufft-float.h	\
	starpufft-double.h	\
	cudax_kernels.h		\
	starpufftx.c		\
	starpufftx1d.c		\
	starpufftx2d.c		\
	starpufftx3d.c		\
	cuda_kernels.cu		\
	cudaf_kernels.cu	\
	cudax_kernels.cu

libstarpufft_@STARPU_EFFECTIVE_VERSION@_la_SOURCES = starpufft.c starpufftf.c starpufft_common.c
libstarpufft_@STARPU_EFFECTIVE_VERSION@_la_LDFLAGS = $(ldflags) -no-undefined					\
  -version-info $(LIBSTARPUFFT_INTERFACE_CURRENT):$(LIBSTARPUFFT_INTERFACE_REVISION):$(LIBSTARPUFFT_INTERFACE_AGE)

libstarpufft_@STARPU_EFFECTIVE_VERSION@_la_LIBADD =
if STARPU_USE_CUDA
NVCCFLAGS += -Xcompiler -fPIC -Xlinker -fPIC

libstarpufft_@STARPU_EFFECTIVE_VERSION@_la_LIBADD += cudaf_kernels.o

if STARPU_HAVE_CUFFTDOUBLECOMPLEX
libstarpufft_@STARPU_EFFECTIVE_VERSION@_la_LIBADD += cuda_kernels.o
endif

libstarpufft_@STARPU_EFFECTIVE_VERSION@_la_LIBS = $(LIBS) $(STARPU_CUDA_LDFLAGS)
endif
