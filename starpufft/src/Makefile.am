# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2010-2018                                CNRS
# Copyright (C) 2009-2017                                Université de Bordeaux
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

AM_CFLAGS = $(GLOBAL_AM_CFLAGS) $(HWLOC_CFLAGS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/starpufft/include/ -I$(top_builddir)/include -I$(top_builddir)/src -I$(top_srcdir)/src/ $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS)

lib_LTLIBRARIES = libstarpufft-@STARPU_EFFECTIVE_VERSION@.la

EXTRA_DIST =			\
	starpufft-float.h	\
	starpufft-double.h	\
	cudax_kernels.h		\
	starpufftx.c		\
	starpufftx1d.c		\
	starpufftx2d.c		\
	cuda_kernels.cu		\
	cudaf_kernels.cu	\
	cudax_kernels.cu

libstarpufft_@STARPU_EFFECTIVE_VERSION@_la_SOURCES = starpufft.c starpufftf.c starpufft_common.c
libstarpufft_@STARPU_EFFECTIVE_VERSION@_la_LIBADD = $(top_builddir)/src/libstarpu-@STARPU_EFFECTIVE_VERSION@.la $(FFTW_LIBS) $(FFTWF_LIBS) $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS) $(STARPU_CUFFT_LDFLAGS) $(STARPU_COI_LDFLAGS) $(STARPU_SCIF_LDFLAGS)
libstarpufft_@STARPU_EFFECTIVE_VERSION@_la_CFLAGS = $(FFTWF_CFLAGS)
libstarpufft_@STARPU_EFFECTIVE_VERSION@_la_LDFLAGS = $(ldflags) -no-undefined					\
  -version-info $(LIBSTARPUFFT_INTERFACE_CURRENT):$(LIBSTARPUFFT_INTERFACE_REVISION):$(LIBSTARPUFFT_INTERFACE_AGE)

if STARPU_USE_CUDA
NVCCFLAGS += -Xcompiler -fPIC -Xlinker -fPIC

cudaf_kernels.o: cudaf_kernels.cu
	$(V_nvcc) $(NVCC) $(AM_CPPFLAGS) $< -c -o $@ --compiler-options -fno-strict-aliasing  $(NVCCFLAGS) -I${includedir}

libstarpufft_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += cudaf_kernels.cu

if STARPU_HAVE_CUFFTDOUBLECOMPLEX
cuda_kernels.o: cuda_kernels.cu
	$(NVCC) $(AM_CPPFLAGS) $< -c -o $@ --compiler-options -fno-strict-aliasing  $(NVCCFLAGS) -I${includedir}

libstarpufft_@STARPU_EFFECTIVE_VERSION@_la_SOURCES += cuda_kernels.cu
endif

libstarpufft_@STARPU_EFFECTIVE_VERSION@_la_LIBADD +=  $(STARPU_CUDA_LDFLAGS)
endif

showcheck:
	-cat /dev/null
showsuite:
	-cat /dev/null
