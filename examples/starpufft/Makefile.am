#
# StarPU
# Copyright (C) INRIA 2009 (see AUTHORS file)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#
#
AM_CPPFLAGS = -I$(top_srcdir)/include/

lib_LTLIBRARIES = libstarpufft.la

EXTRA_DIST =			\
	starpufft.h		\
	float.h			\
	double.h		\
	cudax_kernels.h		\
	starpufftx.c		\
	starpufftx1d.c		\
	starpufftx2d.c		\
	cuda_kernels.cu		\
	cudaf_kernels.cu	\
	cudax_kernels.cu	\
	testx.c			\
	testx_threads.c	
	

libstarpufft_la_SOURCES = starpufft.c starpufftf.c starpufft-common.c
libstarpufft_la_LIBADD = $(top_builddir)/src/libstarpu.la $(FFTW_LIBS) $(FFTWF_LIBS)
libstarpufft_la_CFLAGS = $(FFTWF_CFLAGS)

if USE_CUDA
# TODO define NVCCFLAGS
NVCC ?= nvcc
NVCCFLAGS += -Xcompiler -fPIC -Xlinker -fPIC

cuda_kernels.o: cuda_kernels.cu
	$(NVCC) $(AM_CPPFLAGS) $< -c -o $@ --compiler-options -fno-strict-aliasing  $(NVCCFLAGS) -I${includedir} -arch sm_13 

cudaf_kernels.o: cudaf_kernels.cu
	$(NVCC) $(AM_CPPFLAGS) $< -c -o $@ --compiler-options -fno-strict-aliasing  $(NVCCFLAGS) -I${includedir}

libstarpufft_la_SOURCES += cuda_kernels.cu cudaf_kernels.cu
am_libstarpufft_la_OBJECTS = cuda_kernels.o cudaf_kernels.o starpufft.lo starpufftf.lo starpufft-common.lo
libstarpufft_la_LIBADD += -lcufft
endif

check_PROGRAMS = test testf
test_LDADD = libstarpufft.la $(top_builddir)/src/libstarpu.la $(FFTW_LIBS)
testf_LDADD = libstarpufft.la $(top_builddir)/src/libstarpu.la $(FFTWF_LIBS)
#test_threads_LDADD = libstarpufft.la $(top_builddir)/src/libstarpu.la -lfftw3_threads
#testf_threads_LDADD = libstarpufft.la $(top_builddir)/src/libstarpu.la -lfftw3f_threads
