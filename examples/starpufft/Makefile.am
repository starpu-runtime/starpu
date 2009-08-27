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

libstarpufft_la_SOURCES = starpufft.c
libstarpufft_la_LDFLAGS = $(top_builddir)/src/libstarpu.la $(FFTWF_LIBS)
libstarpufft_la_CFLAGS = $(FFTWF_CFLAGS)

if USE_CUDA
libstarpufft_la_LDFLAGS += -lcufft
endif

check_PROGRAMS = test
test_LDFLAGS = libstarpufft.la $(top_builddir)/src/libstarpu.la $(FFTWF_LIBS)
