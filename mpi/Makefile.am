#
# StarPU
# Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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

CC=mpicc

LIBS = $(top_builddir)/src/libstarpu.la @LIBS@
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/mpi/ -I$(top_srcdir)/src/

lib_LTLIBRARIES = libstarpumpi.la

libstarpumpi_la_SOURCES = starpu_mpi.c
libstarpumpi_la_LIBADD = $(top_builddir)/src/libstarpu.la

TESTS = $(check_PROGRAMS)

check_PROGRAMS =

check_PROGRAMS +=					\
	tests/pingpong					\
	tests/ring

tests_pingpong_LDADD =					\
	libstarpumpi.la

tests_pingpong_SOURCES =				\
	tests/pingpong.c

tests_ring_LDADD =					\
	libstarpumpi.la

tests_ring_SOURCES =					\
	tests/ring.c
