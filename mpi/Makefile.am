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

SUBDIRS = examples

CC=$(MPICC)

if USE_CUDA
# TODO define NVCCFLAGS
NVCC ?= nvcc

.cu.cubin:
	$(MKDIR_P) `dirname $@`
	$(NVCC) -cubin $< -o $@ --compiler-options -fno-strict-aliasing  $(NVCCFLAGS)

.cu.o:
	$(NVCC) $< -c -o $@ --compiler-options -fno-strict-aliasing  $(NVCCFLAGS) -I$(top_srcdir)/include/
endif


LIBS = $(top_builddir)/src/libstarpu.la @LIBS@
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/mpi/ -I$(top_srcdir)/src/

lib_LTLIBRARIES = libstarpumpi.la

libstarpumpi_la_LIBADD = $(top_builddir)/src/libstarpu.la

noinst_HEADERS =					\
	starpu_mpi_private.h

include_HEADERS =					\
	starpu_mpi.h					\
	starpu_mpi_datatype.h

libstarpumpi_la_SOURCES =				\
	starpu_mpi.c					\
	starpu_mpi_helper.c				\
	starpu_mpi_datatype.c

mpiexamplebindir=$(libdir)/starpu/mpi/

mpiexamplebin_PROGRAMS =				\
	tests/pingpong					\
	tests/mpi_test					\
	tests/mpi_isend					\
	tests/mpi_irecv					\
	tests/mpi_isend_detached			\
	tests/mpi_irecv_detached			\
	tests/mpi_detached_tag				\
	tests/ring					\
	tests/ring_async				\
	tests/block_interface				\
	tests/block_interface_pinned

tests_mpi_isend_LDADD =					\
	libstarpumpi.la

tests_mpi_isend_SOURCES =				\
	tests/mpi_isend.c

tests_mpi_irecv_LDADD =					\
	libstarpumpi.la

tests_mpi_irecv_SOURCES =				\
	tests/mpi_irecv.c

tests_mpi_isend_detached_LDADD =			\
	libstarpumpi.la

tests_mpi_isend_detached_SOURCES =			\
	tests/mpi_isend_detached.c

tests_mpi_irecv_detached_LDADD =			\
	libstarpumpi.la

tests_mpi_irecv_detached_SOURCES =			\
	tests/mpi_irecv_detached.c

tests_mpi_detached_tag_LDADD =				\
	libstarpumpi.la

tests_mpi_detached_tag_SOURCES =			\
	tests/mpi_detached_tag.c

tests_pingpong_LDADD =					\
	libstarpumpi.la

tests_pingpong_SOURCES =				\
	tests/pingpong.c

tests_mpi_test_LDADD =					\
	libstarpumpi.la

tests_mpi_test_SOURCES =				\
	tests/mpi_test.c

tests_ring_LDADD =					\
	libstarpumpi.la

tests_ring_SOURCES =					\
	tests/ring.c

tests_ring_async_LDADD =				\
	libstarpumpi.la

tests_ring_async_SOURCES =				\
	tests/ring_async.c

tests_block_interface_LDADD =				\
	libstarpumpi.la

tests_block_interface_SOURCES =				\
	tests/block_interface.c

tests_block_interface_pinned_LDADD =			\
	libstarpumpi.la

tests_block_interface_pinned_SOURCES =			\
	tests/block_interface_pinned.c

if USE_CUDA
tests_ring_SOURCES += tests/ring_kernel.cu
tests_ring_async_SOURCES += tests/ring_kernel.cu
endif
