# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
include $(top_srcdir)/make/starpu-tests.mk
SUBDIRS =

CLEANFILES = *.gcno *.gcda *.linkinfo

AM_CFLAGS	+= $(DLB_CFLAGS)
AM_CPPFLAGS	= -I$(top_srcdir)/include -I$(top_srcdir)/src -I$(top_builddir)/src -I$(top_builddir)/include
AM_CPPFLAGS	+= -I$(top_srcdir)/starpurm/include -I$(top_srcdir)/starpurm/src -I$(top_builddir)/starpurm/src -I$(top_builddir)/starpurm/include $(STARPU_H_CPPFLAGS)
AM_LDFLAGS = @STARPU_EXPORT_DYNAMIC@
LIBS += $(top_builddir)/starpurm/src/libstarpurm-@STARPU_EFFECTIVE_VERSION@.la $(top_builddir)/src/@LIBSTARPU_LINK@ $(STARPU_EXPORTED_LIBS)
LIBS += $(HWLOC_LIBS) $(DLB_LIBS)

EXTRA_DIST	=				\
	chameleon/dgemm.c			\
	block_test/block_test.c			\
	cuda_vector_scale/vector_scale.c	\
	cuda_vector_scale/vs_cuda_kernel.cu

examplebindir = $(libdir)/starpu/examples/starpurm

examplebin_PROGRAMS 	=	$(STARPU_EXAMPLES)
TESTS			=	$(STARPU_EXAMPLES)

STARPU_EXAMPLES	=	\
	async_spawn	\
	spawn		\
	vector_scale

