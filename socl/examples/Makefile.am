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

AM_CFLAGS += $(MAGMA_CFLAGS)
AM_CPPFLAGS = $(STARPU_H_CPPFLAGS) -DCL_TARGET_OPENCL_VERSION=120
AM_LDFLAGS = @STARPU_EXPORT_DYNAMIC@
LIBS += $(top_builddir)/socl/src/libsocl-@STARPU_EFFECTIVE_VERSION@.la $(top_builddir)/src/@LIBSTARPU_LINK@ $(STARPU_EXPORTED_LIBS)
LIBS += $(STARPU_OPENCL_LDFLAGS)

SOCL_EXAMPLES	=
if !STARPU_SIMGRID
TESTS		=	$(SOCL_EXAMPLES)
endif

check_PROGRAMS	=	$(SOCL_EXAMPLES)

CLEANFILES = *.gcno *.gcda starpu_idle_microsec.log

examplebindir = $(libdir)/starpu/examples/socl/
examplebin_PROGRAMS =

examplebin_PROGRAMS +=		\
	basic/basic		\
	basicsplit/basicsplit	\
	testmap/testmap		\
	clinfo/clinfo 		\
	matmul/matmul 		\
	mandelbrot/mandelbrot	\
	mansched/mansched


SOCL_EXAMPLES +=		\
	basic/basic		\
	basicsplit/basicsplit	\
	testmap/testmap		\
	clinfo/clinfo		\
	matmul/matmul		 \
	mansched/mansched

matmul_matmul_LDADD = -lm

#mandelbrot_mandelbrot_CPPFLAGS = $(AM_CPPFLAGS)
#if STARPU_HAVE_X11
#mandelbrot_mandelbrot_CPPFLAGS += $(X_CFLAGS)
#mandelbrot_mandelbrot_LDADD = $(X_PRE_LIBS) $(X_LIBS) -lX11 $(X_EXTRA_LIBS)
#endif
