# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
include $(top_srcdir)/make/starpu-loader.mk

BUILT_SOURCES =

CLEANFILES = *.gcno *.gcda *.linkinfo starpu_idle_microsec.log

EXTRA_DIST =					\
	axpy/axpy.jl				\
	axpy/axpy.sh				\
	black_scholes/black_scholes.jl		\
	callback/callback.jl			\
	callback/callback.sh			\
	check_deps/check_deps.jl		\
	check_deps/check_deps.sh		\
	cholesky/cholesky_codelets.jl		\
	cholesky/cholesky_common.jl		\
	cholesky/cholesky_native.jl		\
	cholesky/cholesky_implicit.jl		\
	cholesky/cholesky_tag.jl		\
	cholesky/cholesky.sh			\
	dependency/end_dep.jl			\
	dependency/end_dep.sh			\
	dependency/tag_dep.jl			\
	dependency/tag_dep.sh			\
	dependency/task_dep.sh			\
	dependency/task_dep.jl			\
	gemm/gemm.jl				\
	gemm/gemm_native.jl			\
	gemm/gemm.sh				\
	mandelbrot/mandelbrot_native.jl		\
	mandelbrot/mandelbrot.jl		\
	mandelbrot/mandelbrot.sh		\
	mult/mult_native.jl			\
	mult/mult.jl				\
	mult/perf.sh				\
	mult/mult_starpu.sh			\
	task_insert_color/task_insert_color.jl	\
	task_insert_color/task_insert_color.sh	\
	variable/variable.jl			\
	variable/variable_native.jl		\
	variable/variable.sh			\
	vector_scal/vector_scal.jl		\
	vector_scal/vector_scal.sh

examplebindir = $(libdir)/starpu/julia

examplebin_PROGRAMS =

AM_CFLAGS += $(MAGMA_CFLAGS) $(APP_CFLAGS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/examples/ -I$(top_builddir)/include $(STARPU_H_CPPFLAGS)
AM_LDFLAGS = @STARPU_EXPORT_DYNAMIC@
LIBS += $(top_builddir)/src/@LIBSTARPU_LINK@ ../src/libstarpujulia-@STARPU_EFFECTIVE_VERSION@.la $(STARPU_EXPORTED_LIBS)
LIBS += $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS)
LIBS += -lm

check_PROGRAMS = $(LOADER) $(starpu_julia_EXAMPLES)
SHELL_TESTS	=
STARPU_JULIA_EXAMPLES	=

examplebin_PROGRAMS 	+=	$(STARPU_JULIA_EXAMPLES)

TESTS			=	$(SHELL_TESTS) $(STARPU_JULIA_EXAMPLES)

######################
#      Examples      #
######################

SHELL_TESTS	+=	check_deps/check_deps.sh

STARPU_JULIA_EXAMPLES	+=	mult/mult
mult_mult_SOURCES	=	mult/mult.c mult/cpu_mult.c
SHELL_TESTS		+=	mult/mult_starpu.sh

STARPU_JULIA_EXAMPLES				+=	task_insert_color/task_insert_color
SHELL_TESTS					+=	task_insert_color/task_insert_color.sh

SHELL_TESTS	+=	variable/variable.sh
SHELL_TESTS	+=	vector_scal/vector_scal.sh

STARPU_JULIA_EXAMPLES		+=	mandelbrot/mandelbrot
mandelbrot_mandelbrot_SOURCES	=	mandelbrot/mandelbrot.c mandelbrot/cpu_mandelbrot.c mandelbrot/cpu_mandelbrot.h
SHELL_TESTS			+=	mandelbrot/mandelbrot.sh

STARPU_JULIA_EXAMPLES		+= 	callback/callback
SHELL_TESTS			+=	callback/callback.sh

SHELL_TESTS			+=	dependency/tag_dep.sh
SHELL_TESTS			+=	dependency/task_dep.sh
SHELL_TESTS			+=	dependency/end_dep.sh

if !STARPU_NO_BLAS_LIB
SHELL_TESTS			+=	axpy/axpy.sh
SHELL_TESTS			+=	cholesky/cholesky.sh
SHELL_TESTS			+=	gemm/gemm.sh
endif
