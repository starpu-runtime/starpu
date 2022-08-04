# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
# Copyright (C) 2013       Thibaut Lambert
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

CC=$(MPICC)
CCLD=$(MPICC)
FC=$(MPIFORT)
FCLD=$(MPIFORT)

LAUNCHER		+= 	$(STARPU_MPIEXEC)
LAUNCHER_ENV		+=	$(MPI_RUN_ENV)

include $(top_srcdir)/starpu-loader.mk
if STARPU_SIMGRID
LOADER_BIN = $(LAUNCHER)
endif

if STARPU_MPI_CHECK
TESTS			=	$(starpu_mpi_EXAMPLES)
endif

check_PROGRAMS = $(LOADER) $(starpu_mpi_EXAMPLES)
starpu_mpi_EXAMPLES =

BUILT_SOURCES =

CLEANFILES = *.gcno *.gcda *.linkinfo starpu_idle_microsec.log *.mod native_fortran/fstarpu_mod.f90 native_fortran/fstarpu_mpi_mod.f90

EXTRA_DIST = 				\
	mpi_lu/mpi_lu-float.h		\
	mpi_lu/mpi_lu-double.h		\
	mpi_lu/plu_example.c		\
	mpi_lu/plu_implicit_example.c	\
	mpi_lu/plu_outofcore_example.c	\
	mpi_lu/plu_solve.c		\
	mpi_lu/pxlu.h			\
	mpi_lu/pxlu.c			\
	mpi_lu/pxlu_implicit.c		\
	mpi_lu/pxlu_kernels.h		\
	mpi_lu/pxlu_kernels.c		\
	matrix_mult/mm_2dbc.c 		\
	native_fortran/nf_mm_2dbc.f90 		\
	matrix_decomposition/mpi_cholesky.h 		\
	matrix_decomposition/mpi_cholesky_codelets.h 	\
	matrix_decomposition/mpi_cholesky_kernels.h	\
	matrix_decomposition/mpi_cholesky_models.h 	\
	matrix_decomposition/mpi_decomposition_params.h	\
	matrix_decomposition/mpi_decomposition_matrix.h	\
	user_datatype/my_interface.h	\
	benchs/abstract_sendrecv_bench.h\
	benchs/bench_helper.h		\
	benchs/gemm_helper.h		\
	benchs/burst_helper.h		\
	helper.h			\
	perf.sh

examplebindir = $(libdir)/starpu/mpi

examplebin_PROGRAMS =

AM_CFLAGS += $(MAGMA_CFLAGS) $(APP_CFLAGS)
AM_CXXFLAGS += $(MAGMA_CFLAGS) $(APP_CXXFLAGS)
AM_FFLAGS += $(MAGMA_CFLAGS) $(APP_FFLAGS)
AM_FCFLAGS += $(MAGMA_CFLAGS) $(APP_FCFLAGS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/examples/ -I$(top_builddir)/include -I$(top_srcdir)/mpi/include $(STARPU_H_CPPFLAGS)
AM_LDFLAGS = @STARPU_EXPORT_DYNAMIC@
LIBS += $(top_builddir)/src/@LIBSTARPU_LINK@ ../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la $(STARPU_EXPORTED_LIBS)
LIBS += $(STARPU_CUDA_LDFLAGS)
LIBS += -lm $(MAGMA_LIBS)

###################
# Stencil example #
###################
examplebin_PROGRAMS +=		\
	stencil/stencil5
starpu_mpi_EXAMPLES	+=	\
	stencil/stencil5

if STARPU_USE_MPI_MPI
examplebin_PROGRAMS +=		\
	stencil/stencil5_lb
starpu_mpi_EXAMPLES	+=	\
	stencil/stencil5_lb
endif

##################
# Cache examples #
##################
examplebin_PROGRAMS +=		\
	cache/cache		\
	cache/cache_disable
starpu_mpi_EXAMPLES +=		\
	cache/cache		\
	cache/cache_disable


##################
# MPI LU example #
##################

if !STARPU_NO_BLAS_LIB

examplebin_PROGRAMS += 				\
	mpi_lu/plu_example_float		\
	mpi_lu/plu_example_double		\
	mpi_lu/plu_implicit_example_float	\
	mpi_lu/plu_implicit_example_double	\
	mpi_lu/plu_outofcore_example_float	\
	mpi_lu/plu_outofcore_example_double

if !STARPU_SIMGRID
starpu_mpi_EXAMPLES	+=			\
	mpi_lu/plu_example_float		\
	mpi_lu/plu_example_double		\
	mpi_lu/plu_implicit_example_float	\
	mpi_lu/plu_implicit_example_double	\
	mpi_lu/plu_outofcore_example_float	\
	mpi_lu/plu_outofcore_example_double
endif

mpi_lu_plu_example_float_LDADD =	\
	$(STARPU_LIBNUMA_LDFLAGS)	\
	$(STARPU_BLAS_LDFLAGS) -lm

mpi_lu_plu_example_float_SOURCES =	\
	mpi_lu/plu_example_float.c	\
	mpi_lu/plu_solve_float.c	\
	mpi_lu/pslu_kernels.c		\
	mpi_lu/pslu.c			\
	../../examples/common/blas.c

mpi_lu_plu_example_double_LDADD =	\
	$(STARPU_LIBNUMA_LDFLAGS)	\
	$(STARPU_BLAS_LDFLAGS) -lm

mpi_lu_plu_example_double_SOURCES =	\
	mpi_lu/plu_example_double.c	\
	mpi_lu/plu_solve_double.c  	\
	mpi_lu/pdlu_kernels.c	    	\
	mpi_lu/pdlu.c		    	\
	../../examples/common/blas.c

mpi_lu_plu_implicit_example_float_LDADD =	\
	$(STARPU_LIBNUMA_LDFLAGS)		\
	$(STARPU_BLAS_LDFLAGS) -lm

mpi_lu_plu_implicit_example_float_SOURCES =	\
	mpi_lu/plu_implicit_example_float.c	\
	mpi_lu/plu_solve_float.c		\
	mpi_lu/pslu_kernels.c			\
	mpi_lu/pslu_implicit.c			\
	../../examples/common/blas.c

mpi_lu_plu_implicit_example_double_LDADD =	\
	$(STARPU_LIBNUMA_LDFLAGS)		\
	$(STARPU_BLAS_LDFLAGS) -lm

mpi_lu_plu_implicit_example_double_SOURCES =	\
	mpi_lu/plu_implicit_example_double.c	\
	mpi_lu/plu_solve_double.c		\
	mpi_lu/pdlu_kernels.c			\
	mpi_lu/pdlu_implicit.c			\
	../../examples/common/blas.c

mpi_lu_plu_outofcore_example_float_LDADD =	\
	$(STARPU_LIBNUMA_LDFLAGS)		\
	$(STARPU_BLAS_LDFLAGS) -lm

mpi_lu_plu_outofcore_example_float_SOURCES =	\
	mpi_lu/plu_outofcore_example_float.c	\
	mpi_lu/plu_solve_float.c		\
	mpi_lu/pslu_kernels.c			\
	mpi_lu/pslu_implicit.c			\
	../../examples/common/blas.c

mpi_lu_plu_outofcore_example_double_LDADD =	\
	$(STARPU_LIBNUMA_LDFLAGS)		\
	$(STARPU_BLAS_LDFLAGS) -lm

mpi_lu_plu_outofcore_example_double_SOURCES =	\
	mpi_lu/plu_outofcore_example_double.c	\
	mpi_lu/plu_solve_double.c		\
	mpi_lu/pdlu_kernels.c			\
	mpi_lu/pdlu_implicit.c			\
	../../examples/common/blas.c
endif

########################
# MPI Cholesky example #
########################

if !STARPU_NO_BLAS_LIB
examplebin_PROGRAMS +=					\
	matrix_decomposition/mpi_cholesky		\
	matrix_decomposition/mpi_cholesky_distributed

matrix_decomposition_mpi_cholesky_SOURCES	=	\
	matrix_decomposition/mpi_cholesky.c		\
	matrix_decomposition/mpi_cholesky_models.c	\
	matrix_decomposition/mpi_cholesky_kernels.c	\
	matrix_decomposition/mpi_cholesky_codelets.c	\
	matrix_decomposition/mpi_decomposition_params.c	\
	matrix_decomposition/mpi_decomposition_matrix.c	\
	../../examples/common/blas.c

matrix_decomposition_mpi_cholesky_LDADD =		\
	$(STARPU_BLAS_LDFLAGS) -lm

matrix_decomposition_mpi_cholesky_distributed_SOURCES =	\
	matrix_decomposition/mpi_cholesky_distributed.c	\
	matrix_decomposition/mpi_cholesky_models.c	\
	matrix_decomposition/mpi_cholesky_kernels.c	\
	matrix_decomposition/mpi_cholesky_codelets.c	\
	matrix_decomposition/mpi_decomposition_params.c	\
	matrix_decomposition/mpi_decomposition_matrix.c	\
	../../examples/common/blas.c

matrix_decomposition_mpi_cholesky_distributed_LDADD =	\
	$(STARPU_BLAS_LDFLAGS) -lm

starpu_mpi_EXAMPLES +=					\
	matrix_decomposition/mpi_cholesky		\
	matrix_decomposition/mpi_cholesky_distributed
endif

##############
# CG example #
##############

if !STARPU_SIMGRID
if !STARPU_NO_BLAS_LIB
examplebin_PROGRAMS += cg/cg

cg_cg_SOURCES =				\
	cg/cg.c				\
	../../examples/common/blas.c

cg_cg_LDADD =				\
	$(STARPU_BLAS_LDFLAGS)
endif
endif

############################
# MPI Matrix mult examples #
############################

examplebin_PROGRAMS +=			\
	matrix_mult/mm

matrix_mult_mm_LDADD =			\
	-lm

if !STARPU_SIMGRID
starpu_mpi_EXAMPLES +=			\
	matrix_mult/mm
endif

if !STARPU_NO_BLAS_LIB
examplebin_PROGRAMS +=			\
	matrix_mult/mm_2dbc

matrix_mult_mm_2dbc_SOURCES	=	\
	matrix_mult/mm_2dbc.c		\
	../../examples/common/blas.c

matrix_mult_mm_2dbc_LDADD =		\
	$(STARPU_LIBNUMA_LDFLAGS) 	\
	$(STARPU_BLAS_LDFLAGS) -lm

if !STARPU_SIMGRID
starpu_mpi_EXAMPLES +=			\
	matrix_mult/mm_2dbc
endif
endif

########################
# MPI STARPU_MPI_REDUX #
########################

examplebin_PROGRAMS +=				\
	mpi_redux/mpi_redux     		\
	mpi_redux/mpi_redux_tree

mpi_redux_mpi_redux_LDADD =			\
	-lm
mpi_redux_mpi_redux_tree_LDADD =		\
	-lm
if !STARPU_SIMGRID
starpu_mpi_EXAMPLES +=				\
	mpi_redux/mpi_redux                     \
	mpi_redux/mpi_redux_tree
endif

##########################################
# Native Fortran MPI Matrix mult example #
##########################################

if STARPU_HAVE_MPIFORT
if !STARPU_SANITIZE
examplebin_PROGRAMS +=				\
	native_fortran/nf_mm			\
	native_fortran/nf_mm_task_build		\
	native_fortran/nf_basic_ring

native_fortran_nf_mm_SOURCES	=		\
	native_fortran/nf_mm_cl.f90		\
	native_fortran/fstarpu_mpi_mod.f90	\
	native_fortran/fstarpu_mod.f90		\
	native_fortran/nf_mm.f90

native_fortran_nf_mm_LDADD =			\
	-lm

native_fortran_nf_mm_task_build_SOURCES	=	\
	native_fortran/nf_mm_cl.f90		\
	native_fortran/fstarpu_mpi_mod.f90	\
	native_fortran/fstarpu_mod.f90		\
	native_fortran/nf_mm_task_build.f90

native_fortran_nf_mm_task_build_LDADD =		\
	-lm

native_fortran_nf_basic_ring_SOURCES	=	\
	native_fortran/fstarpu_mpi_mod.f90	\
	native_fortran/fstarpu_mod.f90		\
	native_fortran/nf_basic_ring.f90

native_fortran_nf_basic_ring_LDADD =		\
	-lm

if !STARPU_SIMGRID
starpu_mpi_EXAMPLES +=				\
	native_fortran/nf_mm			\
	native_fortran/nf_mm_task_build		\
	native_fortran/nf_basic_ring
endif

if !STARPU_NO_BLAS_LIB
if STARPU_HAVE_LIBLAPACK
examplebin_PROGRAMS +=		\
	native_fortran/nf_mm_2dbc

native_fortran_nf_mm_2dbc_SOURCES	=	\
	native_fortran/nf_mm_cl_blas.f90	\
	native_fortran/fstarpu_mpi_mod.f90	\
	native_fortran/fstarpu_mod.f90		\
	native_fortran/nf_mm_2dbc.f90

native_fortran_nf_mm_2dbc_LDADD =	\
	$(STARPU_LIBNUMA_LDFLAGS) 	\
	$(STARPU_BLAS_LDFLAGS) -lm -llapack

if !STARPU_SIMGRID
starpu_mpi_EXAMPLES +=			\
	native_fortran/nf_mm_2dbc
endif
endif
endif

endif
endif

########################################
# Native Fortran MPI STARPU_REDUX test #
########################################

if STARPU_HAVE_MPIFORT
if !STARPU_SANITIZE
examplebin_PROGRAMS +=				\
	native_fortran/nf_mpi_redux		\
	native_fortran/nf_mpi_redux_tree

native_fortran_nf_mpi_redux_SOURCES =		\
	native_fortran/fstarpu_mpi_mod.f90	\
	native_fortran/fstarpu_mod.f90		\
	native_fortran/nf_mpi_redux.f90

native_fortran_nf_mpi_redux_LDADD =		\
	-lm

native_fortran_nf_mpi_redux_tree_SOURCES =	\
	native_fortran/fstarpu_mpi_mod.f90	\
	native_fortran/fstarpu_mod.f90		\
	native_fortran/nf_mpi_redux_tree.f90

native_fortran_nf_mpi_redux_tree_LDADD =	\
	-lm

if !STARPU_SIMGRID
starpu_mpi_EXAMPLES +=				\
	native_fortran/nf_mpi_redux		\
	native_fortran/nf_mpi_redux_tree
endif
endif
endif

########################################
# Native Fortran MPI STARPU_REDUX test #
########################################

if STARPU_HAVE_MPIFORT
if !STARPU_SANITIZE
examplebin_PROGRAMS +=				\
	native_fortran/nf_redux_test

native_fortran_nf_redux_test_SOURCES	=	\
	native_fortran/fstarpu_mpi_mod.f90	\
	native_fortran/fstarpu_mod.f90		\
	native_fortran/nf_redux_test.f90

native_fortran_nf_redux_test_LDADD =		\
	-lm

if !STARPU_SIMGRID
starpu_mpi_EXAMPLES +=				\
	native_fortran/nf_redux_test
endif
endif
endif


###################
# complex example #
###################

examplebin_PROGRAMS +=			\
	complex/mpi_complex

complex_mpi_complex_SOURCES =		\
	complex/mpi_complex.c		\
	../../examples/interface/complex_interface.c

starpu_mpi_EXAMPLES	+=		\
	complex/mpi_complex

#########################
# user_datatype example #
#########################

examplebin_PROGRAMS +=				\
	user_datatype/user_datatype		\
	user_datatype/user_datatype2		\
	user_datatype/user_datatype_early	\
	user_datatype/user_datatype_interface

user_datatype_user_datatype_SOURCES =		\
	user_datatype/user_datatype.c		\
	user_datatype/my_interface.c

user_datatype_user_datatype2_SOURCES =		\
	user_datatype/user_datatype2.c		\
	user_datatype/my_interface.c

user_datatype_user_datatype_early_SOURCES =	\
	user_datatype/user_datatype_early.c	\
	user_datatype/my_interface.c

user_datatype_user_datatype_interface_SOURCES =	\
	user_datatype/user_datatype_interface.c	\
	user_datatype/my_interface.c

if !STARPU_SIMGRID
starpu_mpi_EXAMPLES	+=			\
	user_datatype/user_datatype2		\
	user_datatype/user_datatype_early	\
	user_datatype/user_datatype		\
	user_datatype/user_datatype_interface
endif

###################
# comm example #
###################

examplebin_PROGRAMS +=			\
	comm/comm			\
	comm/mix_comm

if !STARPU_SIMGRID
starpu_mpi_EXAMPLES	+=		\
	comm/comm			\
	comm/mix_comm
endif

##################
# filter example #
##################

examplebin_PROGRAMS +=			\
	filters/filter

if !STARPU_SIMGRID
starpu_mpi_EXAMPLES	+=		\
	filters/filter
endif

# Native Fortran example

# - link over source file to build our own object
native_fortran/fstarpu_mod.f90:
	@$(MKDIR_P) $(dir $@)
	$(V_ln) $(LN_S) $(abs_top_srcdir)/include/$(notdir $@) $@
native_fortran/fstarpu_mpi_mod.f90:
	@$(MKDIR_P) $(dir $@)
	$(V_ln) $(LN_S) $(abs_top_srcdir)/mpi/include/$(notdir $@) $@

if STARPU_HAVE_MPIFORT
if !STARPU_SANITIZE
# - express the creation of .mod along .o
fstarpu_mod.mod: native_fortran/fstarpu_mod.o
fstarpu_mpi_mod.mod: native_fortran/fstarpu_mpi_mod.o
nf_mm_cl.mod: native_fortran/nf_mm_cl.o
nf_mm_cl_blas.mod: native_fortran/nf_mm_cl_blas.o

# - list explicit dependences to control proper module files dependencies
native_fortran/fstarpu_mpi_mod.o: fstarpu_mod.mod
native_fortran/nf_mm_cl.o: fstarpu_mod.mod fstarpu_mpi_mod.mod
native_fortran/nf_mm_cl_blas.o: fstarpu_mod.mod fstarpu_mpi_mod.mod
native_fortran/nf_mm.o: nf_mm_cl.mod fstarpu_mpi_mod.mod fstarpu_mod.mod
native_fortran/nf_mm_2dbc.o: nf_mm_cl.mod nf_mm_cl_blas.mod fstarpu_mpi_mod.mod fstarpu_mod.mod
native_fortran/nf_mm_task_build.o: nf_mm_cl.mod fstarpu_mpi_mod.mod fstarpu_mod.mod
native_fortran/nf_basic_ring.o: fstarpu_mpi_mod.mod fstarpu_mod.mod
native_fortran/nf_redux_test.o: fstarpu_mpi_mod.mod fstarpu_mod.mod
native_fortran/nf_mpi_redux.o: fstarpu_mpi_mod.mod fstarpu_mod.mod
native_fortran/nf_mpi_redux_tree.o: fstarpu_mpi_mod.mod fstarpu_mod.mod
endif
endif


##########
# benchs #
##########

examplebin_PROGRAMS +=		\
	benchs/sendrecv_bench	\
	benchs/burst

if !STARPU_USE_MPI_MPI
examplebin_PROGRAMS +=		\
	benchs/sendrecv_parallel_tasks_bench
endif

if !STARPU_NO_BLAS_LIB
examplebin_PROGRAMS +=			\
	benchs/sendrecv_gemm_bench	\
	benchs/burst_gemm
endif

if !STARPU_SIMGRID
starpu_mpi_EXAMPLES	+=	\
	benchs/sendrecv_bench	\
	benchs/burst

if STARPU_MPI_SYNC_CLOCKS
examplebin_PROGRAMS +=		\
	benchs/bcast_bench

starpu_mpi_EXAMPLES	+=	\
	benchs/bcast_bench
endif

if !STARPU_USE_MPI_MPI
starpu_mpi_EXAMPLES	+=	\
	benchs/sendrecv_parallel_tasks_bench
endif

if !STARPU_NO_BLAS_LIB
starpu_mpi_EXAMPLES	+=		\
	benchs/sendrecv_gemm_bench	\
	benchs/burst_gemm
endif
endif

benchs_sendrecv_bench_SOURCES = benchs/sendrecv_bench.c
benchs_sendrecv_bench_SOURCES += benchs/bench_helper.c
benchs_sendrecv_bench_SOURCES += benchs/abstract_sendrecv_bench.c

benchs_bcast_bench_SOURCES = benchs/bcast_bench.c
benchs_bcast_bench_SOURCES += benchs/bench_helper.c
benchs_bcast_bench_LDADD = $(MPI_SYNC_CLOCKS_LDFLAGS)
benchs_bcast_bench_CFLAGS = $(MPI_SYNC_CLOCKS_CFLAGS)

benchs_sendrecv_parallel_tasks_bench_SOURCES = benchs/sendrecv_parallel_tasks_bench.c
benchs_sendrecv_parallel_tasks_bench_SOURCES += benchs/bench_helper.c

benchs_burst_SOURCES = benchs/burst.c
benchs_burst_SOURCES += benchs/burst_helper.c

if !STARPU_NO_BLAS_LIB
benchs_sendrecv_gemm_bench_SOURCES = benchs/sendrecv_gemm_bench.c
benchs_sendrecv_gemm_bench_SOURCES += benchs/bench_helper.c
benchs_sendrecv_gemm_bench_SOURCES += benchs/gemm_helper.c
benchs_sendrecv_gemm_bench_SOURCES += benchs/abstract_sendrecv_bench.c
benchs_sendrecv_gemm_bench_SOURCES += ../../examples/common/blas.c

benchs_sendrecv_gemm_bench_LDADD = $(STARPU_BLAS_LDFLAGS)

benchs_burst_gemm_SOURCES = benchs/burst_gemm.c
benchs_burst_gemm_SOURCES += benchs/gemm_helper.c
benchs_burst_gemm_SOURCES += benchs/burst_helper.c
benchs_burst_gemm_SOURCES += ../../examples/common/blas.c

benchs_burst_gemm_LDADD = $(STARPU_BLAS_LDFLAGS)
endif
