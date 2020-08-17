# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
include $(top_srcdir)/starpu.mk

CC=$(MPICC)
CCLD=$(MPICC)
FC=$(MPIFORT)
FCLD=$(MPIFORT)

noinst_PROGRAMS		=

if STARPU_HAVE_WINDOWS
LOADER_BIN		=
else
loader_CPPFLAGS 	= 	$(AM_CFLAGS) $(AM_CPPFLAGS) -I$(top_builddir)/src/
if !STARPU_SIMGRID
LOADER			=	loader
LOADER_BIN		=	$(abs_top_builddir)/mpi/examples/$(LOADER)
noinst_PROGRAMS		+=	loader
endif
loader_SOURCES		=	../../tests/loader.c
endif

if STARPU_SIMGRID
MPI			=	$(abs_top_builddir)/tools/starpu_smpirun -np 4 -platform $(abs_top_srcdir)/tools/perfmodels/cluster.xml -hostfile $(abs_top_srcdir)/tools/perfmodels/hostfile
else
# we always test on 4 processes, the execution time is not that bigger
if STARPU_QUICK_CHECK
MPI			=	$(MPIEXEC) $(MPIEXEC_ARGS) -np 4
else
MPI			=	$(MPIEXEC) $(MPIEXEC_ARGS) -np 4
endif
endif

if STARPU_HAVE_AM111
TESTS_ENVIRONMENT	=	STARPU_WORKERS_NOBIND=1 STARPU_NCPU=3 top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)"
LOG_COMPILER	 	=	$(MPI) $(LOADER_BIN)
else
TESTS_ENVIRONMENT 	=	STARPU_WORKERS_NOBIND=1 STARPU_NCPU=3 top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)" $(MPI) $(LOADER_BIN)
endif

if STARPU_MPI_CHECK
TESTS			=	$(starpu_mpi_EXAMPLES)
endif

check_PROGRAMS = $(LOADER) $(starpu_mpi_EXAMPLES)
starpu_mpi_EXAMPLES =

BUILT_SOURCES =

CLEANFILES = *.gcno *.gcda *.linkinfo starpu_idle_microsec.log *.mod

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
	matrix_decomposition/mpi_cholesky.h 		\
	matrix_decomposition/mpi_cholesky_codelets.h 	\
	matrix_decomposition/mpi_cholesky_kernels.h	\
	matrix_decomposition/mpi_cholesky_models.h 	\
	matrix_decomposition/mpi_decomposition_params.h	\
	matrix_decomposition/mpi_decomposition_matrix.h	\
	user_datatype/my_interface.h			\
	helper.h

examplebindir = $(libdir)/starpu/mpi

examplebin_PROGRAMS =

if STARPU_USE_CUDA
if STARPU_COVERITY
include $(top_srcdir)/starpu-mynvcc.mk
else
NVCCFLAGS += --compiler-options -fno-strict-aliasing  -I$(top_srcdir)/include/ -I$(top_builddir)/include/ $(HWLOC_CFLAGS)

.cu.cubin:
	$(V_nvcc) $(NVCC) -cubin $< -o $@ $(NVCCFLAGS)

.cu.o:
	$(V_nvcc) $(NVCC) $< -c -o $@ $(NVCCFLAGS)
endif
endif

AM_CFLAGS = -Wall $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS) $(FXT_CFLAGS) $(MAGMA_CFLAGS) $(HWLOC_CFLAGS) $(GLOBAL_AM_CFLAGS) -Wno-unused
LIBS = $(top_builddir)/src/@LIBSTARPU_LINK@ ../src/libstarpumpi-@STARPU_EFFECTIVE_VERSION@.la -lm @LIBS@ $(FXT_LIBS) $(MAGMA_LIBS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/examples/ -I$(top_builddir)/include -I$(top_srcdir)/mpi/include
AM_LDFLAGS = $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS) $(FXT_LDFLAGS) $(STARPU_COI_LDFLAGS) $(STARPU_SCIF_LDFLAGS)

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

examplebin_PROGRAMS += 			\
	mpi_lu/plu_example_float	\
	mpi_lu/plu_example_double	\
	mpi_lu/plu_implicit_example_float	\
	mpi_lu/plu_implicit_example_double	\
	mpi_lu/plu_outofcore_example_float	\
	mpi_lu/plu_outofcore_example_double

if !STARPU_SIMGRID
starpu_mpi_EXAMPLES	+=	\
	mpi_lu/plu_implicit_example_float	\
	mpi_lu/plu_implicit_example_double	\
	mpi_lu/plu_outofcore_example_float	\
	mpi_lu/plu_outofcore_example_double
endif

mpi_lu_plu_example_float_LDADD =	\
	$(STARPU_LIBNUMA_LDFLAGS)				\
	$(STARPU_BLAS_LDFLAGS) -lm

mpi_lu_plu_example_float_SOURCES =	\
	mpi_lu/plu_example_float.c	\
	mpi_lu/plu_solve_float.c	\
	mpi_lu/pslu_kernels.c		\
	mpi_lu/pslu.c			\
	../../examples/common/blas.c

mpi_lu_plu_example_double_LDADD =	\
	$(STARPU_LIBNUMA_LDFLAGS)				\
	$(STARPU_BLAS_LDFLAGS) -lm

mpi_lu_plu_example_double_SOURCES =	\
	mpi_lu/plu_example_double.c	\
	mpi_lu/plu_solve_double.c  	\
	mpi_lu/pdlu_kernels.c	    	\
	mpi_lu/pdlu.c		    	\
	../../examples/common/blas.c

mpi_lu_plu_implicit_example_float_LDADD =	\
	$(STARPU_LIBNUMA_LDFLAGS)				\
	$(STARPU_BLAS_LDFLAGS) -lm

mpi_lu_plu_implicit_example_float_SOURCES =	\
	mpi_lu/plu_implicit_example_float.c	\
	mpi_lu/plu_solve_float.c		\
	mpi_lu/pslu_kernels.c			\
	mpi_lu/pslu_implicit.c			\
	../../examples/common/blas.c

mpi_lu_plu_implicit_example_double_LDADD =	\
	$(STARPU_LIBNUMA_LDFLAGS)				\
	$(STARPU_BLAS_LDFLAGS) -lm

mpi_lu_plu_implicit_example_double_SOURCES =	\
	mpi_lu/plu_implicit_example_double.c	\
	mpi_lu/plu_solve_double.c		\
	mpi_lu/pdlu_kernels.c			\
	mpi_lu/pdlu_implicit.c			\
	../../examples/common/blas.c

mpi_lu_plu_outofcore_example_float_LDADD =	\
	$(STARPU_LIBNUMA_LDFLAGS)				\
	$(STARPU_BLAS_LDFLAGS) -lm

mpi_lu_plu_outofcore_example_float_SOURCES =	\
	mpi_lu/plu_outofcore_example_float.c	\
	mpi_lu/plu_solve_float.c		\
	mpi_lu/pslu_kernels.c			\
	mpi_lu/pslu_implicit.c			\
	../../examples/common/blas.c

mpi_lu_plu_outofcore_example_double_LDADD =	\
	$(STARPU_LIBNUMA_LDFLAGS)				\
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
examplebin_PROGRAMS +=		\
	matrix_decomposition/mpi_cholesky			\
	matrix_decomposition/mpi_cholesky_distributed

matrix_decomposition_mpi_cholesky_SOURCES	=		\
	matrix_decomposition/mpi_cholesky.c		\
	matrix_decomposition/mpi_cholesky_models.c		\
	matrix_decomposition/mpi_cholesky_kernels.c	\
	matrix_decomposition/mpi_cholesky_codelets.c	\
	matrix_decomposition/mpi_decomposition_params.c	\
	matrix_decomposition/mpi_decomposition_matrix.c	\
	../../examples/common/blas.c

matrix_decomposition_mpi_cholesky_LDADD =			\
	$(STARPU_BLAS_LDFLAGS) -lm

matrix_decomposition_mpi_cholesky_distributed_SOURCES =	\
	matrix_decomposition/mpi_cholesky_distributed.c	\
	matrix_decomposition/mpi_cholesky_models.c		\
	matrix_decomposition/mpi_cholesky_kernels.c	\
	matrix_decomposition/mpi_cholesky_codelets.c	\
	matrix_decomposition/mpi_decomposition_params.c	\
	matrix_decomposition/mpi_decomposition_matrix.c	\
	../../examples/common/blas.c

matrix_decomposition_mpi_cholesky_distributed_LDADD =	\
	$(STARPU_BLAS_LDFLAGS) -lm

starpu_mpi_EXAMPLES +=				\
	matrix_decomposition/mpi_cholesky			\
	matrix_decomposition/mpi_cholesky_distributed
endif

########################
# MPI Matrix mult example #
########################

examplebin_PROGRAMS +=		\
	matrix_mult/mm

matrix_mult_mm_SOURCES	=		\
	matrix_mult/mm.c

matrix_mult_mm_LDADD =			\
	-lm

if !STARPU_SIMGRID
starpu_mpi_EXAMPLES +=				\
	matrix_mult/mm
endif

##########################################
# Native Fortran MPI Matrix mult example #
##########################################

if STARPU_HAVE_MPIFORT
if !STARPU_SANITIZE
examplebin_PROGRAMS +=		\
	native_fortran/nf_mm	\
	native_fortran/nf_mm_task_build	\
	native_fortran/nf_basic_ring

native_fortran_nf_mm_SOURCES	=			\
	native_fortran/nf_mm_cl.f90			\
	$(top_srcdir)/mpi/include/fstarpu_mpi_mod.f90	\
	$(top_srcdir)/include/fstarpu_mod.f90		\
	native_fortran/nf_mm.f90

native_fortran_nf_mm_LDADD =					\
	-lm

native_fortran_nf_mm_task_build_SOURCES	=			\
	native_fortran/nf_mm_cl.f90			\
	$(top_srcdir)/mpi/include/fstarpu_mpi_mod.f90	\
	$(top_srcdir)/include/fstarpu_mod.f90		\
	native_fortran/nf_mm_task_build.f90

native_fortran_nf_mm_task_build_LDADD =					\
	-lm

native_fortran_nf_basic_ring_SOURCES	=			\
	$(top_srcdir)/mpi/include/fstarpu_mpi_mod.f90	\
	$(top_srcdir)/include/fstarpu_mod.f90		\
	native_fortran/nf_basic_ring.f90

native_fortran_nf_basic_ring_LDADD =					\
	-lm

if !STARPU_SIMGRID
starpu_mpi_EXAMPLES +=				\
	native_fortran/nf_mm			\
	native_fortran/nf_mm_task_build		\
	native_fortran/nf_basic_ring
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

starpu_mpi_EXAMPLES	+=			\
	complex/mpi_complex

#########################
# user_datatype example #
#########################

examplebin_PROGRAMS +=				\
	user_datatype/user_datatype		\
	user_datatype/user_datatype2		\
	user_datatype/user_datatype_early

user_datatype_user_datatype_SOURCES =		\
	user_datatype/user_datatype.c		\
	user_datatype/my_interface.c

user_datatype_user_datatype2_SOURCES =		\
	user_datatype/user_datatype2.c		\
	user_datatype/my_interface.c

user_datatype_user_datatype_early_SOURCES =	\
	user_datatype/user_datatype_early.c	\
	user_datatype/my_interface.c

if !STARPU_SIMGRID
starpu_mpi_EXAMPLES	+=			\
	user_datatype/user_datatype2		\
	user_datatype/user_datatype_early	\
	user_datatype/user_datatype
endif

###################
# comm example #
###################

examplebin_PROGRAMS +=			\
	comm/comm			\
	comm/mix_comm

if !STARPU_SIMGRID
starpu_mpi_EXAMPLES	+=			\
	comm/comm				\
	comm/mix_comm
endif

##################
# filter example #
##################

examplebin_PROGRAMS +=			\
	filters/filter

if !STARPU_SIMGRID
starpu_mpi_EXAMPLES	+=			\
	filters/filter
endif

if STARPU_HAVE_MPIFORT
if !STARPU_SANITIZE
# Native Fortran example
# - list explicit dependences to control proper module files generation
# - the overriding rule fully disables the corresponing default rule, thus
#   the default rule body must be copied entirely
fstarpu_mod.mod: fstarpu_mod.o
fstarpu_mpi_mod.mod: fstarpu_mpi_mod.o
nf_mm_cl.mod: nf_mm_cl.o

fstarpu_mod.o: $(top_srcdir)/include/fstarpu_mod.f90
	$(AM_V_FC)$(FC) $(native_fortran_nf_mm_FCFLAGS) $(FCFLAGS) -c -o $@ '$(top_srcdir)/'include/fstarpu_mod.f90

fstarpu_mpi_mod.o: $(top_srcdir)/mpi/include/fstarpu_mpi_mod.f90 fstarpu_mod.mod
	$(AM_V_FC)$(FC) $(native_fortran_nf_mm_FCFLAGS) $(FCFLAGS) -c -o $@ '$(top_srcdir)/'mpi/include/fstarpu_mpi_mod.f90

nf_mm_cl.o: $(top_srcdir)/mpi/examples/native_fortran/nf_mm_cl.f90 fstarpu_mpi_mod.mod fstarpu_mod.mod
	$(AM_V_FC)$(FC) $(native_fortran_nf_mm_FCFLAGS) $(FCFLAGS) -c -o $@ `test -f 'native_fortran/nf_mm_cl.f90' || echo '$(srcdir)/'`native_fortran/nf_mm_cl.f90

nf_mm.o: $(top_srcdir)/mpi/examples/native_fortran/nf_mm.f90 nf_mm_cl.mod fstarpu_mpi_mod.mod fstarpu_mod.mod
	$(AM_V_FC)$(FC) $(native_fortran_nf_mm_FCFLAGS) $(FCFLAGS) -c -o $@ `test -f 'native_fortran/nf_mm.f90' || echo '$(srcdir)/'`native_fortran/nf_mm.f90

nf_mm_task_build.o: $(top_srcdir)/mpi/examples/native_fortran/nf_mm_task_build.f90 nf_mm_cl.mod fstarpu_mpi_mod.mod fstarpu_mod.mod
	$(AM_V_FC)$(FC) $(native_fortran_nf_mm__task_build_FCFLAGS) $(FCFLAGS) -c -o $@ `test -f 'native_fortran/nf_mm_task_build.f90' || echo '$(srcdir)/'`native_fortran/nf_mm_task_build.f90

nf_basic_ring.o: $(top_srcdir)/mpi/examples/native_fortran/nf_basic_ring.f90 fstarpu_mpi_mod.mod fstarpu_mod.mod
	$(AM_V_FC)$(FC) $(native_fortran_nf_basic_ring_FCFLAGS) $(FCFLAGS) -c -o $@ `test -f 'native_fortran/nf_basic_ring.f90' || echo '$(srcdir)/'`native_fortran/nf_basic_ring.f90
endif
endif
