# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
# Copyright (C) 2011       Télécom-SudParis
# Copyright (C) 2016       Uppsala University
# Copyright (C) 2017       Erwan Leria
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

AM_CFLAGS = $(MAGMA_CFLAGS) $(HWLOC_CFLAGS) -Wall $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS) $(GLOBAL_AM_CFLAGS) -Wno-unused
AM_CXXFLAGS = $(MAGMA_CFLAGS) $(HWLOC_CFLAGS) -Wall $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS) $(GLOBAL_AM_CXXFLAGS) -Wno-unused
LIBS = $(top_builddir)/src/@LIBSTARPU_LINK@ $(MAGMA_LIBS) $(HWLOC_LIBS) @LIBS@ $(FXT_LIBS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/examples/ -I$(top_builddir)/include
AM_LDFLAGS = @STARPU_EXPORT_DYNAMIC@ $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS) $(STARPU_COI_LDFLAGS) $(STARPU_SCIF_LDFLAGS)

SUBDIRS = stencil

BUILT_SOURCES =

if STARPU_USE_OPENCL
nobase_STARPU_OPENCL_DATA_DATA =
endif

EXTRA_DIST = 					\
	README.txt				\
	axpy/axpy.h				\
	axpy/axpy_opencl_kernel.cl		\
	basic_examples/vector_scal_opencl_kernel.cl \
	basic_examples/multiformat_types.h  \
	basic_examples/multiformat_opencl_kernel.cl  \
	basic_examples/multiformat_conversion_codelets_opencl_kernel.cl \
	common/blas_model.c			\
	spmd/vector_scal_spmd.c			\
	spmv/spmv_cuda.cu			\
	spmv/spmv_opencl.cl			\
	spmv/matrix_market/examples/fidapm05.mtx \
	mult/xgemm.c				\
	mult/sgemm.sh				\
	lu/xlu.c				\
	lu/xlu_pivot.c				\
	lu/xlu_implicit.c			\
	lu/xlu_implicit_pivot.c			\
	lu/xlu_kernels.c			\
	lu/lu_example.c				\
	incrementer/incrementer_kernels_opencl_kernel.cl 	\
	basic_examples/variable_kernels_opencl_kernel.cl	\
	matvecmult/matvecmult_kernel.cl				\
	basic_examples/block_opencl_kernel.cl			\
	filters/fblock_opencl_kernel.cl		\
	filters/custom_mf/conversion_opencl.cl  \
	filters/custom_mf/custom_opencl.cl \
	filters/custom_mf/custom_types.h \
	interface/complex_kernels.cl \
	reductions/dot_product.h	\
	reductions/dot_product_opencl_kernels.cl	\
	scheduler/schedulers.sh				\
	scheduler/schedulers_context.sh			\
	fortran/Makefile				\
	sched_ctx/axpy_partition_gpu.h			\
	sched_ctx/axpy_partition_gpu.cu			\
	heat/heat.sh					\
	cholesky/cholesky.sh				\
	cholesky/cholesky_compiled.c			\
	lu/lu.sh


CLEANFILES = *.gcno *.gcda *.linkinfo *.mod starpu_idle_microsec.log *.mps *.dot *.pl *.png *.output tasks.rec perfs.rec perfs2.rec

if STARPU_USE_CUDA

if STARPU_COVERITY
include $(top_srcdir)/starpu-mynvcc.mk
else
NVCCFLAGS += --compiler-options -fno-strict-aliasing  -I$(top_srcdir)/include/ -I$(top_builddir)/include/ $(HWLOC_CFLAGS)

.cu.o:
	$(V_nvcc) $(NVCC) $< -c -o $@ $(NVCCFLAGS)
endif

endif

if STARPU_HAVE_ICC
.icc.o:
	$(V_icc) $(ICC) -x c $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) $(AM_CFLAGS) $(CFLAGS) $< -c -o $@
endif

examplebindir = $(libdir)/starpu/examples/

examplebin_PROGRAMS =

noinst_HEADERS = 				\
	axpy/axpy.h                             \
	cg/cg.h					\
	heat/lu_kernels_model.h			\
	heat/dw_sparse_cg.h			\
	heat/heat.h				\
	heat/dw_factolu.h			\
	lu/xlu.h				\
	lu/xlu_kernels.h			\
	lu/lu-float.h				\
	lu/lu-double.h				\
	lu/complex_float.h			\
	lu/complex_double.h			\
	lu/blas_complex.h			\
	cholesky/cholesky.h			\
	sched_ctx_utils/sched_ctx_utils.h	\
	common/blas_model.h			\
	common/blas.h				\
	mult/simple.h				\
	mult/double.h				\
	fortran/StarPU_fortran.h		\
	ppm_downscaler/ppm_downscaler.h		\
	ppm_downscaler/yuv_downscaler.h		\
	spmv/matrix_market/mmio.h		\
	spmv/matrix_market/mm_to_bcsr.h		\
	spmv/spmv.h				\
	spmv/dw_block_spmv.h                    \
	basic_examples/multiformat_types.h      \
	filters/custom_mf/custom_interface.h    \
	filters/custom_mf/custom_types.h	\
	interface/complex_interface.h		\
	interface/complex_codelet.h		\
	pi/pi.h					\
	pi/SobolQRNG/sobol.h			\
	pi/SobolQRNG/sobol_gold.h		\
	pi/SobolQRNG/sobol_gpu.h		\
	pi/SobolQRNG/sobol_primitives.h         \
	reductions/dot_product.h                \
	basic_examples/vector_scal_cpu_template.h \
	sched_ctx/axpy_partition_gpu.h

#####################################
# What to install and what to check #
#####################################

examplebin_PROGRAMS 	+=	$(STARPU_EXAMPLES)

TESTS			=	$(SHELL_TESTS) $(STARPU_EXAMPLES)

SHELL_TESTS =

if !STARPU_USE_MPI_MASTER_SLAVE
SHELL_TESTS			+=	scheduler/schedulers.sh
SHELL_TESTS			+=	scheduler/schedulers_context.sh
if !STARPU_NO_BLAS_LIB
SHELL_TESTS			+=	mult/sgemm.sh
endif
endif

check_PROGRAMS		=	$(STARPU_EXAMPLES)
noinst_PROGRAMS		=

if !STARPU_HAVE_WINDOWS
## test loader program
if !STARPU_CROSS_COMPILING
LOADER			=	loader
loader_CPPFLAGS 	=	$(AM_CFLAGS) $(AM_CPPFLAGS) -I$(top_builddir)/src/
LOADER_BIN		=	$(abs_top_builddir)/examples/$(LOADER)
loader_SOURCES		=	../tests/loader.c
noinst_PROGRAMS		+=	loader
else
LOADER			=
LOADER_BIN		=	$(top_builddir)/examples/loader-cross.sh
endif

if STARPU_USE_MPI_MASTER_SLAVE
LOADER_BIN2		= $(MPI_LAUNCHER) $(LOADER_BIN)
else
LOADER_BIN2		= $(LOADER_BIN)
endif

if STARPU_HAVE_AM111
TESTS_ENVIRONMENT	=	$(MPI_RUN_ARGS) top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)"
LOG_COMPILER		=	$(LOADER_BIN2)
else
TESTS_ENVIRONMENT	=	$(MPI_RUN_ARGS) top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)" $(LOADER_BIN2)
endif

endif

# STARPU_EXAMPLES list all applications which have to be compiled and checked
# Applications which should only be compiled are added directly in examplebin_PROGRAMS
# see for instance mandelbrot/mandelbrot

STARPU_EXAMPLES =

STARPU_EXAMPLES +=				\
	sched_ctx/prio				\
	scheduler/dummy_sched			\
	scheduler/dummy_modular_sched		\
	worker_collections/worker_list_example  \
	api/bcsr_data_interface			\
	api/block_data_interface		\
	api/coo_data_interface			\
	api/csr_data_interface			\
	api/matrix_data_interface		\
	api/multiformat_data_interface		\
	api/variable_data_interface		\
	api/vector_data_interface		\
	api/void_data_interface

if !STARPU_SIMGRID
STARPU_EXAMPLES +=				\
	basic_examples/hello_world		\
	basic_examples/topology			\
	basic_examples/vector_scal		\
	basic_examples/mult			\
	basic_examples/block			\
	basic_examples/variable			\
	basic_examples/multiformat              \
	basic_examples/dynamic_handles          \
	basic_examples/task_insert_color	\
	mlr/mlr					\
	cpp/incrementer_cpp			\
	cpp/add_vectors				\
	cpp/add_vectors_interface		\
	filters/fread				\
	filters/fvector				\
	filters/fblock				\
	filters/fmatrix				\
	filters/fmultiple_manual		\
	filters/fmultiple_submit		\
	filters/fmultiple_submit_readonly	\
	filters/fmultiple_submit_implicit	\
	filters/frecursive			\
	filters/fplan_notautomatic		\
	tag_example/tag_example			\
	tag_example/tag_example2		\
	tag_example/tag_example3		\
	tag_example/tag_example4		\
	tag_example/tag_restartable		\
	spmd/vector_scal_spmd			\
	spmv/spmv				\
	callback/callback			\
	callback/prologue			\
	incrementer/incrementer			\
	binary/binary				\
	interface/complex			\
	matvecmult/matvecmult			\
	profiling/profiling			\
	scheduler/heteroprio_test		\
	sched_ctx/sched_ctx			\
	sched_ctx/sched_ctx_empty		\
	sched_ctx/sched_ctx_delete		\
	sched_ctx/two_cpu_contexts		\
	sched_ctx/dummy_sched_with_ctx		\
	worker_collections/worker_tree_example  \
	reductions/dot_product			\
	reductions/minmax_reduction		\
	dependency/task_end_dep			\
	dependency/task_end_dep_add		\
	dependency/sequential_consistency

endif

if !STARPU_SIMGRID
STARPU_EXAMPLES +=				\
	scheduler/dummy_sched

if STARPU_HAVE_CXX11
STARPU_EXAMPLES +=	\
	cpp/add_vectors_cpp11
endif

if STARPU_HAVE_F77
if STARPU_HAVE_F77_H
STARPU_EXAMPLES +=				\
	fortran/hello
endif

STARPU_EXAMPLES +=				\
	basic_examples/vector_scal_fortran
endif

if STARPU_HAVE_FC
if !STARPU_SANITIZE
STARPU_EXAMPLES +=				\
	fortran90/f90_example			\
	native_fortran/nf_vector		\
	native_fortran/nf_matrix		\
	native_fortran/nf_example		\
	native_fortran/nf_dynbuf		\
	native_fortran/nf_varbuf		\
	native_fortran/nf_sched_ctx		\
	native_fortran/nf_partition
endif
endif
endif

if !STARPU_NO_BLAS_LIB
STARPU_EXAMPLES +=				\
	mult/sgemm 				\
	mult/dgemm				\
	lu/lu_example_float			\
	lu/lu_example_double			\
	lu/lu_implicit_example_float		\
	lu/lu_implicit_example_double		\
	cholesky/cholesky_tag			\
	cholesky/cholesky_tile_tag		\
	cholesky/cholesky_implicit		\
	cholesky/cholesky_compil

if !STARPU_SIMGRID
STARPU_EXAMPLES +=				\
	axpy/axpy				\
	cholesky/cholesky_grain_tag		\
	heat/heat				\
	cg/cg					\
	pipeline/pipeline

if !STARPU_USE_MPI_MASTER_SLAVE
SHELL_TESTS += \
	heat/heat.sh				\
	lu/lu.sh

endif

endif

if STARPU_SIMGRID
if !STARPU_QUICK_CHECK
SHELL_TESTS += \
	 cholesky/cholesky.sh
endif
endif
endif

if !STARPU_SIMGRID

if STARPU_MKL_BLAS_LIB
STARPU_EXAMPLES +=				\
	lu/lu_example_complex_float		\
	lu/lu_example_complex_double		\
	lu/lu_implicit_example_complex_float	\
	lu/lu_implicit_example_complex_double
endif

if STARPU_HAVE_CBLAS_H
STARPU_EXAMPLES +=				\
	spmv/dw_block_spmv
endif

if !STARPU_SIMGRID
if STARPU_HAVE_F77
if STARPU_HAVE_F77_H
STARPU_EXAMPLES +=				\
	fortran/hello
endif
STARPU_EXAMPLES +=				\
	basic_examples/vector_scal_fortran
endif
endif

if STARPU_HAVE_OPENMP
STARPU_EXAMPLES +=				\
	openmp/vector_scal_omp			\
	sched_ctx/sched_ctx_without_sched_policy\
	sched_ctx/nested_sched_ctxs		\
	sched_ctx/sched_ctx_without_sched_policy_awake\
	sched_ctx/parallel_tasks_reuse_handle	\
	sched_ctx/parallel_code

if STARPU_HAVE_HWLOC
if STARPU_HWLOC_HAVE_TOPOLOGY_DUP
STARPU_EXAMPLES +=				\
	sched_ctx/parallel_tasks_with_cluster_api
endif
endif
endif

endif !STARPU_SIMGRID

if STARPU_USE_CUDA
STARPU_EXAMPLES +=				\
	sched_ctx/gpu_partition
sched_ctx_gpu_partition_SOURCES =		\
	sched_ctx/gpu_partition.c		\
	sched_ctx/axpy_partition_gpu.cu
endif

##################
# Basic examples #
##################

basic_examples_vector_scal_SOURCES =		\
	basic_examples/vector_scal.c		\
	basic_examples/vector_scal_cpu.c

if STARPU_HAVE_ICC
if STARPU_CROSS_COMPILING
basic_examples_vector_scal_LINK = $(LIBTOOL) $(AM_V_lt) --tag=CC $(AM_LIBTOOLFLAGS) $(LIBTOOLFLAGS) --mode=link $(CCLD) $(basic_examples_vector_scal_CFLAGS) $(CFLAGS) $(AM_LDFLAGS) $(LDFLAGS) -o $@
else
basic_examples_vector_scal_SOURCES +=		\
	basic_examples/vector_scal_cpu_icc.icc
basic_examples_vector_scal_LINK = $(LIBTOOL) $(AM_V_lt) --tag=CC $(AM_LIBTOOLFLAGS) $(LIBTOOLFLAGS) --mode=link $(ICC) $(basic_examples_vector_scal_CFLAGS) $(CFLAGS) $(AM_LDFLAGS) $(LDFLAGS) -o $@
endif
else
basic_examples_vector_scal_LINK = $(LIBTOOL) $(AM_V_lt) --tag=CC $(AM_LIBTOOLFLAGS) $(LIBTOOLFLAGS) --mode=link $(CCLD) $(basic_examples_vector_scal_CFLAGS) $(CFLAGS) $(AM_LDFLAGS) $(LDFLAGS) -o $@
endif

if STARPU_USE_CUDA
basic_examples_vector_scal_SOURCES +=		\
	basic_examples/vector_scal_cuda.cu
endif

if STARPU_USE_OPENCL
basic_examples_vector_scal_SOURCES +=		\
	basic_examples/vector_scal_opencl.c
nobase_STARPU_OPENCL_DATA_DATA += \
	basic_examples/vector_scal_opencl_kernel.cl
endif

if STARPU_HAVE_F77
basic_examples_vector_scal_fortran_SOURCES =	\
	basic_examples/vector_scal_fortran.F	\
	basic_examples/vector_scal_c.c		\
	basic_examples/vector_scal_cpu.c

if STARPU_USE_CUDA
basic_examples_vector_scal_fortran_SOURCES +=	\
	basic_examples/vector_scal_cuda.cu
basic_examples_vector_scal_fortran_LDADD =	\
	$(STARPU_CUDA_FORTRAN_LDFLAGS)
endif

if STARPU_HAVE_F77_H
fortran_hello_SOURCES	=		\
	fortran/hello_c.c		\
	fortran/hello.F			\
	fortran/StarPU_fortran.h
endif
endif

if STARPU_HAVE_FC
fortran90_f90_example_SOURCES =	\
	fortran90/mod_types.f90		\
	$(top_srcdir)/include/starpu_mod.f90	\
	fortran90/mod_interface.f90	\
	fortran90/mod_compute.f90	\
	fortran90/marshalling.c		\
	fortran90/f90_example.f90

native_fortran_nf_vector_SOURCES =	\
	native_fortran/nf_codelets.f90		\
	$(top_srcdir)/include/fstarpu_mod.f90	\
	native_fortran/nf_vector.f90

native_fortran_nf_matrix_SOURCES =	\
	native_fortran/nf_codelets.f90		\
	$(top_srcdir)/include/fstarpu_mod.f90	\
	native_fortran/nf_matrix.f90

native_fortran_nf_example_SOURCES =	\
	native_fortran/nf_types.f90		\
	native_fortran/nf_compute.f90		\
	$(top_srcdir)/include/fstarpu_mod.f90	\
	native_fortran/nf_example.f90

native_fortran_nf_dynbuf_SOURCES =	\
	native_fortran/nf_dynbuf_cl.f90		\
	$(top_srcdir)/include/fstarpu_mod.f90	\
	native_fortran/nf_dynbuf.f90

native_fortran_nf_varbuf_SOURCES =	\
	native_fortran/nf_varbuf_cl.f90		\
	$(top_srcdir)/include/fstarpu_mod.f90	\
	native_fortran/nf_varbuf.f90

native_fortran_nf_sched_ctx_SOURCES =	\
	native_fortran/nf_sched_ctx_cl.f90		\
	$(top_srcdir)/include/fstarpu_mod.f90	\
	native_fortran/nf_sched_ctx.f90

native_fortran_nf_partition_SOURCES =	\
	native_fortran/nf_partition_cl.f90		\
	$(top_srcdir)/include/fstarpu_mod.f90	\
	native_fortran/nf_partition.f90
endif

#######################
# Multiformat example #
#######################
basic_examples_multiformat_SOURCES =                                    \
	basic_examples/multiformat.c                                    \
	basic_examples/multiformat_conversion_codelets.c

if STARPU_USE_CUDA
basic_examples_multiformat_SOURCES +=                                    \
	basic_examples/multiformat_cuda.cu                               \
	basic_examples/multiformat_conversion_codelets_cuda.cu
endif

if STARPU_USE_OPENCL
basic_examples_multiformat_SOURCES +=                                    \
	basic_examples/multiformat_opencl.c                              \
	basic_examples/multiformat_conversion_codelets_opencl.c

nobase_STARPU_OPENCL_DATA_DATA +=                                        \
	basic_examples/multiformat_opencl_kernel.cl                      \
	basic_examples/multiformat_conversion_codelets_opencl_kernel.cl
endif

#################
# block example #
#################

basic_examples_block_SOURCES =			\
	basic_examples/block.c			\
	basic_examples/block_cpu.c

if STARPU_USE_CUDA
basic_examples_block_SOURCES +=			\
	basic_examples/block_cuda.cu
endif

if STARPU_USE_OPENCL
basic_examples_block_SOURCES +=			\
	basic_examples/block_opencl.c
nobase_STARPU_OPENCL_DATA_DATA += 		\
	basic_examples/block_opencl_kernel.cl
endif

####################
# Variable example #
####################

basic_examples_variable_SOURCES =		\
	basic_examples/variable.c		\
	basic_examples/variable_kernels_cpu.c

if STARPU_USE_CUDA
basic_examples_variable_SOURCES +=		\
	basic_examples/variable_kernels.cu
endif
if STARPU_USE_OPENCL
basic_examples_variable_SOURCES +=		\
	basic_examples/variable_kernels_opencl.c
nobase_STARPU_OPENCL_DATA_DATA += 		\
	basic_examples/variable_kernels_opencl_kernel.cl
endif

###########
# Filters #
###########

filters_fblock_SOURCES =			\
	filters/fblock.c			\
	filters/fblock_cpu.c

if STARPU_USE_CUDA
filters_fblock_SOURCES +=			\
	filters/fblock_cuda.cu
endif
if STARPU_USE_OPENCL
filters_fblock_SOURCES +=			\
	filters/fblock_opencl.c
nobase_STARPU_OPENCL_DATA_DATA += \
	filters/fblock_opencl_kernel.cl
endif

filters_fmultiple_manual_SOURCES =		\
	filters/fmultiple_manual.c

if STARPU_USE_CUDA
filters_fmultiple_manual_SOURCES +=		\
	filters/fmultiple_cuda.cu
endif

filters_fmultiple_submit_SOURCES =		\
	filters/fmultiple_submit.c

if STARPU_USE_CUDA
filters_fmultiple_submit_SOURCES +=		\
	filters/fmultiple_cuda.cu
endif

filters_fmultiple_submit_readonly_SOURCES =	\
	filters/fmultiple_submit_readonly.c

if STARPU_USE_CUDA
filters_fmultiple_submit_readonly_SOURCES +=	\
	filters/fmultiple_cuda.cu
endif

filters_fmultiple_submit_implicit_SOURCES =		\
	filters/fmultiple_submit_implicit.c

if STARPU_USE_CUDA
filters_fmultiple_submit_implicit_SOURCES +=		\
	filters/fmultiple_cuda.cu
endif

examplebin_PROGRAMS +=				\
	filters/shadow				\
	filters/shadow2d			\
	filters/shadow3d

#############################
# Custom multiformat filter #
#############################

#TODO: see why the application is failing
#lt-custom_mf_filter: .../src/datawizard/malloc.c:784: starpu_free_on_node: Assertion `chunk != _starpu_chunk_list_end(chunks[dst_node])' failed.

examplebin_PROGRAMS +=				\
	filters/custom_mf/custom_mf_filter

filters_custom_mf_custom_mf_filter_SOURCES=\
	filters/custom_mf/custom_mf_filter.c \
	filters/custom_mf/custom_interface.c   \
	filters/custom_mf/custom_conversion_codelets.c

if STARPU_USE_CUDA
filters_custom_mf_custom_mf_filter_SOURCES += \
	filters/custom_mf/conversion.cu \
	filters/custom_mf/cuda.cu
endif

if STARPU_USE_OPENCL
filters_custom_mf_custom_mf_filter_SOURCES += \
	filters/custom_mf/conversion_opencl.c \
	filters/custom_mf/custom_opencl.c
nobase_STARPU_OPENCL_DATA_DATA += \
	filters/custom_mf/conversion_opencl.cl \
	filters/custom_mf/custom_opencl.cl
endif

################
# AXPY example #
################

if !STARPU_NO_BLAS_LIB
axpy_axpy_SOURCES =				\
	axpy/axpy.c				\
	common/blas.c
if STARPU_USE_OPENCL
axpy_axpy_SOURCES += \
	axpy/axpy_opencl.c
nobase_STARPU_OPENCL_DATA_DATA +=		\
	axpy/axpy_opencl_kernel.cl
endif

axpy_axpy_LDADD =				\
	$(STARPU_BLAS_LDFLAGS)
endif

################
# Mult example #
################

if !STARPU_NO_BLAS_LIB

mult_sgemm_SOURCES = 				\
	mult/sgemm.c				\
	common/blas.c

mult_sgemm_LDADD =				\
	$(STARPU_BLAS_LDFLAGS)

mult_dgemm_SOURCES = 				\
	mult/dgemm.c				\
	common/blas.c

mult_dgemm_LDADD =				\
	$(STARPU_BLAS_LDFLAGS)

endif

####################
# Cholesky example #
####################

if !STARPU_NO_BLAS_LIB

cholesky_cholesky_tag_SOURCES =			\
	cholesky/cholesky_tag.c			\
	cholesky/cholesky_models.c		\
	cholesky/cholesky_kernels.c		\
	common/blas.c

cholesky_cholesky_tag_LDADD =			\
	$(STARPU_BLAS_LDFLAGS)

cholesky_cholesky_tile_tag_SOURCES =		\
	cholesky/cholesky_tile_tag.c		\
	cholesky/cholesky_models.c		\
	cholesky/cholesky_kernels.c		\
	common/blas.c

cholesky_cholesky_tile_tag_LDADD =		\
	$(STARPU_BLAS_LDFLAGS)

cholesky_cholesky_grain_tag_SOURCES =		\
	cholesky/cholesky_grain_tag.c		\
	cholesky/cholesky_models.c		\
	cholesky/cholesky_kernels.c		\
	common/blas.c

cholesky_cholesky_grain_tag_LDADD =		\
	$(STARPU_BLAS_LDFLAGS)

cholesky_cholesky_implicit_SOURCES =		\
	cholesky/cholesky_implicit.c		\
	cholesky/cholesky_models.c		\
	cholesky/cholesky_kernels.c		\
	sched_ctx_utils/sched_ctx_utils.c	\
	common/blas.c

cholesky_cholesky_implicit_LDADD =		\
	$(STARPU_BLAS_LDFLAGS)

cholesky_cholesky_compil_SOURCES =		\
	cholesky/cholesky_compil.c		\
	cholesky/cholesky_models.c		\
	cholesky/cholesky_kernels.c		\
	sched_ctx_utils/sched_ctx_utils.c	\
	common/blas.c

cholesky_cholesky_compil_LDADD =		\
	$(STARPU_BLAS_LDFLAGS)

endif

##############
# LU example #
##############

if !STARPU_NO_BLAS_LIB

lu_lu_example_float_SOURCES =			\
	lu/lu_example_float.c			\
	lu/slu.c				\
	lu/slu_pivot.c				\
	lu/slu_kernels.c			\
	common/blas.c

lu_lu_example_float_LDADD =			\
	$(STARPU_BLAS_LDFLAGS)

lu_lu_example_double_SOURCES =			\
	lu/lu_example_double.c			\
	lu/dlu.c				\
	lu/dlu_pivot.c				\
	lu/dlu_kernels.c			\
	common/blas.c

lu_lu_example_double_LDADD =			\
	$(STARPU_BLAS_LDFLAGS)

lu_lu_implicit_example_float_SOURCES =		\
	lu/lu_example_float.c			\
	lu/slu_implicit.c			\
	lu/slu_implicit_pivot.c			\
	lu/slu_kernels.c			\
	common/blas.c

lu_lu_implicit_example_float_LDADD =		\
	$(STARPU_BLAS_LDFLAGS)

lu_lu_implicit_example_double_SOURCES =		\
	lu/lu_example_double.c			\
	lu/dlu_implicit.c			\
	lu/dlu_implicit_pivot.c			\
	lu/dlu_kernels.c			\
	common/blas.c

lu_lu_implicit_example_double_LDADD =		\
	$(STARPU_BLAS_LDFLAGS)

if STARPU_MKL_BLAS_LIB
lu_lu_example_complex_float_SOURCES =		\
	lu/lu_example_complex_float.c		\
	lu/clu.c				\
	lu/clu_pivot.c				\
	lu/clu_kernels.c			\
	lu/blas_complex.c			\
	common/blas.c

lu_lu_example_complex_float_LDADD =		\
	$(STARPU_BLAS_LDFLAGS)

lu_lu_implicit_example_complex_float_SOURCES =	\
	lu/lu_example_complex_float.c		\
	lu/clu_implicit.c			\
	lu/clu_implicit_pivot.c			\
	lu/clu_kernels.c			\
	lu/blas_complex.c			\
	common/blas.c

lu_lu_implicit_example_complex_float_LDADD =	\
	$(STARPU_BLAS_LDFLAGS)

lu_lu_example_complex_double_SOURCES =		\
	lu/lu_example_complex_double.c		\
	lu/zlu.c				\
	lu/zlu_pivot.c				\
	lu/zlu_kernels.c			\
	lu/blas_complex.c			\
	common/blas.c

lu_lu_example_complex_double_LDADD =		\
	$(STARPU_BLAS_LDFLAGS)

lu_lu_implicit_example_complex_double_SOURCES =	\
	lu/lu_example_complex_double.c		\
	lu/zlu_implicit.c			\
	lu/zlu_implicit_pivot.c			\
	lu/zlu_kernels.c			\
	lu/blas_complex.c			\
	common/blas.c

lu_lu_implicit_example_complex_double_LDADD =	\
	$(STARPU_BLAS_LDFLAGS)

endif
endif


################
# Heat example #
################

if !STARPU_NO_BLAS_LIB

heat_heat_SOURCES =				\
	heat/heat.c				\
	heat/dw_factolu.c			\
	heat/dw_factolu_tag.c			\
	heat/dw_factolu_grain.c			\
	heat/dw_sparse_cg.c			\
	heat/heat_display.c			\
	heat/lu_kernels_model.c			\
	heat/dw_sparse_cg_kernels.c		\
	heat/dw_factolu_kernels.c		\
	common/blas.c

heat_heat_LDADD =				\
	$(STARPU_OPENGL_RENDER_LDFLAGS)		\
	$(STARPU_BLAS_LDFLAGS)

endif

##############
# CG example #
##############

if !STARPU_NO_BLAS_LIB

cg_cg_SOURCES =					\
	cg/cg.c					\
	cg/cg_kernels.c				\
	common/blas.c

cg_cg_LDADD =					\
	$(STARPU_BLAS_LDFLAGS)

endif

################
# SPMD example #
################

spmd_vector_scal_spmd_SOURCES =			\
	spmd/vector_scal_spmd.c

################
# SpMV example #
################

spmv_spmv_SOURCES = 				\
	spmv/spmv.c				\
	spmv/spmv_kernels.c

if STARPU_USE_CUDA
spmv_spmv_SOURCES +=				\
	spmv/spmv_cuda.cu
endif

spmv_dw_block_spmv_SOURCES =			\
	spmv/dw_block_spmv.c			\
	spmv/dw_block_spmv_kernels.c		\
	spmv/matrix_market/mm_to_bcsr.c		\
	spmv/matrix_market/mmio.c

spmv_dw_block_spmv_LDADD =			\
	$(STARPU_BLAS_LDFLAGS)

###########################
# C++ Incrementer example #
###########################

cpp_incrementer_cpp_SOURCES	=	\
	cpp/incrementer_cpp.cpp

if STARPU_USE_CUDA
cpp_incrementer_cpp_SOURCES +=	\
	incrementer/incrementer_kernels.cu
endif
if STARPU_USE_OPENCL
cpp_incrementer_cpp_SOURCES += \
	incrementer/incrementer_kernels_opencl.c
endif

###########################
# C++ Add vectors example #
###########################

cpp_add_vectors_SOURCES	=	\
	cpp/add_vectors.cpp

cpp_add_vectors_interface_SOURCES	=	\
	cpp/add_vectors_interface.cpp

if STARPU_HAVE_CXX11
cpp_add_vectors_cpp11_SOURCES	=	\
	cpp/add_vectors_cpp11.cpp
endif

#######################
# Incrementer example #
#######################

incrementer_incrementer_SOURCES =	\
	incrementer/incrementer.c
if STARPU_USE_CUDA
incrementer_incrementer_SOURCES +=	\
	incrementer/incrementer_kernels.cu
endif
if STARPU_USE_OPENCL
incrementer_incrementer_SOURCES +=	\
	incrementer/incrementer_kernels_opencl.c
nobase_STARPU_OPENCL_DATA_DATA += \
	incrementer/incrementer_kernels_opencl_kernel.cl
endif

##################
# Binary example #
##################

binary_binary_SOURCES =	\
	binary/binary.c
if STARPU_USE_OPENCL
binary_binary_SOURCES +=	\
	incrementer/incrementer_kernels_opencl.c
endif

#####################
# interface example #
#####################

interface_complex_SOURCES	=	\
	interface/complex.c		\
	interface/complex_interface.c	\
	interface/complex_filters.c
if STARPU_USE_CUDA
interface_complex_SOURCES	+=	\
	interface/complex_kernels.cu
endif

if STARPU_USE_OPENCL
interface_complex_SOURCES +=\
	interface/complex_kernels_opencl.c
nobase_STARPU_OPENCL_DATA_DATA += \
	interface/complex_kernels.cl
endif

######################
# matVecMult example #
######################

if STARPU_USE_OPENCL
nobase_STARPU_OPENCL_DATA_DATA += \
	matvecmult/matvecmult_kernel.cl
endif

#######################
# dot_product example #
#######################

reductions_dot_product_SOURCES =		\
	reductions/dot_product.c
if STARPU_USE_CUDA
reductions_dot_product_SOURCES +=		\
	reductions/dot_product_kernels.cu
endif
if STARPU_USE_OPENCL
nobase_STARPU_OPENCL_DATA_DATA += \
	reductions/dot_product_opencl_kernels.cl
endif

##################
# Mandelbrot Set #
##################

examplebin_PROGRAMS +=				\
	mandelbrot/mandelbrot

mandelbrot_mandelbrot_CPPFLAGS = $(AM_CFLAGS) $(AM_CPPFLAGS)
if STARPU_HAVE_X11
mandelbrot_mandelbrot_CPPFLAGS += $(X_CFLAGS)
mandelbrot_mandelbrot_LDADD = $(X_PRE_LIBS) $(X_LIBS) -lX11 $(X_EXTRA_LIBS)
endif

####################
# Image downscaler #
####################

examplebin_PROGRAMS +=				\
	ppm_downscaler/ppm_downscaler		\
	ppm_downscaler/yuv_downscaler

######
# Pi #
######

if !STARPU_HAVE_WINDOWS

examplebin_PROGRAMS +=				\
	pi/pi					\
	pi/pi_redux

pi_pi_SOURCES =					\
	pi/pi.c					\
	pi/SobolQRNG/sobol_gold.c		\
	pi/SobolQRNG/sobol_primitives.c

if STARPU_USE_CUDA
pi_pi_SOURCES +=				\
	pi/pi_kernel.cu				\
	pi/SobolQRNG/sobol_gpu.cu
endif

pi_pi_redux_SOURCES =				\
	pi/pi_redux.c

if STARPU_USE_CUDA
pi_pi_redux_SOURCES +=				\
	pi/pi_redux_kernel.cu
pi_pi_redux_LDADD =				\
	$(STARPU_CURAND_LDFLAGS)
endif
endif

###########################
# OpenGL interoperability #
###########################

if STARPU_HAVE_OPENGL
examplebin_PROGRAMS +=				\
	gl_interop/gl_interop			\
	gl_interop/gl_interop_idle

gl_interop_gl_interop_SOURCES =			\
	gl_interop/gl_interop.c

gl_interop_gl_interop_LDADD =			\
	$(STARPU_OPENGL_RENDER_LDFLAGS)

gl_interop_gl_interop_idle_SOURCES =		\
	gl_interop/gl_interop_idle.c

gl_interop_gl_interop_idle_LDADD =		\
	$(STARPU_OPENGL_RENDER_LDFLAGS)
endif

####################
# pipeline example #
####################

if !STARPU_NO_BLAS_LIB
pipeline_pipeline_SOURCES	=	\
	pipeline/pipeline.c		\
	common/blas.c

pipeline_pipeline_LDADD =		\
	$(STARPU_BLAS_LDFLAGS)
endif

##################
# openmp example #
##################

if STARPU_HAVE_OPENMP
openmp_vector_scal_omp_CFLAGS = \
	$(AM_CFLAGS) -fopenmp

sched_ctx_parallel_code_CFLAGS = \
	$(AM_CFLAGS) -fopenmp

sched_ctx_sched_ctx_without_sched_policy_CFLAGS = \
	$(AM_CFLAGS) -fopenmp

sched_ctx_nested_sched_ctxs_CFLAGS = \
	$(AM_CFLAGS) -fopenmp

sched_ctx_parallel_tasks_reuse_handle_CFLAGS = \
	$(AM_CFLAGS) -fopenmp

endif

if STARPU_HAVE_FC
# Fortran90 example
# - list explicit dependences to control proper module files generation
# - the overriding rule fully disables the corresponing default rule, thus
#   the default rule body must be copied entirely
mod_types.mod: mod_types.o
mod_compute.mod: mod_compute.o
mod_interface.mod: mod_interface.o
starpu_mod.mod: starpu_mod.o

starpu_mod.o: $(top_srcdir)/include/starpu_mod.f90
	$(AM_V_FC)$(FC) $(fortran90_f90_example_FCFLAGS) $(FCFLAGS) -c -o $@ '$(top_srcdir)/'include/starpu_mod.f90

mod_compute.o: $(top_srcdir)/examples/fortran90/mod_compute.f90 mod_types.mod mod_interface.mod starpu_mod.mod
	$(AM_V_FC)$(FC) $(fortran90_f90_example_FCFLAGS) $(FCFLAGS) -c -o $@ `test -f 'fortran90/mod_compute.f90' || echo '$(srcdir)/'`fortran90/mod_compute.f90

f90_example.o: $(top_srcdir)/examples/fortran90/f90_example.f90 $(top_srcdir)/examples/fortran90/marshalling.c mod_types.mod mod_interface.mod mod_compute.mod starpu_mod.mod
	$(AM_V_FC)$(FC) $(fortran90_f90_example_FCFLAGS) $(FCFLAGS) -c -o $@ `test -f 'fortran90/f90_example.f90' || echo '$(srcdir)/'`fortran90/f90_example.f90

# Native Fortran example
# - list explicit dependences to control proper module files generation
# - the overriding rule fully disables the corresponing default rule, thus
#   the default rule body must be copied entirely
nf_types.mod: nf_types.o
nf_compute.mod: nf_compute.o
fstarpu_mod.mod: fstarpu_mod.o
nf_codelets.mod: nf_codelets.o
nf_dynbuf_cl.mod: nf_dynbuf_cl.o
nf_varbuf_cl.mod: nf_varbuf_cl.o
nf_sched_ctx_cl.mod: nf_sched_ctx_cl.o
nf_partition_cl.mod: nf_partition_cl.o

fstarpu_mod.o: $(top_srcdir)/include/fstarpu_mod.f90
	$(AM_V_FC)$(FC) $(native_fortran_nf_vector_FCFLAGS) $(FCFLAGS) -c -o $@ '$(top_srcdir)/'include/fstarpu_mod.f90

nf_codelets.o: $(top_srcdir)/examples/native_fortran/nf_codelets.f90 fstarpu_mod.mod
	$(AM_V_FC)$(FC) $(native_fortran_nf_vector_FCFLAGS) $(FCFLAGS) -c -o $@ `test -f 'native_fortran/nf_codelets.f90' || echo '$(srcdir)/'`native_fortran/nf_codelets.f90

nf_vector.o: $(top_srcdir)/examples/native_fortran/nf_vector.f90 nf_codelets.mod fstarpu_mod.mod
	$(AM_V_FC)$(FC) $(native_fortran_nf_vector_FCFLAGS) $(FCFLAGS) -c -o $@ `test -f 'native_fortran/nf_vector.f90' || echo '$(srcdir)/'`native_fortran/nf_vector.f90

nf_matrix.o: $(top_srcdir)/examples/native_fortran/nf_matrix.f90 nf_codelets.mod fstarpu_mod.mod
	$(AM_V_FC)$(FC) $(native_fortran_nf_matrix_FCFLAGS) $(FCFLAGS) -c -o $@ `test -f 'native_fortran/nf_matrix.f90' || echo '$(srcdir)/'`native_fortran/nf_matrix.f90

nf_compute.o: $(top_srcdir)/examples/native_fortran/nf_compute.f90 nf_types.mod fstarpu_mod.mod
	$(AM_V_FC)$(FC) $(native_fortran_nf_example_FCFLAGS) $(FCFLAGS) -c -o $@ `test -f 'native_fortran/nf_compute.f90' || echo '$(srcdir)/'`native_fortran/nf_compute.f90

nf_example.o: $(top_srcdir)/examples/native_fortran/nf_example.f90 nf_types.mod nf_compute.mod fstarpu_mod.mod
	$(AM_V_FC)$(FC) $(native_fortran_nf_example_FCFLAGS) $(FCFLAGS) -c -o $@ `test -f 'native_fortran/nf_example.f90' || echo '$(srcdir)/'`native_fortran/nf_example.f90

nf_dynbuf_cl.o: $(top_srcdir)/examples/native_fortran/nf_dynbuf_cl.f90 nf_types.mod fstarpu_mod.mod
	$(AM_V_FC)$(FC) $(native_fortran_nf_dynbuf_FCFLAGS) $(FCFLAGS) -c -o $@ `test -f 'native_fortran/nf_dynbuf_cl.f90' || echo '$(srcdir)/'`native_fortran/nf_dynbuf_cl.f90

nf_dynbuf.o: $(top_srcdir)/examples/native_fortran/nf_dynbuf.f90 nf_types.mod nf_dynbuf_cl.mod fstarpu_mod.mod
	$(AM_V_FC)$(FC) $(native_fortran_nf_dynbuf_FCFLAGS) $(FCFLAGS) -c -o $@ `test -f 'native_fortran/nf_dynbuf.f90' || echo '$(srcdir)/'`native_fortran/nf_dynbuf.f90

nf_varbuf_cl.o: $(top_srcdir)/examples/native_fortran/nf_varbuf_cl.f90 nf_types.mod fstarpu_mod.mod
	$(AM_V_FC)$(FC) $(native_fortran_nf_varbuf_FCFLAGS) $(FCFLAGS) -c -o $@ `test -f 'native_fortran/nf_varbuf_cl.f90' || echo '$(srcdir)/'`native_fortran/nf_varbuf_cl.f90

nf_varbuf.o: $(top_srcdir)/examples/native_fortran/nf_varbuf.f90 nf_types.mod nf_varbuf_cl.mod fstarpu_mod.mod
	$(AM_V_FC)$(FC) $(native_fortran_nf_varbuf_FCFLAGS) $(FCFLAGS) -c -o $@ `test -f 'native_fortran/nf_varbuf.f90' || echo '$(srcdir)/'`native_fortran/nf_varbuf.f90

nf_sched_ctx_cl.o: $(top_srcdir)/examples/native_fortran/nf_sched_ctx_cl.f90 nf_types.mod fstarpu_mod.mod
	$(AM_V_FC)$(FC) $(native_fortran_nf_sched_ctx_FCFLAGS) $(FCFLAGS) -c -o $@ `test -f 'native_fortran/nf_sched_ctx_cl.f90' || echo '$(srcdir)/'`native_fortran/nf_sched_ctx_cl.f90

nf_sched_ctx.o: $(top_srcdir)/examples/native_fortran/nf_sched_ctx.f90 nf_types.mod nf_sched_ctx_cl.mod fstarpu_mod.mod
	$(AM_V_FC)$(FC) $(native_fortran_nf_sched_ctx_FCFLAGS) $(FCFLAGS) -c -o $@ `test -f 'native_fortran/nf_sched_ctx.f90' || echo '$(srcdir)/'`native_fortran/nf_sched_ctx.f90

nf_partition_cl.o: $(top_srcdir)/examples/native_fortran/nf_partition_cl.f90 nf_types.mod fstarpu_mod.mod
	$(AM_V_FC)$(FC) $(native_fortran_nf_partition_FCFLAGS) $(FCFLAGS) -c -o $@ `test -f 'native_fortran/nf_partition_cl.f90' || echo '$(srcdir)/'`native_fortran/nf_partition_cl.f90

nf_partition.o: $(top_srcdir)/examples/native_fortran/nf_partition.f90 nf_types.mod nf_partition_cl.mod fstarpu_mod.mod
	$(AM_V_FC)$(FC) $(native_fortran_nf_partition_FCFLAGS) $(FCFLAGS) -c -o $@ `test -f 'native_fortran/nf_partition.f90' || echo '$(srcdir)/'`native_fortran/nf_partition.f90

endif
