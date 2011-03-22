# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009, 2010-2011  Université de Bordeaux 1
# Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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

AM_CFLAGS = $(HWLOC_CFLAGS) -Wall
LIBS = $(top_builddir)/src/libstarpu.la $(HWLOC_LIBS) @LIBS@
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/examples/ -I$(top_builddir)/include

TESTS	=	$(check_PROGRAMS)

SUBDIRS = stencil

if STARPU_HAVE_FFTW
if STARPU_HAVE_FFTWL
SUBDIRS += starpufft
endif
endif

check_PROGRAMS =

BUILT_SOURCES =

if STARPU_USE_OPENCL
nobase_STARPU_OPENCL_DATA_DATA =
endif

EXTRA_DIST = 					\
	basic_examples/vector_scal_opencl_kernel.cl \
	spmv/spmv_cuda.cu			\
	spmv/spmv_opencl.cl			\
	gordon/null_kernel_gordon.c		\
	mult/xgemm.c				\
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
	filters/fblock_opencl_kernel.cl

CLEANFILES = 					\
	gordon/null_kernel_gordon.spuelf


CLEANFILES += *.gcno *.gcda *.linkinfo

if STARPU_USE_CUDA

NVCCFLAGS += --compiler-options -fno-strict-aliasing  $(HWLOC_CFLAGS) -I$(top_srcdir)/include/ -I$(top_builddir)/include/ -I$(top_srcdir)/examples/  -arch sm_13

.cu.o:
	$(NVCC) $< -c -o $@ $(NVCCFLAGS)

endif

if STARPU_USE_GORDON

SPU_CC ?= spu-gcc
SPU_LD ?= spu-ld

SPULDFLAGS =
SPULIBS = -lblas #-lc -lgloss -lc

.c.spuo:
	$(MKDIR_P) `dirname $@`
	$(SPU_CC) -c -fpic $< -o $@

.spuo.spuelf:
	$(MKDIR_P) `dirname $@`
	$(SPU_LD) $(SPULDFLAGS) $< -o $@ $(SPULIBS)

BUILT_SOURCES +=				\
	gordon/null_kernel_gordon.spuelf

endif

examplebindir = $(libdir)/starpu/examples/

examplebin_PROGRAMS =

noinst_HEADERS = 				\
	cg/cg.h					\
	heat/lu_kernels_model.h			\
	heat/dw_sparse_cg.h			\
	heat/heat.h				\
	heat/dw_factolu.h			\
	lu/xlu.h				\
	lu/xlu_kernels.h			\
	lu/float.h				\
	lu/double.h				\
	pi/pi.h					\
	pi/SobolQRNG/sobol.h			\
	pi/SobolQRNG/sobol_gold.h		\
	pi/SobolQRNG/sobol_gpu.h		\
	pi/SobolQRNG/sobol_primitives.h		\
	cholesky/cholesky.h			\
	common/blas_model.h			\
	common/blas.h				\
	mult/simple.h				\
	mult/double.h				\
	gordon/null.h				\
	fortran/bindings/StarPU_fortran.h	\
	ppm_downscaler/ppm_downscaler.h		\
	ppm_downscaler/yuv_downscaler.h		\
	spmv/matrix_market/mmio.h		\
	spmv/matrix_market/mm_to_bcsr.h		\
	spmv/dw_spmv.h				\
	spmv/dw_block_spmv.h


##################
# Basic examples #
##################

check_PROGRAMS +=				\
	basic_examples/hello_world

examplebin_PROGRAMS +=				\
	basic_examples/hello_world

basic_examples_hello_world_SOURCES =		\
	basic_examples/hello_world.c

check_PROGRAMS +=				\
	basic_examples/vector_scal
examplebin_PROGRAMS +=				\
	basic_examples/vector_scal

basic_examples_vector_scal_SOURCES =		\
	basic_examples/vector_scal.c		\
	basic_examples/vector_scal_cpu.c

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

if STARPU_HAVE_F77_H
check_PROGRAMS +=				\
	basic_examples/vector_scal_fortran
examplebin_PROGRAMS +=				\
	basic_examples/vector_scal_fortran

basic_examples_vector_scal_fortran_SOURCES =	\
	basic_examples/vector_scal_fortran.F	\
	basic_examples/vector_scal_c.c		\
	basic_examples/vector_scal_cpu.c

if STARPU_USE_CUDA
basic_examples_vector_scal_fortran_SOURCES +=	\
	basic_examples/vector_scal_cuda.cu
endif
endif

check_PROGRAMS +=				\
	basic_examples/mult
examplebin_PROGRAMS +=				\
	basic_examples/mult

basic_examples_mult_SOURCES =			\
	basic_examples/mult.c

#################
# block example #
#################

check_PROGRAMS +=				\
	basic_examples/block

examplebin_PROGRAMS +=				\
	basic_examples/block

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

check_PROGRAMS +=				\
	basic_examples/variable

examplebin_PROGRAMS +=				\
	basic_examples/variable

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

check_PROGRAMS	+=	\
	filters/fvector				\
	filters/fblock				\
	filters/fmatrix

examplebin_PROGRAMS +=				\
	filters/fvector				\
	filters/fblock				\
	filters/fmatrix

filters_fvector_SOURCES =			\
	filters/fvector.c

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

filters_fmatrix_SOURCES =			\
	filters/fmatrix.c

###################
# PPM downscaling #
###################

#check_PROGRAMS +=				\
#	ppm_downscaler/ppm_downscaler

examplebin_PROGRAMS +=				\
	ppm_downscaler/ppm_downscaler

ppm_downscaler_ppm_downscaler_SOURCES =		\
	ppm_downscaler/ppm_downscaler.c

#check_PROGRAMS +=				\
#	ppm_downscaler/yuv_downscaler

examplebin_PROGRAMS +=				\
	ppm_downscaler/yuv_downscaler

ppm_downscaler_yuv_downscaler_SOURCES =		\
	ppm_downscaler/yuv_downscaler.c

######
# Pi #
######

check_PROGRAMS +=				\
	pi/pi					\
	pi/pi_redux

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
endif


################
# AXPY example #
################

if !NO_BLAS_LIB

check_PROGRAMS +=				\
	axpy/axpy

examplebin_PROGRAMS +=				\
	axpy/axpy

axpy_axpy_SOURCES =				\
	axpy/axpy.c				\
	common/blas.c

endif

################
# Mult example #
################

if !NO_BLAS_LIB

check_PROGRAMS += 				\
	mult/sgemm 				\
	mult/dgemm

examplebin_PROGRAMS += 				\
	mult/sgemm 				\
	mult/dgemm

mult_sgemm_SOURCES = 				\
	mult/sgemm.c				\
	common/blas.c

mult_dgemm_SOURCES = 				\
	mult/dgemm.c				\
	common/blas.c

endif

####################
# Cholesky example #
####################

if !NO_BLAS_LIB

check_PROGRAMS += 				\
	cholesky/cholesky_tag			\
	cholesky/cholesky_tile_tag		\
	cholesky/cholesky_grain_tag		\
	cholesky/cholesky_implicit

examplebin_PROGRAMS += 				\
	cholesky/cholesky_tag			\
	cholesky/cholesky_tile_tag		\
	cholesky/cholesky_grain_tag		\
	cholesky/cholesky_implicit

cholesky_cholesky_tag_SOURCES =			\
	cholesky/cholesky_tag.c			\
	cholesky/cholesky_models.c		\
	cholesky/cholesky_kernels.c		\
	common/blas.c

cholesky_cholesky_tile_tag_SOURCES =		\
	cholesky/cholesky_tile_tag.c		\
	cholesky/cholesky_models.c		\
	cholesky/cholesky_kernels.c		\
	common/blas.c

cholesky_cholesky_grain_tag_SOURCES =		\
	cholesky/cholesky_grain_tag.c		\
	cholesky/cholesky_models.c		\
	cholesky/cholesky_kernels.c		\
	common/blas.c

cholesky_cholesky_implicit_SOURCES =		\
	cholesky/cholesky_implicit.c		\
	cholesky/cholesky_models.c		\
	cholesky/cholesky_kernels.c		\
	common/blas.c

endif

##############
# LU example #
##############

if !NO_BLAS_LIB

check_PROGRAMS +=				\
	lu/lu_example_float			\
	lu/lu_example_double

examplebin_PROGRAMS += 				\
	lu/lu_example_float			\
	lu/lu_example_double

lu_lu_example_float_SOURCES =			\
	lu/lu_example_float.c			\
	lu/slu.c				\
	lu/slu_pivot.c				\
	lu/slu_kernels.c			\
	common/blas.c

lu_lu_example_double_SOURCES =			\
	lu/lu_example_double.c			\
	lu/dlu.c				\
	lu/dlu_pivot.c				\
	lu/dlu_kernels.c			\
	common/blas.c

check_PROGRAMS += 				\
	lu/lu_implicit_example_float		\
	lu/lu_implicit_example_double

examplebin_PROGRAMS += 				\
	lu/lu_implicit_example_float		\
	lu/lu_implicit_example_double

lu_lu_implicit_example_float_SOURCES =		\
	lu/lu_example_float.c			\
	lu/slu_implicit.c			\
	lu/slu_implicit_pivot.c			\
	lu/slu_kernels.c			\
	common/blas.c

lu_lu_implicit_example_double_SOURCES =		\
	lu/lu_example_double.c			\
	lu/dlu_implicit.c			\
	lu/dlu_implicit_pivot.c			\
	lu/dlu_kernels.c			\
	common/blas.c

endif


################
# Heat example #
################

if !NO_BLAS_LIB

check_PROGRAMS += heat/heat

examplebin_PROGRAMS += heat/heat

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

endif

##############
# CG example #
##############

if !NO_BLAS_LIB

check_PROGRAMS += cg/cg

examplebin_PROGRAMS += cg/cg

cg_cg_SOURCES =					\
	cg/cg.c					\
	cg/cg_kernels.c				\
	common/blas.c

if STARPU_USE_CUDA
cg_cg_SOURCES +=				\
	cg/cg_dot_kernel.cu
endif

endif



################
# Tag examples #
################

check_PROGRAMS +=			\
	tag_example/tag_example			\
	tag_example/tag_example3			\
	tag_example/tag_example2	\
	tag_example/tag_restartable

examplebin_PROGRAMS +=			\
	tag_example/tag_example			\
	tag_example/tag_example3		\
	tag_example/tag_example2	\
	tag_example/tag_restartable

tag_example_tag_example_SOURCES =		\
	tag_example/tag_example.c

tag_example_tag_example2_SOURCES =		\
	tag_example/tag_example2.c

tag_example_tag_example3_SOURCES =		\
	tag_example/tag_example3.c

tag_example_tag_restartable_SOURCES =		\
	tag_example/tag_restartable.c

################
# SpMV example #
################

check_PROGRAMS += 				\
	spmv/dw_spmv

examplebin_PROGRAMS += 				\
	spmv/dw_spmv

spmv_dw_spmv_SOURCES = 				\
	spmv/dw_spmv.c

if STARPU_USE_CUDA
spmv_dw_spmv_SOURCES +=				\
	spmv/spmv_cuda.cu
endif

if ATLAS_BLAS_LIB
check_PROGRAMS += 				\
	spmv/dw_block_spmv

examplebin_PROGRAMS += 				\
	spmv/dw_block_spmv

spmv_dw_block_spmv_SOURCES =			\
	spmv/dw_block_spmv.c			\
	spmv/dw_block_spmv_kernels.c		\
	spmv/matrix_market/mm_to_bcsr.c		\
	spmv/matrix_market/mmio.c
endif

#######################
# Incrementer example #
#######################


check_PROGRAMS +=				\
	incrementer/incrementer

examplebin_PROGRAMS +=				\
	incrementer/incrementer

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

######################
# matVecMult example #
######################

check_PROGRAMS +=				\
	matvecmult/matvecmult

examplebin_PROGRAMS +=				\
	matvecmult/matvecmult

matvecmult_matvecmult_SOURCES =	\
	matvecmult/matvecmult.c

if STARPU_USE_OPENCL
nobase_STARPU_OPENCL_DATA_DATA += \
	matvecmult/matvecmult_kernel.cl
endif

#####################
# profiling example #
#####################

check_PROGRAMS +=				\
	profiling/profiling

examplebin_PROGRAMS +=				\
	profiling/profiling

profiling_profiling_SOURCES =			\
	profiling/profiling.c

#####################
# scheduler example #
#####################

check_PROGRAMS +=				\
	scheduler/dummy_sched

examplebin_PROGRAMS +=				\
	scheduler/dummy_sched

scheduler_dummy_sched_SOURCES =			\
	scheduler/dummy_sched.c

#######################
# dot_product example #
#######################

check_PROGRAMS +=				\
	reductions/dot_product

examplebin_PROGRAMS +=				\
	reductions/dot_product

reductions_dot_product_SOURCES =		\
	reductions/dot_product.c

#####################
# Min/Max reduction #
#####################

check_PROGRAMS +=				\
	reductions/minmax_reduction

examplebin_PROGRAMS +=				\
	reductions/minmax_reduction

reductions_minmax_reduction_SOURCES =		\
	reductions/minmax_reduction.c

##################
# Mandelbrot Set #
##################

#check_PROGRAMS +=				\
#	mandelbrot/mandelbrot
examplebin_PROGRAMS +=				\
	mandelbrot/mandelbrot

mandelbrot_mandelbrot_SOURCES =			\
	mandelbrot/mandelbrot.c

if HAVE_X11
mandelbrot_mandelbrot_CPPFLAGS = $(X_CFLAGS) $(AM_CFLAGS) $(AM_CPPFLAGS)
mandelbrot_mandelbrot_LDADD = $(X_PRE_LIBS) -lX11 $(X_LIBS) $(X_EXTRA_LIBS)
endif
