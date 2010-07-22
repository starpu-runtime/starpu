#
# StarPU
# Copyright (C) INRIA 2008-2010 (see AUTHORS file)
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

LIBS = $(top_builddir)/src/libstarpu.la @LIBS@
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/examples/ -I$(top_builddir)/include

TESTS = $(check_PROGRAMS)

if STARPU_HAVE_FFTW
if STARPU_HAVE_FFTWL
SUBDIRS = starpufft
endif
endif

check_PROGRAMS =

BUILT_SOURCES =

EXTRA_DIST = 					\
	basic_examples/vector_scal_opencl_codelet.cl \
	cuda/incrementer_cuda.cu		\
	spmv/spmv_cuda.cu			\
	gordon/null_kernel_gordon.c		\
	mult/xgemm.c				\
	mult/xgemm_kernels.c			\
	mult/gordon/func_sgemm_ibm.c		\
	lu/xlu.c				\
	lu/xlu_pivot.c				\
	lu/xlu_implicit.c			\
	lu/xlu_implicit_pivot.c			\
	lu/xlu_kernels.c			\
	lu/lu_example.c				\
	incrementer/incrementer_kernels_opencl_codelet.cl 	\
	variable/variable_kernels_opencl_codelet.cl		\
	matvecmult/matvecmult_kernel.cl				\
	block/block_opencl_kernel.cl					\
	filters/fblock_opencl_codelet.cl

CLEANFILES = 					\
	gordon/null_kernel_gordon.spuelf	\
	mult/gordon/func_sgemm_ibm.spuelf


CLEANFILES += *.gcno *.gcda *.linkinfo

if STARPU_USE_CUDA

.cu.cubin:
	$(MKDIR_P) `dirname $@`
	$(NVCC) -cubin $< -o $@ --compiler-options -fno-strict-aliasing  $(NVCCFLAGS)

.cu.o:
	$(NVCC) $< -c -o $@ --compiler-options -fno-strict-aliasing  $(NVCCFLAGS) -I$(top_srcdir)/include/ -I$(top_builddir)/include/


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
	gordon/null_kernel_gordon.spuelf	\
	mult/gordon/func_sgemm_ibm.spuelf

endif

examplebindir = $(libdir)/starpu/examples/

examplebin_PROGRAMS =

noinst_HEADERS = 				\
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
	cholesky/dw_cholesky_models.h		\
	cholesky/dw_cholesky.h			\
	common/blas_model.h			\
	common/blas.h				\
	mult/dw_mult.h				\
	mult/gordon/func_sgemm_ibm.h		\
	cuda/incrementer_cuda.h			\
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

examplebin_PROGRAMS +=				\
	basic_examples/hello_world

basic_examples_hello_world_SOURCES =		\
	basic_examples/hello_world.c

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
endif

nobase_STARPU_OPENCL_DATA_DATA = \
	basic_examples/vector_scal_opencl_codelet.cl

examplebin_PROGRAMS +=				\
	basic_examples/mult

basic_examples_mult_SOURCES =			\
	basic_examples/mult.c

###########
# Filters #
###########

examplebin_PROGRAMS +=				\
	filters/fvector				\
	filters/fblock				\
	filters/fmatrix

filters_fvector_SOURCES =			\
	filters/fvector.c

filters_fblock_SOURCES =			\
	filters/fblock.c

if STARPU_USE_CUDA
filters_fblock_SOURCES +=			\
	filters/fblock_cuda.cu
endif
if STARPU_USE_OPENCL
filters_fblock_SOURCES +=			\
	filters/fblock_opencl.c
endif

nobase_STARPU_OPENCL_DATA_DATA += \
	filters/fblock_opencl_codelet.cl

filters_fmatrix_SOURCES =			\
	filters/fmatrix.c

###################
# PPM downscaling #
###################

examplebin_PROGRAMS +=				\
	ppm_downscaler/ppm_downscaler

ppm_downscaler_ppm_downscaler_SOURCES =		\
	ppm_downscaler/ppm_downscaler.c

examplebin_PROGRAMS +=				\
	ppm_downscaler/yuv_downscaler

ppm_downscaler_yuv_downscaler_SOURCES =		\
	ppm_downscaler/yuv_downscaler.c

######
# Pi #
######

examplebin_PROGRAMS +=				\
	pi/pi

pi_pi_SOURCES =					\
	pi/pi.c					\
	pi/SobolQRNG/sobol_gold.c		\
	pi/SobolQRNG/sobol_primitives.c

if STARPU_USE_CUDA
pi_pi_SOURCES +=				\
	pi/pi_kernel.cu				\
	pi/SobolQRNG/sobol_gpu.cu
endif


################
# AXPY example #
################

if !NO_BLAS_LIB

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

examplebin_PROGRAMS += 				\
	mult/sgemm 				\
	mult/dgemm 				\
	mult/dw_mult_no_stride			\
	mult/dw_mult_no_stride_no_tag

mult_sgemm_SOURCES = 				\
	mult/sgemm.c				\
	common/blas.c				\
	common/blas_model.c

mult_dgemm_SOURCES = 				\
	mult/dgemm.c				\
	common/blas.c				\
	common/blas_model.c

mult_dw_mult_no_stride_SOURCES = 		\
	mult/dw_mult_no_stride.c		\
	mult/sgemm_kernels.c			\
	common/blas.c				\
	common/blas_model.c

mult_dw_mult_no_stride_no_tag_SOURCES =		\
	mult/dw_mult_no_stride_no_tag.c		\
	mult/sgemm_kernels.c			\
	common/blas.c				\
	common/blas_model.c

endif

####################
# Cholesky example #
####################

if !NO_BLAS_LIB

examplebin_PROGRAMS += 			\
	cholesky/dw_cholesky			\
	cholesky/dw_cholesky_no_stride		\
	cholesky/dw_cholesky_grain

cholesky_dw_cholesky_SOURCES =			\
	cholesky/dw_cholesky.c			\
	cholesky/dw_cholesky_models.c		\
	cholesky/dw_cholesky_kernels.c		\
	common/blas.c

cholesky_dw_cholesky_no_stride_SOURCES =	\
	cholesky/dw_cholesky_no_stride.c	\
	cholesky/dw_cholesky_models.c		\
	cholesky/dw_cholesky_kernels.c		\
	common/blas.c

cholesky_dw_cholesky_grain_SOURCES =		\
	cholesky/dw_cholesky_grain.c		\
	cholesky/dw_cholesky_models.c		\
	cholesky/dw_cholesky_kernels.c		\
	common/blas.c

endif

##############
# LU example #
##############

if !NO_BLAS_LIB

check_PROGRAMS +=				\
	lu/lu_example_float			\
	lu/lu_implicit_example_float

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

examplebin_PROGRAMS += 				\
	spmv/dw_spmv

spmv_dw_spmv_SOURCES = 				\
	spmv/dw_spmv.c

if STARPU_USE_CUDA
spmv_dw_spmv_SOURCES +=				\
	spmv/spmv_cuda.cu
endif

if ATLAS_BLAS_LIB
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
endif

nobase_STARPU_OPENCL_DATA_DATA += \
	incrementer/incrementer_kernels_opencl_codelet.cl

####################
# Variable example #
####################

check_PROGRAMS +=				\
	variable/variable

examplebin_PROGRAMS +=				\
	variable/variable

variable_variable_SOURCES =			\
	variable/variable.c			\
	variable/variable_kernels_cpu.c

if STARPU_USE_CUDA
variable_variable_SOURCES +=			\
	variable/variable_kernels.cu
endif
if STARPU_USE_OPENCL
variable_variable_SOURCES +=			\
	variable/variable_kernels_opencl.c
endif

nobase_STARPU_OPENCL_DATA_DATA += 		\
	variable/variable_kernels_opencl_codelet.cl


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

#################
# block example #
#################

check_PROGRAMS +=				\
	block/block

examplebin_PROGRAMS +=				\
	block/block

block_block_SOURCES =				\
	block/block.c				\
	block/block_cpu.c

if STARPU_USE_CUDA
block_block_SOURCES +=				\
	block/block_cuda.cu
endif

if STARPU_USE_OPENCL
block_block_SOURCES +=				\
	block/block_opencl.c
nobase_STARPU_OPENCL_DATA_DATA += 		\
	block/block_opencl_kernel.cl
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
