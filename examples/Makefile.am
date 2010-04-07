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
	cuda/incrementer_cuda.cu		\
	spmv/spmv_cuda.cu			\
	gordon/null_kernel_gordon.c		\
	mult/xgemm.c				\
	mult/xgemm_kernels.c			\
	mult/gordon/func_sgemm_ibm.c		\
	lu/xlu.c				\
	lu/xlu_pivot.c				\
	lu/xlu_kernels.c			\
	lu/lu_example.c

CLEANFILES = 					\
	gordon/null_kernel_gordon.spuelf	\
	mult/gordon/func_sgemm_ibm.spuelf


CLEANFILES += *.gcno *.gcda *.linkinfo

if STARPU_USE_CUDA

# TODO define NVCCFLAGS
NVCC ?= nvcc

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
	strassen/strassen.h			\
	strassen/strassen_models.h		\
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
	basic_examples/vector_scal.c

examplebin_PROGRAMS +=				\
	basic_examples/mult

basic_examples_mult_SOURCES =			\
	basic_examples/mult.c

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

####################
# Strassen example #
####################

if ATLAS_BLAS_LIB

examplebin_PROGRAMS += strassen/dw_strassen

strassen_dw_strassen_SOURCES = 			\
	strassen/strassen.c			\
	strassen/strassen_kernels.c		\
	strassen/test_strassen.c		\
	strassen/strassen_models.c		\
	common/blas.c

endif

#####################
# Strassen2 example #
#####################

if !NO_BLAS_LIB

check_PROGRAMS += strassen2/strassen

examplebin_PROGRAMS += strassen2/strassen

strassen2_strassen_SOURCES = 			\
	strassen2/strassen2.c			\
	strassen2/strassen2_kernels.c		\
	common/blas.c

endif

################
# SpMV example #
################

if ATLAS_BLAS_LIB

examplebin_PROGRAMS += 				\
	spmv/dw_spmv				\
	spmv/dw_block_spmv

spmv_dw_spmv_SOURCES = 				\
	spmv/dw_spmv.c

if STARPU_USE_CUDA
spmv_dw_spmv_SOURCES +=				\
	spmv/spmv_cuda.cu
endif

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

if STARPU_USE_CUDA
incrementer_incrementer_SOURCES =	\
	incrementer/incrementer.c	\
	incrementer/incrementer_kernels.cu
else
incrementer_incrementer_SOURCES =	\
	incrementer/incrementer.c
endif
