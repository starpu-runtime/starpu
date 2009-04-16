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
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/examples/

TESTS = $(check_PROGRAMS)

check_PROGRAMS =

EXTRA_DIST = 					\
	cuda/incrementer_cuda.cu		\
	cuda/spmv_cuda.cu

CLEANFILES = 					\
	cuda/incrementer_cuda.cubin		\
	cuda/spmv_cuda.cubin			

CLEANFILES += *.gcno *.gcda *.linkinfo

if USE_CUDA

# TODO define NVCCFLAGS
NVCC ?= nvcc

.cu.cubin:
	$(MKDIR_P) `dirname $@`
	$(NVCC) -cubin $< -o $@ --compiler-options -fno-strict-aliasing  $(NVCCFLAGS)

.cu.o:
	$(NVCC) $< -c -o $@ --compiler-options -fno-strict-aliasing  $(NVCCFLAGS) -I${includedir}


BUILT_SOURCES =					\
	cuda/incrementer_cuda.cubin		\
	cuda/spmv_cuda.cubin			

endif

examplebindir = $(libdir)/starpu/examples/

examplebin_PROGRAMS =

noinst_HEADERS = 				\
	heat/lu_kernels_model.h			\
	heat/dw_sparse_cg.h			\
	heat/heat.h				\
	heat/dw_factolu.h			\
	cholesky/dw_cholesky_models.h		\
	cholesky/dw_cholesky.h			\
	common/blas_model.h			\
	common/blas.h				\
	mult/dw_mult.h				\
	cuda/incrementer_cuda.h			\
	fortran/bindings/StarPU-fortran.h	\
	strassen/strassen.h			\
	strassen/strassen_models.h		\
	spmv/matrix-market/mmio.h		\
	spmv/matrix-market/mm_to_bcsr.h		\
	spmv/dw_spmv.h				\
	spmv/dw_block_spmv.h

################
# Mult example #
################

if !NO_BLAS_LIB

examplebin_PROGRAMS += 			\
	mult/dw_mult 				\
	mult/dw_mult_no_stride			\
	mult/dw_mult_no_stride_no_tag 

mult_dw_mult_SOURCES = 				\
	mult/dw_mult.c				\
	common/blas.c				\
	common/blas_model.c
		
mult_dw_mult_no_stride_SOURCES = 		\
	mult/dw_mult_no_stride.c		\
	common/blas.c				\
	common/blas_model.c

mult_dw_mult_no_stride_no_tag_SOURCES =		\
	mult/dw_mult_no_stride_no_tag.c		\
	common/blas.c				\
	common/blas_model.c

endif

####################
# Cholesky example #
####################

if !NO_BLAS_LIB

examplebin_PROGRAMS += 			\
	cholesky/dw_cholesky			\
	cholesky/dw_cholesky_no_stride

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
	tag_example/tag_example2

examplebin_PROGRAMS +=			\
	tag_example/tag_example			\
	tag_example/tag_example3		\
	tag_example/tag_example2

tag_example_tag_example_SOURCES =		\
	tag_example/tag_example.c

tag_example_tag_example2_SOURCES =		\
	tag_example/tag_example2.c

tag_example_tag_example3_SOURCES =		\
	tag_example/tag_example2.c

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

spmv_dw_block_spmv_SOURCES =			\
	spmv/dw_block_spmv.c			\
	spmv/dw_block_spmv_kernels.c		\
	spmv/matrix-market/mm_to_bcsr.c		\
	spmv/matrix-market/mmio.c

endif

#######################
# Incrementer example #
#######################


check_PROGRAMS +=				\
	incrementer/incrementer			\
	incrementer/incrementer_runtime

examplebin_PROGRAMS +=				\
	incrementer/incrementer			\
	incrementer/incrementer_runtime

incrementer_incrementer_SOURCES =		\
	incrementer/incrementer.c

if USE_CUDA
incrementer_incrementer_runtime_SOURCES =	\
	incrementer/incrementer_runtime.c	\
	incrementer/incrementer_runtime_kernels.cu
else
incrementer_incrementer_runtime_SOURCES =	\
	incrementer/incrementer_runtime.c
endif
