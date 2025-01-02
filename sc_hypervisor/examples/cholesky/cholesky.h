/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013-2013  Thibaut Lambert
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#ifndef __DW_CHOLESKY_H__
#define __DW_CHOLESKY_H__

#include <limits.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#ifdef STARPU_USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas.h>
#endif

#include <common/blas.h>
#include <starpu.h>
#include <starpu_bound.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)
#define NMAXBLOCKS	32

#define TAG_POTRF(k)	((starpu_tag_t)((1ULL<<60) | (unsigned long long)(k)))
#define TAG_TRSM(k,j)	((starpu_tag_t)((3ULL<<60) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j))))
#define TAG_GEMM(k,i,j)	((starpu_tag_t)(((4ULL<<60) | ((unsigned long long)(k)<<32) 	\
					| ((unsigned long long)(i)<<16)	\
					| (unsigned long long)(j))))

#define TAG_POTRF_AUX(k, prefix)	((starpu_tag_t)((((unsigned long long)(prefix))<<60)  |  (1ULL<<56) | (unsigned long long)(k)))
#define TAG_TRSM_AUX(k,j, prefix)	((starpu_tag_t)((((unsigned long long)(prefix))<<60)  			\
					|  ((3ULL<<56) |(((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j))))
#define TAG_GEMM_AUX(k,i,j, prefix)    ((starpu_tag_t)((((unsigned long long)(prefix))<<60)	\
					|  ((4ULL<<56) | ((unsigned long long)(k)<<32)  	\
					| ((unsigned long long)(i)<<16) 			\
					| (unsigned long long)(j))))

#define BLOCKSIZE	(size/nblocks)

#define BLAS3_FLOP(n1,n2,n3)    \
        (2*((uint64_t)n1)*((uint64_t)n2)*((uint64_t)n3))

/* This is from magma

  -- Innovative Computing Laboratory
  -- Electrical Engineering and Computer Science Department
  -- University of Tennessee
  -- (C) Copyright 2009

  Redistribution  and  use  in  source and binary forms, with or without
  modification,  are  permitted  provided  that the following conditions
  are met:

  * Redistributions  of  source  code  must  retain  the above copyright
    notice,  this  list  of  conditions  and  the  following  disclaimer.
  * Redistributions  in  binary  form must reproduce the above copyright
    notice,  this list of conditions and the following disclaimer in the
    documentation  and/or other materials provided with the distribution.
  * Neither  the  name of the University of Tennessee, Knoxville nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

  THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  ``AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A  PARTICULAR  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT NOT
  LIMITED  TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA,  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY  OF  LIABILITY,  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF  THIS  SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  */

#define FMULS_POTRF(__n) ((double)(__n) * (((1. / 6.) * (double)(__n) + 0.5) * (double)(__n) + (1. / 3.)))
#define FADDS_POTRF(__n) ((double)(__n) * (((1. / 6.) * (double)(__n)) * (double)(__n) - (1. / 6.)))

#define FLOPS_SPOTRF(__n) (FMULS_POTRF((__n)) + FADDS_POTRF((__n)))

#define FMULS_TRMM_2(__m, __n) (0.5 * (double)(__n) * (double)(__m) * ((double)(__m)+1.))
#define FADDS_TRMM_2(__m, __n) (0.5 * (double)(__n) * (double)(__m) * ((double)(__m)-1.))

#define FMULS_TRMM(__m, __n) (/*((__side) == PlasmaLeft) ? FMULS_TRMM_2((__m), (__n)) :*/ FMULS_TRMM_2((__n), (__m)))
#define FADDS_TRMM(__m, __n) (/*((__side) == PlasmaLeft) ? FADDS_TRMM_2((__m), (__n)) :*/ FADDS_TRMM_2((__n), (__m)))

#define FMULS_TRSM FMULS_TRMM
#define FADDS_TRSM FMULS_TRMM

#define FLOPS_STRSM(__m, __n) (FMULS_TRSM((__m), (__n)) + FADDS_TRSM((__m), (__n)))


#define FMULS_SYRK(__k, __n) (0.5 * (double)(__k) * (double)(__n) * ((double)(__n)+1.))
#define FADDS_SYRK(__k, __n) (0.5 * (double)(__k) * (double)(__n) * ((double)(__n)+1.))

#define FLOPS_SSYRK(__k, __n) (FMULS_SYRK((__k), (__n)) + FADDS_SYRK((__k), (__n)))


#define FMULS_GEMM(__m, __n, __k) ((double)(__m) * (double)(__n) * (double)(__k))
#define FADDS_GEMM(__m, __n, __k) ((double)(__m) * (double)(__n) * (double)(__k))

#define FLOPS_SGEMM(__m, __n, __k) (FMULS_GEMM((__m), (__n), (__k)) + FADDS_GEMM((__m), (__n), (__k)))

/* End of magma code */

extern unsigned g_size;
extern unsigned g_nblocks;
extern unsigned g_nbigblocks;
extern unsigned g_pinned;
extern unsigned g_noprio;
extern unsigned g_check;
extern unsigned g_bound;
extern unsigned g_with_ctxs;
extern unsigned g_with_noctxs;
extern unsigned g_chole1;
extern unsigned g_chole2;

extern struct starpu_perfmodel chol_model_potrf;
extern struct starpu_perfmodel chol_model_trsm;
extern struct starpu_perfmodel chol_model_syrk;
extern struct starpu_perfmodel chol_model_gemm;

void chol_cpu_codelet_update_potrf(void **, void *);
void chol_cpu_codelet_update_trsm(void **, void *);
void chol_cpu_codelet_update_syrk(void **, void *);
void chol_cpu_codelet_update_gemm(void **, void *);

extern struct starpu_codelet cl_potrf;
extern struct starpu_codelet cl_trsm;
extern struct starpu_codelet cl_syrk;
extern struct starpu_codelet cl_gemm;

double cpu_chol_task_potrf_cost(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl);
double cpu_chol_task_trsm_cost(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl);
double cpu_chol_task_syrk_cost(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl);
double cpu_chol_task_gemm_cost(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl);

#ifdef STARPU_USE_CUDA
void chol_cublas_codelet_update_potrf(void *descr[], void *_args);
void chol_cublas_codelet_update_trsm(void *descr[], void *_args);
void chol_cublas_codelet_update_syrk(void *descr[], void *_args);
void chol_cublas_codelet_update_gemm(void *descr[], void *_args);

double cuda_chol_task_potrf_cost(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl);
double cuda_chol_task_trsm_cost(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl);
double cuda_chol_task_syrk_cost(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl);
double cuda_chol_task_gemm_cost(struct starpu_task *task, struct starpu_perfmodel_arch* arch, unsigned nimpl);
#endif

void initialize_chol_model(struct starpu_perfmodel* model, char* symbol,
			   double (*cpu_cost_function)(struct starpu_task *, struct starpu_perfmodel_arch*, unsigned),
			   double (*cuda_cost_function)(struct starpu_task *, struct starpu_perfmodel_arch*, unsigned));

void parse_args(int argc, char **argv);

#endif /* __DW_CHOLESKY_H__ */
