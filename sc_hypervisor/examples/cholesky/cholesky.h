/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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

#define TAG_POTRF(k)	((starpu_tag_t)( (1ULL<<60) | (unsigned long long)(k)))
#define TAG_TRSM(k,j)	((starpu_tag_t)(((3ULL<<60) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j))))
#define TAG_GEMM(k,i,j)	((starpu_tag_t)(((4ULL<<60) | ((unsigned long long)(k)<<32) 	\
					| ((unsigned long long)(i)<<16)	\
					| (unsigned long long)(j))))

#define TAG_POTRF_AUX(k, prefix)	((starpu_tag_t)( (((unsigned long long)(prefix))<<60)  |  (1ULL<<56) | (unsigned long long)(k)))
#define TAG_TRSM_AUX(k,j, prefix)	((starpu_tag_t)( (((unsigned long long)(prefix))<<60)  			\
					|  ((3ULL<<56) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j))))
#define TAG_GEMM_AUX(k,i,j, prefix)    ((starpu_tag_t)(  (((unsigned long long)(prefix))<<60)	\
					|  ((4ULL<<56) | ((unsigned long long)(k)<<32)  	\
					| ((unsigned long long)(i)<<16) 			\
					| (unsigned long long)(j))))

#define BLOCKSIZE	(size/nblocks)

#define BLAS3_FLOP(n1,n2,n3)    \
        (2*((uint64_t)n1)*((uint64_t)n2)*((uint64_t)n3))

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
