/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <semaphore.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <core/dependencies/tags.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <ctype.h>
#include <pthread.h>
#include <signal.h>
#include <cblas.h>

#include <datawizard/datawizard.h>
#include <task-models/blas_model.h>
#include <common/fxt.h>

#include <starpu.h>

#ifdef USE_CUDA
#include <cuda.h>
#endif

#define BLOCK	75

#include "starpu-blas-wrapper.h"

extern struct data_interface_ops_t interface_blas_ops;

static int core_sgemm = 0;
static int cublas_sgemm = 0;
static int core_strsm = 0;
static int cublas_strsm = 0;

static int inited = 0;

void STARPU_INIT(void)
{
	if (!inited) {
		inited = 1;
		starpu_init(NULL);	
	}
}

void STARPU_TERMINATE(void)
{
	starpu_shutdown();

	fprintf(stderr, "sgemm : core %d cublas %d\n", core_sgemm, cublas_sgemm);
	fprintf(stderr, "strsm : core %d cublas %d\n", core_strsm, cublas_strsm);
}

/*
 *
 *	Specific to PaStiX !
 *
 */

/*
 *	
 *	We "need" some custom filters
 *
 *			 VECTOR
 *			  (n)
 *			/  |   \		
 * 		   VECTOR  BLAS  VECTOR
 * 		    (n1)  (n2)	 
 *	
 *	if n1 = 0 :
 * 			VECTOR
 *			/   \
 *		     BLAS  VECTOR
 */

struct divide_vector_in_blas_filter_args {
	uint32_t n1, n2; /* (total size of the first portion (vector length) n < root's n ! */
	uint32_t stride; /* stride of the first portion (need to be a multiple of n */
};

unsigned divide_vector_in_blas_filter(starpu_filter *f, starpu_data_handle root_data)
{
	starpu_vector_interface_t *vector_root = &root_data->interface[0].vector;
		uint32_t nx = vector_root->nx;
		size_t elemsize = vector_root->elemsize;

	struct divide_vector_in_blas_filter_args *args = f->filter_arg_ptr;
		unsigned n1 = args->n1;
		unsigned n2 = args->n2;
		unsigned stride = args->stride;
		STARPU_ASSERT(n1 + n2 < nx);
		unsigned n3 = nx - n1 - n2;
		

	/* first allocate the children starpu_data_handle */
	root_data->children = calloc((n1==0)?2:3, sizeof(starpu_data_handle));
	STARPU_ASSERT(root_data->children);

	STARPU_ASSERT((n2 % args->stride) == 0);

	unsigned child = 0;
	unsigned node;
	
	if (n1 > 0)
	{
		for (node = 0; node < MAXNODES; node++)
		{
			starpu_vector_interface_t *local = &root_data->children[child].interface[node].vector;
	
			local->nx = n1;
			local->elemsize = elemsize;
	
			if (root_data->per_node[node].allocated) {
				local->ptr = root_data->interface[node].vector.ptr;
			}
	
		}

		child++;
	}
	
	for (node = 0; node < MAXNODES; node++)
	{
		starpu_blas_interface_t *local = &root_data->children[child].interface[node].blas;

		local->nx = stride;
		local->ny = n2/stride;
		local->ld = stride;
		local->elemsize = elemsize;

		if (root_data->per_node[node].allocated) {
			local->ptr = root_data->interface[node].vector.ptr + n1*elemsize;
		}

		struct starpu_data_state_t *state = &root_data->children[child];
		state->ops = &interface_blas_ops;
	}

	child++;

	for (node = 0; node < MAXNODES; node++)
	{
		starpu_vector_interface_t *local = &root_data->children[child].interface[node].vector;

		local->nx = n3;
		local->elemsize = elemsize;

		if (root_data->per_node[node].allocated) {
			local->ptr = root_data->interface[node].vector.ptr + (n1+n2)*elemsize;
		}
	}

	return (n1==0)?2:3;
}


static data_state *cblktab;

static void _cublas_cblk_strsm_callback(void *sem)
{
	sem_t *semptr = sem;
	sem_post(semptr);
}


void STARPU_MONITOR_DATA(unsigned ncols)
{
	cblktab = calloc(ncols, sizeof(data_state));
}

void STARPU_MONITOR_CBLK(unsigned col, float *data, unsigned stride, unsigned width)
{
	//void starpu_register_blas_data(struct starpu_data_state_t *state, uint32_t home_node,
        //                uintptr_t ptr, uint32_t ld, uint32_t nx,
        //                uint32_t ny, size_t elemsize);

	//fprintf(stderr, "col %d data %p stride %d width %d\n", col, data, stride, width);

	starpu_register_blas_data(&cblktab[col], 0 /* home */,
			(uintptr_t) data, stride, stride, width, sizeof(float));
	
}

static data_state work_block_1;
static data_state work_block_2;

void allocate_maxbloktab_on_cublas(starpu_data_interface_t *descr __attribute__((unused)), void *arg __attribute__((unused)))
{
	starpu_request_data_allocation(&work_block_1, 1);
	starpu_request_data_allocation(&work_block_2, 1);


	starpu_filter f1, f2;
	struct divide_vector_in_blas_filter_args args1, args2;

	f1.filter_func = divide_vector_in_blas_filter;
		args1.n1 = 1; /* XXX random ... */
		args1.n2 = 2;
		args1.stride = 1;

	f1.filter_arg_ptr = &args1;
	starpu_partition_data(&work_block_1, &f1);

	f2.filter_func = divide_vector_in_blas_filter;
		args2.n1 = 0;
		args2.n2 = 2;
		args2.stride = 1;
	f2.filter_arg_ptr = &args2;
	starpu_partition_data(&work_block_2, &f2);
}

void STARPU_DECLARE_WORK_BLOCKS(float *maxbloktab1, float *maxbloktab2, unsigned solv_coefmax)
{
	starpu_register_vector_data(&work_block_1, 0 /* home */, (uintptr_t)maxbloktab1, solv_coefmax, sizeof(float));
	starpu_register_vector_data(&work_block_2, 0 /* home */, (uintptr_t)maxbloktab2, solv_coefmax, sizeof(float));

	starpu_codelet cl;
	job_t j;
	sem_t sem;

	/* initialize codelet */
	cl.where = CUDA;
	cl.cuda_func = allocate_maxbloktab_on_cublas;
	
	j = job_create();
	j->cb = _cublas_cblk_strsm_callback;
	j->argcb = &sem;
	j->cl = &cl;
	j->cl_arg = NULL;

	j->nbuffers = 0;
	j->cl->model = NULL;

	sem_init(&sem, 0, 0U);
	
	/* submit the codelet */
	submit_job(j);

	/* wait for its completion */
	sem_wait(&sem);
	sem_destroy(&sem);

}

void _core_cblk_strsm(starpu_data_interface_t *descr, void *arg __attribute__((unused)))
{
	uint32_t nx, ny, ld;
	nx = descr[0].blas.nx;
	ny = descr[0].blas.ny;
	ld = descr[0].blas.ld;

	float *diag_cblkdata, *extra_cblkdata;
	diag_cblkdata = (float *)descr[0].blas.ptr;
	extra_cblkdata = diag_cblkdata + ny;

	unsigned m = nx - ny;
	unsigned n = ny;

//	SOPALIN_TRSM("R","L","T","U",dimb,dima,fun,ga,stride,gb,stride);
	core_strsm++;

	cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasUnit, m, n, 1.0f, 
			diag_cblkdata, ld, extra_cblkdata, ld);
}


void _cublas_cblk_strsm(starpu_data_interface_t *descr, void *arg __attribute__((unused)))
{
	uint32_t nx, ny, ld;
	nx = descr[0].blas.nx;
	ny = descr[0].blas.ny;
	ld = descr[0].blas.ld;

	float *diag_cblkdata, *extra_cblkdata;
	diag_cblkdata = (float *)descr[0].blas.ptr;
	extra_cblkdata = diag_cblkdata + ny;

	unsigned m = nx - ny;
	unsigned n = ny;
	
	cublas_strsm++;

	cublasStrsm ('R', 'L', 'T', 'U', m, n, 1.0, 
		diag_cblkdata, ld, 
		extra_cblkdata, ld);
	cublasStatus st = cublasGetError();
	if (st) fprintf(stderr, "ERROR %d\n", st);
	STARPU_ASSERT(st == CUBLAS_STATUS_SUCCESS);
}

static struct starpu_perfmodel_t starpu_cblk_strsm = {
	.per_arch = { 
		[STARPU_CORE_DEFAULT] = { .cost_model = starpu_cblk_strsm_core_cost },
		[STARPU_CUDA_DEFAULT] = { .cost_model = starpu_cblk_strsm_cuda_cost }
	},
//	.type = REGRESSION_BASED,
	.type = PER_ARCH,
	.symbol = "starpu_cblk_strsm"
};


void STARPU_CBLK_STRSM(unsigned col)
{
	/* perform a strsm on the block column */
	starpu_codelet cl;
	job_t j;
	sem_t sem;

	/* initialize codelet */
	cl.where = CORE|CUDA;
	cl.core_func = _core_cblk_strsm;
	cl.cuda_func = _cublas_cblk_strsm;
	
	j = job_create();
//	j->where = (starpu_get_blas_nx(&cblktab[col]) > BLOCK && starpu_get_blas_ny(&cblktab[col]) > BLOCK)? CUBLAS:CORE;
	j->cb = _cublas_cblk_strsm_callback;
	j->argcb = &sem;
	j->cl = &cl;
	j->cl_arg = NULL;

	j->nbuffers = 1;
	/* we could be a little more precise actually */
	j->buffers[0].handle = &cblktab[col];
	j->buffers[0].mode = STARPU_RW;
	
	j->cl->model = &starpu_cblk_strsm;

	sem_init(&sem, 0, 0U);
	
	/* submit the codelet */
	submit_job(j);

	/* wait for its completion */
	sem_wait(&sem);
	sem_destroy(&sem);
}

struct starpu_compute_contrib_compact_args {
	unsigned stride;
	int dimi;
	int dimj;
	int dima;
};


void _core_compute_contrib_compact(starpu_data_interface_t *descr, void *arg)
{
	struct starpu_compute_contrib_compact_args *args = arg;

	float *gaik = (float *)descr[0].blas.ptr + args->dima;
	float *gb = (float *)descr[1].blas.ptr; 
	unsigned strideb = (unsigned)descr[1].blas.ld;
	float *gc = (float *)descr[2].blas.ptr;
	unsigned stridec = (unsigned)descr[2].blas.ld;

	core_sgemm++;

	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, 
			args->dimi, args->dimj, args->dima,
			1.0f, gaik, args->stride,
			      gb, strideb,
			0.0 , gc, stridec);

}


void _cublas_compute_contrib_compact(starpu_data_interface_t *descr, void *arg)
{
	struct starpu_compute_contrib_compact_args *args = arg;

	float *gaik = (float *)descr[0].blas.ptr + args->dima;
	float *gb = (float *)descr[1].blas.ptr;
	unsigned strideb = (unsigned)descr[1].blas.ld;
	float *gc = (float *)descr[2].blas.ptr;
	unsigned stridec = (unsigned)descr[2].blas.ld;
	
	cublas_sgemm++;

	cublasSgemm('N','T', args->dimi, args->dimj, args->dima, 
			1.0, gaik, args->stride,
			     gb, strideb,
			0.0, gc, stridec);

	cublasStatus st = cublasGetError();
	if (st) fprintf(stderr, "ERROR %d\n", st);
	STARPU_ASSERT(st == CUBLAS_STATUS_SUCCESS);
}


static struct starpu_perfmodel_t starpu_compute_contrib_compact = {
	.per_arch = { 
		[STARPU_CORE_DEFAULT] = { .cost_model = starpu_compute_contrib_compact_core_cost },
		[STARPU_CUDA_DEFAULT] = { .cost_model = starpu_compute_contrib_compact_cuda_cost }
	},
//	.type = REGRESSION_BASED,
	.type = PER_ARCH,
	.symbol = "starpu_compute_contrib_compact"
};

int update_work_blocks(unsigned col, int dimi, int dimj, int dima, int stride)
{
	/* be paranoid XXX */
	notify_data_modification(get_sub_data(&work_block_1, 1, 0), 0);
	notify_data_modification(get_sub_data(&work_block_1, 1, 1), 0);
	//notify_data_modification(get_sub_data(&work_block_1, 1, 2), 0);
	notify_data_modification(get_sub_data(&work_block_2, 1, 0), 0);
	notify_data_modification(get_sub_data(&work_block_2, 1, 1), 0);
	notify_data_modification(&cblktab[col], 0);

	starpu_unpartition_data(&work_block_1, 0);
	starpu_unpartition_data(&work_block_2, 0);

	starpu_filter f1, f2;
	struct divide_vector_in_blas_filter_args args1, args2;

	f1.filter_func = divide_vector_in_blas_filter;
		args1.n1 = stride - dima - dimi; //STARPU_ASSERT(args1.n1 != 0);
		args1.n2 = (stride - dima)*dima;
		args1.stride = (stride - dima);

	f1.filter_arg_ptr = &args1;
	starpu_partition_data(&work_block_1, &f1);

	f2.filter_func = divide_vector_in_blas_filter;
		args2.n1 = 0;
		args2.n2 = dimi*dimj;
		args2.stride = dimi;
	f2.filter_arg_ptr = &args2;
	starpu_partition_data(&work_block_2, &f2);

	return (args1.n1!=0)?3:2;
}

void STARPU_COMPUTE_CONTRIB_COMPACT(unsigned col, int dimi, int dimj, int dima, int stride)
{
//        CUBLAS_SGEMM("N","T",dimi,dimj,dima, 1.0,gaik,stride,gb,stride-dima,
//               0.0 ,gc,dimi);
	
	struct starpu_compute_contrib_compact_args args;
		args.stride = stride;
		args.dimi = dimi;
		args.dimj = dimj;
		args.dima = dima;

	starpu_codelet cl;
	job_t j;
	sem_t sem;

	/* initialize codelet */
	cl.where = CUDA|CORE;
	cl.core_func = _core_compute_contrib_compact;
	cl.cuda_func = _cublas_compute_contrib_compact;
	
	j = job_create();

	j->cb = _cublas_cblk_strsm_callback;
	j->argcb = &sem;
	j->cl = &cl;
	j->cl_arg = &args;
	j->cl->model = &starpu_compute_contrib_compact;

	int ret;
	ret = update_work_blocks(col, dimi, dimj, dima, stride);

	j->nbuffers = 3;
	/* we could be a little more precise actually */
	j->buffers[0].handle = &cblktab[col]; // gaik
	j->buffers[0].mode = STARPU_R;
	j->buffers[1].handle = get_sub_data(&work_block_1, 1, (ret==2)?0:1);
	j->buffers[1].mode = STARPU_R;
	j->buffers[2].handle = get_sub_data(&work_block_2, 1, 0);; 
	j->buffers[2].mode = STARPU_RW; // XXX STARPU_W
	
	sem_init(&sem, 0, 0U);
	
	/* submit the codelet */
	submit_job(j);

	/* wait for its completion */
	sem_wait(&sem);
	sem_destroy(&sem);

}

/*
 *
 *	SGEMM
 *
 */

struct sgemm_args {
	char transa;
	char transb;
	int m, n, k;
	float alpha;
	float beta;
};


void _cublas_sgemm(starpu_data_interface_t *descr, void *arg)
{
	float *A, *B, *C;
	uint32_t nxA, nyA, ldA;
	uint32_t nxB, nyB, ldB;
	uint32_t nxC, nyC, ldC;

	A = (float *)descr[0].blas.ptr;
	nxA = descr[0].blas.nx;
	nyA = descr[0].blas.ny;
	ldA = descr[0].blas.ld;

	B = (float *)descr[1].blas.ptr;
	nxB = descr[1].blas.nx;
	nyB = descr[1].blas.ny;
	ldB = descr[1].blas.ld;

	C = (float *)descr[2].blas.ptr;
	nxC = descr[2].blas.nx;
	nyC = descr[2].blas.ny;
	ldC = descr[2].blas.ld;

	struct sgemm_args *args = arg;

//	fprintf(stderr, "CUBLAS SGEMM nxA %d nyA %d nxB %d nyB %d nxC %d nyC %d lda %d ldb %d ldc %d\n", nxA, nyA, nxB, nyB, nxC, nyC, ldA, ldB, ldC);

//	STARPU_ASSERT(nxA == nxC);
//	STARPU_ASSERT(nyA == nxB);
//	STARPU_ASSERT(nyB == nyC);
//
//	STARPU_ASSERT(nxA <= ldA);
//	STARPU_ASSERT(nxB <= ldB);
//	STARPU_ASSERT(nxC <= ldC);

	cublasSgemm (args->transa, args->transb, args->m, args->n, args->k, args->alpha, A, (int)ldA,
			B, (int)ldB, args->beta, C, (int)ldC);
	cublasStatus st = cublasGetError();
	if (st) fprintf(stderr, "ERROR %d\n", st);
	STARPU_ASSERT(st == CUBLAS_STATUS_SUCCESS);
}

static void _cublas_sgemm_callback(void *sem)
{
	sem_t *semptr = sem;
	sem_post(semptr);
}

void STARPU_SGEMM (const char *transa, const char *transb, const int m,
                   const int n, const int k, const float alpha,
                   const float *A, const int lda, const float *B,
                   const int ldb, const float beta, float *C, const int ldc)
{
	struct sgemm_args args;
		args.transa = *transa;
		args.transb = *transb;
		args.alpha = alpha;
		args.beta = beta;
		args.m = m;
		args.n = n;
		args.k = k;

	data_state A_state;
	data_state B_state;
	data_state C_state;

	starpu_codelet cl;
	job_t j;
	sem_t sem;

//	fprintf(stderr, "STARPU - SGEMM - TRANSA %c TRANSB %c m %d n %d k %d lda %d ldb %d ldc %d \n", *transa, *transb, m, n, k, lda, ldb, ldc);

	if (toupper(*transa) == 'N')
	{
		starpu_register_blas_data(&A_state, 0, (uintptr_t)A, lda, m, k, sizeof(float));
	}
	else 
	{
		starpu_register_blas_data(&A_state, 0, (uintptr_t)A, lda, k, m, sizeof(float));
	}

	if (toupper(*transb) == 'N')
	{
		starpu_register_blas_data(&B_state, 0, (uintptr_t)B, ldb, k, n, sizeof(float));
	}
	else 
	{	
		starpu_register_blas_data(&B_state, 0, (uintptr_t)B, ldb, n, k, sizeof(float));
	}

	starpu_register_blas_data(&C_state, 0, (uintptr_t)C, ldc, m, n, sizeof(float));

	/* initialize codelet */
	cl.where = CUDA;
	//cl.core_func = _core_strsm;
	cl.cuda_func = _cublas_sgemm;
	
	j = job_create();
	j->cb = _cublas_sgemm_callback;
	j->argcb = &sem;
	j->cl = &cl;
	j->cl_arg = &args;

	j->nbuffers = 3;
	j->buffers[0].handle = &A_state;
	j->buffers[0].mode = STARPU_R;
	j->buffers[1].handle = &B_state;
	j->buffers[1].mode = STARPU_R;
	j->buffers[2].handle = &C_state;
	j->buffers[2].mode = STARPU_RW;
	
	j->cl->model = NULL;

	sem_init(&sem, 0, 0U);
	
	/* submit the codelet */
	submit_job(j);

	/* wait for its completion */
	sem_wait(&sem);
	sem_destroy(&sem);

	/* make sure data are in memory again */
	starpu_unpartition_data(&A_state, 0);
	starpu_unpartition_data(&B_state, 0);
	starpu_unpartition_data(&C_state, 0);
	//starpu_delete_data(&A_state);
	//starpu_delete_data(&B_state);
	//starpu_delete_data(&C_state);
	
//	fprintf(stderr, "SGEMM done\n");
}


/*
 *
 *	STRSM
 *
 */

struct strsm_args {
	char side;
	char uplo;
	char transa;
	char diag;
	float alpha;
	int m,n;
};
//
//void _core_strsm(starpu_data_interface_t *descr, void *arg)
//{
//	float *A, *B;
//	uint32_t nxA, nyA, ldA;
//	uint32_t nxB, nyB, ldB;
//
//	A = (float *)descr[0].blas.ptr;
//	nxA = descr[0].blas.nx;
//	nyA = descr[0].blas.ny;
//	ldA = descr[0].blas.ld;
//
//	B = (float *)descr[1].blas.ptr;
//	nxB = descr[1].blas.nx;
//	nyB = descr[1].blas.ny;
//	ldB = descr[1].blas.ld;
//
//	struct strsm_args *args = arg;
//
//	fprintf(stderr, "CORE STRSM nxA %d nyA %d nxB %d nyB %d lda %d ldb %d\n", nxA, nyA, nxB, nyB, ldA, ldB);
//
//	SOPALIN_TRSM("R","L","T","U",dimb,dima,fun,ga,stride,gb,stride);
//	
//}

/* 
 *	
 *	
 *
 */


void CUBLAS_SGEMM (const char *transa, const char *transb, const int m,
                   const int n, const int k, const float alpha,
                   const float *A, const int lda, const float *B,
                   const int ldb, const float beta, float *C, const int ldc)
{
    int ka, kb;
    float *devPtrA, *devPtrB, *devPtrC;

//   printf("CUBLAS SGEMM : m %d n %d k %d lda %d ldb %d ldc %d\n", m, n, k, lda, ldb, ldc);

    /*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
     *           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
     *           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
     *           part of the array  A  must contain the matrix  A,  otherwise
     *           the leading  k by m  part of the array  A  must contain  the
     *           matrix A.
     */
    ka = (toupper(transa[0]) == 'N') ? k : m;
    cublasAlloc (lda * ka, sizeof(devPtrA[0]), (void**)&devPtrA);
    if (toupper(transa[0]) == 'N') {
        cublasSetMatrix (STARPU_MIN(m,lda), k, sizeof(A[0]), A, lda, devPtrA, 
                         lda);
    } else {
        cublasSetMatrix (STARPU_MIN(k,lda), m, sizeof(A[0]), A, lda, devPtrA, 
                         lda);
    }

    /*  B      - REAL             array of DIMENSION ( LDB, kb ), where kb is
     *           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
     *           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
     *           part of the array  B  must contain the matrix  B,  otherwise
     *           the leading  n by k  part of the array  B  must contain  the
     *           matrix B.
     */
    kb = (toupper(transb[0]) == 'N') ? n : k;
    cublasAlloc (ldb * kb, sizeof(devPtrB[0]), (void**)&devPtrB);
    if (toupper(transb[0]) == 'N') {
        cublasSetMatrix (STARPU_MIN(k,ldb), n, sizeof(B[0]), B, ldb, devPtrB, 
                         ldb);
    } else {
        cublasSetMatrix (STARPU_MIN(n,ldb), k, sizeof(B[0]), B, ldb, devPtrB,
                         ldb);
    }
    
    /*  C      - REAL             array of DIMENSION ( LDC, n ).
     *           Before entry, the leading  m by n  part of the array  C must
     *           contain the matrix  C,  except when  beta  is zero, in which
     *           case C need not be set on entry.
     *           On exit, the array  C  is overwritten by the  m by n  matrix
     */
    cublasAlloc ((ldc) * (n), sizeof(devPtrC[0]), (void**)&devPtrC);
    cublasSetMatrix (STARPU_MIN(m,ldc), n, sizeof(C[0]), C, ldc, devPtrC, ldc);

    cublasSgemm (transa[0], transb[0], m, n, k, alpha, devPtrA, lda, 
                 devPtrB, ldb, beta, devPtrC, ldc);

    cublasGetMatrix (STARPU_MIN(m,ldc), n, sizeof(C[0]), devPtrC, ldc, C, ldc);
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
}


