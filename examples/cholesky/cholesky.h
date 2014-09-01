/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
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
#ifdef STARPU_USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas.h>
#endif

#include <common/blas.h>
#include <starpu.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)
#define NMAXBLOCKS	32

#define TAG11(k)	((starpu_tag_t)( (1ULL<<60) | (unsigned long long)(k)))
#define TAG21(k,j)	((starpu_tag_t)(((3ULL<<60) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j))))
#define TAG22(k,i,j)	((starpu_tag_t)(((4ULL<<60) | ((unsigned long long)(k)<<32) 	\
					| ((unsigned long long)(i)<<16)	\
					| (unsigned long long)(j))))

#define TAG11_AUX(k, prefix)	((starpu_tag_t)( (((unsigned long long)(prefix))<<60)  |  (1ULL<<56) | (unsigned long long)(k)))
#define TAG21_AUX(k,j, prefix)	((starpu_tag_t)( (((unsigned long long)(prefix))<<60)  			\
					|  ((3ULL<<56) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j))))
#define TAG22_AUX(k,i,j, prefix)    ((starpu_tag_t)(  (((unsigned long long)(prefix))<<60)	\
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
#define FADDS_POTRF(__n) ((double)(__n) * (((1. / 6.) * (double)(__n)      ) * (double)(__n) - (1. / 6.)))

#define FLOPS_SPOTRF(__n) (     FMULS_POTRF((__n)) +       FADDS_POTRF((__n)) )

#define FMULS_TRMM_2(__m, __n) (0.5 * (double)(__n) * (double)(__m) * ((double)(__m)+1.))
#define FADDS_TRMM_2(__m, __n) (0.5 * (double)(__n) * (double)(__m) * ((double)(__m)-1.))

#define FMULS_TRMM(__m, __n) ( /*( (__side) == PlasmaLeft ) ? FMULS_TRMM_2((__m), (__n)) :*/ FMULS_TRMM_2((__n), (__m)) )
#define FADDS_TRMM(__m, __n) ( /*( (__side) == PlasmaLeft ) ? FADDS_TRMM_2((__m), (__n)) :*/ FADDS_TRMM_2((__n), (__m)) )

#define FMULS_TRSM FMULS_TRMM
#define FADDS_TRSM FMULS_TRMM

#define FLOPS_STRSM(__m, __n) (     FMULS_TRSM((__m), (__n)) +       FADDS_TRSM((__m), (__n)) )


#define FMULS_GEMM(__m, __n, __k) ((double)(__m) * (double)(__n) * (double)(__k))
#define FADDS_GEMM(__m, __n, __k) ((double)(__m) * (double)(__n) * (double)(__k))

#define FLOPS_SGEMM(__m, __n, __k) (     FMULS_GEMM((__m), (__n), (__k)) +       FADDS_GEMM((__m), (__n), (__k)) )

/* End of magma code */

static unsigned size = 4*1024;
static unsigned nblocks = 16;
static unsigned nbigblocks = 8;
static unsigned pinned = 1;
static unsigned noprio = 0;
static unsigned check = 0;
static unsigned bound = 0;
static unsigned bound_deps = 0;
static unsigned bound_lp = 0;
static unsigned bound_mps = 0;
static unsigned with_ctxs = 0;
static unsigned with_noctxs = 0;
static unsigned chole1 = 0;
static unsigned chole2 = 0;

void chol_cpu_codelet_update_u11(void **, void *);
void chol_cpu_codelet_update_u21(void **, void *);
void chol_cpu_codelet_update_u22(void **, void *);

#ifdef STARPU_USE_CUDA
void chol_cublas_codelet_update_u11(void *descr[], void *_args);
void chol_cublas_codelet_update_u21(void *descr[], void *_args);
void chol_cublas_codelet_update_u22(void *descr[], void *_args);
#endif

extern struct starpu_perfmodel chol_model_11;
extern struct starpu_perfmodel chol_model_21;
extern struct starpu_perfmodel chol_model_22;

static void STARPU_ATTRIBUTE_UNUSED parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-with_ctxs") == 0) 
		{
			with_ctxs = 1;
			break;
		} else
		if (strcmp(argv[i], "-with_noctxs") == 0) 
		{
			with_noctxs = 1;
			break;
		} else
		
		if (strcmp(argv[i], "-chole1") == 0) 
		{
			chole1 = 1;
			break;
		} else

		if (strcmp(argv[i], "-chole2") == 0) 
		{
			chole2 = 1;
			break;
		} else

		if (strcmp(argv[i], "-size") == 0)
		{
		        char *argptr;
			size = strtol(argv[++i], &argptr, 10);
		} else

		if (strcmp(argv[i], "-nblocks") == 0)
		{
		        char *argptr;
			nblocks = strtol(argv[++i], &argptr, 10);
		} else

		if (strcmp(argv[i], "-nbigblocks") == 0)
		{
		        char *argptr;
			nbigblocks = strtol(argv[++i], &argptr, 10);
		} else

		if (strcmp(argv[i], "-no-pin") == 0)
		{
			pinned = 0;
		} else

		if (strcmp(argv[i], "-no-prio") == 0)
		{
			noprio = 1;
		} else

		if (strcmp(argv[i], "-bound") == 0)
		{
			bound = 1;
		} else

		if (strcmp(argv[i], "-bound-lp") == 0)
		{
			bound_lp = 1;
		} else

		if (strcmp(argv[i], "-bound-mps") == 0)
		{
			bound_mps = 1;
		} else

		if (strcmp(argv[i], "-bound-deps") == 0)
		{
			bound_deps = 1;
		} else

		if (strcmp(argv[i], "-check") == 0)
		{
			check = 1;
		} else

		/* if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i],"--help") == 0) */
		{
			fprintf(stderr,"usage : %s [-size size] [-nblocks nblocks] [-no-pin] [-no-prio] [-bound] [-bound-deps] [-bound-lp] [-check]\n", argv[0]);
			fprintf(stderr,"Currently selected: %ux%u and %ux%u blocks\n", size, size, nblocks, nblocks);
			exit(0);
		}
	}
}

#endif /* __DW_CHOLESKY_H__ */
