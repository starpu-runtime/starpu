/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#ifdef STARPU_USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <common/blas.h>
#include <starpu.h>

#ifdef STARPU_HAVE_VALGRIND_H
#include <valgrind/valgrind.h>
#endif

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)
#define PRINTF(fmt, ...) do { if (!getenv("STARPU_SSILENT")) {printf(fmt, ## __VA_ARGS__); }} while(0)
#define NMAXBLOCKS	128

#define TAG_POTRF(k)	((starpu_tag_t)((1ULL<<60) | (unsigned long long)(k)))
#define TAG_TRSM(k,j)	((starpu_tag_t)(((3ULL<<60) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j))))
#define TAG_GEMM(k,i,j)	((starpu_tag_t)(((4ULL<<60) | ((unsigned long long)(k)<<32) 	\
					 | ((unsigned long long)(i)<<16) \
					 | (unsigned long long)(j))))

#define TAG_POTRF_AUX(k, prefix)	((starpu_tag_t)((((unsigned long long)(prefix))<<60)  |  (1ULL<<56) | (unsigned long long)(k)))
#define TAG_TRSM_AUX(k,j, prefix)	((starpu_tag_t)((((unsigned long long)(prefix))<<60) \
							 |  ((3ULL<<56) | (((unsigned long long)(k))<<32) \
							     | (unsigned long long)(j))))
#define TAG_GEMM_AUX(k,i,j, prefix)    ((starpu_tag_t)((((unsigned long long)(prefix))<<60) \
							 |  ((4ULL<<56) | ((unsigned long long)(k)<<32)	\
							     | ((unsigned long long)(i)<<16) \
							     | (unsigned long long)(j))))

#define BLOCKSIZE	(size_p/nblocks_p)

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

static unsigned size_p;
static unsigned nblocks_p;
static unsigned nbigblocks_p;
static int priority_attribution_p;
static int niter_p;
static int pause_resume_p;

static inline void init_sizes(void)
{
	int power = starpu_cpu_worker_get_count() + 32 * starpu_cuda_worker_get_count();
	int power_cbrt = cbrt(power);
#ifndef STARPU_LONG_CHECK
	power_cbrt /= 2;
#endif

	if (power_cbrt < 1)
		power_cbrt = 1;

#ifdef STARPU_QUICK_CHECK
	if (!size_p)
		size_p = 320*2*power_cbrt;
	if (!nblocks_p)
		nblocks_p = 2*power_cbrt;
	if (!nbigblocks_p)
		nbigblocks_p = power_cbrt;
#else
	if (!size_p)
		size_p = 960*8*power_cbrt;
	if (!nblocks_p)
		nblocks_p = 8*power_cbrt;
	if (!nbigblocks_p)
		nbigblocks_p = 4*power_cbrt;
#endif
	priority_attribution_p = 0;
	niter_p = 1;
	pause_resume_p = 0;
}

static unsigned pinned_p = 1;
static unsigned noprio_p = 0;
static unsigned check_p = 0;
static unsigned bound_p = 0;
static unsigned bound_deps_p = 0;
static unsigned bound_lp_p = 0;
static unsigned bound_mps_p = 0;
static unsigned with_ctxs_p = 0;
static unsigned with_noctxs_p = 0;
static unsigned chole1_p = 0;
static unsigned chole2_p = 0;
static unsigned median_p = 0;

extern struct starpu_perfmodel chol_model_potrf;
extern struct starpu_perfmodel chol_model_trsm;
extern struct starpu_perfmodel chol_model_syrk;
extern struct starpu_perfmodel chol_model_gemm;

extern struct starpu_codelet cl_potrf;
extern struct starpu_codelet cl_trsm;
extern struct starpu_codelet cl_syrk;
extern struct starpu_codelet cl_gemm;
extern struct starpu_codelet cl_potrf_gpu;
extern struct starpu_codelet cl_trsm_gpu;
extern struct starpu_codelet cl_syrk_gpu;
extern struct starpu_codelet cl_gemm_gpu;
extern struct starpu_codelet cl_potrf_cpu;
extern struct starpu_codelet cl_trsm_cpu;
extern struct starpu_codelet cl_syrk_cpu;
extern struct starpu_codelet cl_gemm_cpu;

extern starpu_data_handle_t scratch;

void cholesky_kernel_init(int nb);
void cholesky_kernel_fini(void);

void chol_cpu_codelet_update_potrf(void **, void *);
void chol_cpu_codelet_update_trsm(void **, void *);
void chol_cpu_codelet_update_syrk(void **, void *);
void chol_cpu_codelet_update_gemm(void **, void *);

#ifdef STARPU_USE_CUDA
void chol_cublas_codelet_update_potrf(void *descr[], void *_args);
void chol_cublas_codelet_update_trsm(void *descr[], void *_args);
void chol_cublas_codelet_update_syrk(void *descr[], void *_args);
void chol_cublas_codelet_update_gemm(void *descr[], void *_args);
#endif

static void _parse_args(int argc, char **argv, int extra)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-with_ctxs") == 0)
		{
			with_ctxs_p = 1;
			break;
		}
		else if (strcmp(argv[i], "-with_noctxs") == 0)
		{
			with_noctxs_p = 1;
			break;
		}
		else if (strcmp(argv[i], "-chole1") == 0)
		{
			chole1_p = 1;
			break;
		}
		else if (strcmp(argv[i], "-chole2") == 0)
		{
			chole2_p = 1;
			break;
		}
		else if (strcmp(argv[i], "-size") == 0)
		{
		        char *argptr;
			size_p = strtol(argv[++i], &argptr, 10);
		}
		else if (strcmp(argv[i], "-nblocks") == 0)
		{
		        char *argptr;
			nblocks_p = strtol(argv[++i], &argptr, 10);
		}
		else if (strcmp(argv[i], "-nbigblocks") == 0)
		{
		        char *argptr;
			nbigblocks_p = strtol(argv[++i], &argptr, 10);
		}
		else if (strcmp(argv[i], "-no-pin") == 0)
		{
			pinned_p = 0;
		}
		else if (strcmp(argv[i], "-no-prio") == 0)
		{
			noprio_p = 1;
		}
		else if (extra && strcmp(argv[i], "-priority") == 0)
		{
			char *argptr;
			priority_attribution_p = strtol(argv[++i], &argptr, 10);
			STARPU_ASSERT_MSG(priority_attribution_p >= 0 && priority_attribution_p <= 3, "priority attribution value %d is invalid\n", priority_attribution_p);
		}
		else if (extra && strcmp(argv[i], "-niter") == 0)
		{
			char *argptr;
			niter_p = strtol(argv[++i], &argptr, 10);
		}
		else if (extra && strcmp(argv[i], "-pause-resume") == 0)
		{
			pause_resume_p = 1;
		}
		else if (strcmp(argv[i], "-commute") == 0)
		{
			cl_syrk.modes[1] |= STARPU_COMMUTE;
			cl_gemm.modes[2] |= STARPU_COMMUTE;
		}
		else if (strcmp(argv[i], "-bound") == 0)
		{
			bound_p = 1;
		}
		else if (strcmp(argv[i], "-bound-lp") == 0)
		{
			bound_lp_p = 1;
		}
		else if (strcmp(argv[i], "-bound-mps") == 0)
		{
			bound_mps_p = 1;
		}
		else if (strcmp(argv[i], "-bound-deps") == 0)
		{
			bound_deps_p = 1;
		}
		else if (strcmp(argv[i], "-check") == 0)
		{
			check_p = 1;
		}
		else if (strcmp(argv[i], "-median") == 0)
		{
			median_p = 1;
		}
		else
		/* if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i],"--help") == 0) */
		{
			fprintf(stderr,"usage : %s [-size size] [-nblocks nblocks] [-no-pin] [-no-prio] [-bound] [-bound-deps] [-bound-lp] %s[-check]\n", argv[0], extra == 1 ? "[-priority p (0: StarPU's priorities, 1: Bottom level priorities, 2: Bottom level priorities with tasks times, 3: PaRSEC's priorities)] [-niter n] [-pause-resume] [-median] " : "");
			fprintf(stderr,"Currently selected: %ux%u and %ux%u blocks", size_p, size_p, nblocks_p, nblocks_p);
			if (extra)
				fprintf(stderr, " with priority %d and number of iterations %d\n", priority_attribution_p, niter_p);
			fprintf(stderr, "\n");
			exit(0);
		}
	}

#ifdef STARPU_HAVE_VALGRIND_H
       if (RUNNING_ON_VALGRIND)
	       size_p = 16;
#endif
}

static void parse_args(int argc, char **argv)
{
	_parse_args(argc, argv, 0);
}

#endif /* __DW_CHOLESKY_H__ */
