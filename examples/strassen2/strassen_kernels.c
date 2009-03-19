#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <sys/types.h>
#include <pthread.h>
#include <signal.h>


#include <semaphore.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <core/dependencies/tags.h>
#include <common/timing.h>
#include <common/util.h>
#include <common/blas.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <pthread.h>
#include <signal.h>
#include <common/timing.h>

#include <datawizard/datawizard.h>

#include <task-models/blas_model.h>

#include <common/fxt.h>

#include <core/jobs.h>
#include <core/workers.h>
#include <core/dependencies/tags.h>
#include <datawizard/datawizard.h>

static double cublas_flop = 0.0;
static double cpus_flop = 0.0;

void display_perf(double timing, unsigned size)
{
	double total_flop_n3 = (2.0*size*size*size);
	double total_flop = cublas_flop + cpus_flop;

	fprintf(stderr, "Computation took (ms):\n");
	printf("%2.2f\n", timing/1000);
	fprintf(stderr, "       GFlop : O(n3) -> %2.2f\n",
			(double)total_flop_n3/1000000000.0f);
	fprintf(stderr, "       GFlop : real %2.2f\n",
			(double)total_flop/1000000000.0f);
	fprintf(stderr, "	CPU : %2.2f (%2.2f%%)\n", (double)cpus_flop/1000000000.0, (100.0*cpus_flop)/(cpus_flop + cublas_flop));
	fprintf(stderr, "	GPU : %2.2f (%2.2f%%)\n", (double)cublas_flop/1000000000.0, (100.0*cublas_flop)/(cpus_flop + cublas_flop));
	fprintf(stderr, "       GFlop/s : %2.2f\n", (double)total_flop / (double)timing/1000);
}

static void mult_common_codelet(data_interface_t *buffers, int s, __attribute__((unused))  void *arg)
{
	float *center 	= (float *)buffers[0].blas.ptr;
	float *left 	= (float *)buffers[1].blas.ptr;
	float *right 	= (float *)buffers[2].blas.ptr;

	unsigned n = buffers[0].blas.nx;

	unsigned ld21 = buffers[1].blas.ld;
	unsigned ld12 = buffers[2].blas.ld;
	unsigned ld22 = buffers[0].blas.ld;

	double flop = 2.0*n*n*n;

	switch (s) {
		case 0:
			cpus_flop += flop;
			SGEMM("N", "N", n, n, n, 1.0f, right, ld21, left, ld12, 0.0f, center, ld22);
			break;
#ifdef USE_CUDA
		case 1:
			cublas_flop += flop;
			cublasSgemm('n', 'n', n, n, n, 1.0f, right, ld12, left, ld21, 0.0f, center, ld22);
			break;
#endif
		default:
			STARPU_ASSERT(0);
			break;
	}
}

void mult_core_codelet(data_interface_t *descr, void *_args)
{
	mult_common_codelet(descr, 0, _args);
}

#ifdef USE_CUDA
void mult_cublas_codelet(data_interface_t *descr, void *_args)
{
	mult_common_codelet(descr, 1, _args);
}
#endif

static void add_sub_common_codelet(data_interface_t *buffers, int s, __attribute__((unused))  void *arg, float alpha)
{
	/* C = A op B */

	float *C 	= (float *)buffers[0].blas.ptr;
	float *A 	= (float *)buffers[1].blas.ptr;
	float *B 	= (float *)buffers[2].blas.ptr;

	unsigned n = buffers[0].blas.nx;

	unsigned ldA = buffers[1].blas.ld;
	unsigned ldB = buffers[2].blas.ld;
	unsigned ldC = buffers[0].blas.ld;

	double flop = 2.0*n*n;

	// TODO check dim ...

	unsigned line;

	switch (s) {
		case 0:
			cpus_flop += flop;
			for (line = 0; line < n; line++)
			{
				/* copy line A into C */
				SAXPY(n, 1.0f, &A[line*ldA], 1, &C[line*ldC], 1);
				/* add line B to C = A */
				SAXPY(n, alpha, &B[line*ldB], 1, &C[line*ldC], 1);
			}
			break;
#ifdef USE_CUDA
		case 1:
			cublas_flop += flop;
			for (line = 0; line < n; line++)
			{
				/* copy line A into C */
				cublasSaxpy(n, 1.0f, &A[line*ldA], 1, &C[line*ldC], 1);
				/* add line B to C = A */
				cublasSaxpy(n, alpha, &B[line*ldB], 1, &C[line*ldC], 1);
			}

			break;
#endif
		default:
			STARPU_ASSERT(0);
			break;
	}
}

void sub_core_codelet(data_interface_t *descr, __attribute__((unused))  void *arg)
{
	add_sub_common_codelet(descr, 0, arg, -1.0f);
}

void add_core_codelet(data_interface_t *descr, __attribute__((unused))  void *arg)
{
	add_sub_common_codelet(descr, 0, arg, 1.0f);
}

#ifdef USE_CUDA
void sub_cublas_codelet(data_interface_t *descr, __attribute__((unused))  void *arg)
{
	add_sub_common_codelet(descr, 1, arg, -1.0f);
}

void add_cublas_codelet(data_interface_t *descr, __attribute__((unused))  void *arg)
{
	add_sub_common_codelet(descr, 1, arg, 1.0f);
}
#endif


static void self_add_sub_common_codelet(data_interface_t *buffers, int s, __attribute__((unused))  void *arg, float alpha)
{
	/* C +=/-= A */

	float *C 	= (float *)buffers[0].blas.ptr;
	float *A 	= (float *)buffers[1].blas.ptr;

	unsigned n = buffers[0].blas.nx;

	unsigned ldA = buffers[1].blas.ld;
	unsigned ldC = buffers[0].blas.ld;

	double flop = 1.0*n*n;

	// TODO check dim ...
	
	unsigned line;

	switch (s) {
		case 0:
			cpus_flop += flop;
			for (line = 0; line < n; line++)
			{
				/* add line A to C */
				SAXPY(n, alpha, &A[line*ldA], 1, &C[line*ldC], 1);
			}
			break;
#ifdef USE_CUDA
		case 1:
			cublas_flop += flop;
			for (line = 0; line < n; line++)
			{
				/* add line A to C */
				cublasSaxpy(n, alpha, &A[line*ldA], 1, &C[line*ldC], 1);
			}
			break;
#endif
		default:
			STARPU_ASSERT(0);
			break;
	}
}




void self_add_core_codelet(data_interface_t *descr, __attribute__((unused))  void *arg)
{
	self_add_sub_common_codelet(descr, 0, arg, 1.0f);
}

void self_sub_core_codelet(data_interface_t *descr, __attribute__((unused))  void *arg)
{
	self_add_sub_common_codelet(descr, 0, arg, -1.0f);
}

#ifdef USE_CUDA
void self_add_cublas_codelet(data_interface_t *descr, __attribute__((unused))  void *arg)
{
	self_add_sub_common_codelet(descr, 1, arg, 1.0f);
}

void self_sub_cublas_codelet(data_interface_t *descr, __attribute__((unused))  void *arg)
{
	self_add_sub_common_codelet(descr, 1, arg, -1.0f);
}
#endif

/* this codelet does nothing  */
void null_codelet(__attribute__((unused)) data_interface_t *descr,
		  __attribute__((unused))  void *arg)
{
}
