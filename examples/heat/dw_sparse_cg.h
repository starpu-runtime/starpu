#ifndef __DW_SPARSE_CG_H__
#define __DW_SPARSE_CG_H__

#include <stdio.h>
#include <stdint.h>
#include <semaphore.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/types.h>
#include <pthread.h>
#include <signal.h>

#include <starpu_config.h>
#include <starpu.h>

#ifdef USE_CUDA
#include <cublas.h>
#endif

#include <common/timing.h>
#include "../common/blas.h"

#define MAXITER	100000
#define EPSILON	0.0000001f

/* code parameters */
static uint32_t size = 33554432;
static unsigned usecpu = 0;
static unsigned blocks = 512;
static unsigned grids  = 8;

struct cg_problem {
	data_handle ds_matrixA;
	data_handle ds_vecx;
	data_handle ds_vecb;
	data_handle ds_vecr;
	data_handle ds_vecd;
	data_handle ds_vecq;

	sem_t *sem;
	
	float alpha;
	float beta;
	float delta_0;
	float delta_old;
	float delta_new;
	float epsilon;

	int i;
	unsigned size;
};

/* some useful functions */
static void __attribute__((unused)) parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-size") == 0) {
			char *argptr;
			size = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-block") == 0) {
			char *argptr;
			blocks = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-grid") == 0) {
			char *argptr;
			grids = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-cpu") == 0) {
			usecpu = 1;
		}
	}
}


static void __attribute__ ((unused)) print_results(float *result, unsigned size)
{
	printf("**** RESULTS **** \n");
	unsigned i;

	for (i = 0; i < STARPU_MIN(size, 16); i++)
	{
		printf("%d -> %f\n", i, result[i]);
	}
}

void core_codelet_func_1(data_interface_t *descr, void *arg);

void core_codelet_func_2(data_interface_t *descr, void *arg);

void cublas_codelet_func_3(data_interface_t *descr, void *arg);
void core_codelet_func_3(data_interface_t *descr, void *arg);

void core_codelet_func_4(data_interface_t *descr, void *arg);

void core_codelet_func_5(data_interface_t *descr, void *arg);
void cublas_codelet_func_5(data_interface_t *descr, void *arg);

void cublas_codelet_func_6(data_interface_t *descr, void *arg);
void core_codelet_func_6(data_interface_t *descr, void *arg);

void cublas_codelet_func_7(data_interface_t *descr, void *arg);
void core_codelet_func_7(data_interface_t *descr, void *arg);

void cublas_codelet_func_8(data_interface_t *descr, void *arg);
void core_codelet_func_8(data_interface_t *descr, void *arg);

void cublas_codelet_func_9(data_interface_t *descr, void *arg);
void core_codelet_func_9(data_interface_t *descr, void *arg);

void iteration_cg(void *problem);

void conjugate_gradient(float *nzvalA, float *vecb, float *vecx, uint32_t nnz,
			unsigned nrow, uint32_t *colind, uint32_t *rowptr);

#endif // __DW_SPARSE_CG_H__
