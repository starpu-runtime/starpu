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

#include "strassen.h"
#include <sys/time.h>

unsigned dim = 4096;
unsigned reclevel = 4;
unsigned norandom = 0;

sem_t sem;

float *A;
float *B;
float *C;

starpu_data_handle A_state;
starpu_data_handle B_state;
starpu_data_handle C_state;

struct timeval start;
struct timeval end;


/* to compute MFlop/s */
uint64_t flop_cublas = 0;
uint64_t flop_atlas = 0;

/* to compute MB/s (load/store) */
uint64_t ls_cublas = 0;
uint64_t ls_atlas = 0;

/* 
 * Strassen complexity : n = 2^k matrices, stops at 2^r : recursion = (k-r) levels
 * 	m = n / 2^rec
 * 	M(k) = 7^(k-r) 8^r = 7^rec (m^3)
 * 	A(k) = 4^r (2^r + 5) 7^(k-r) - 6 x 4^k = (m^2)(m+5)*7^rec - 6n^2 
 *
 * 	4n^2.807
 */
double strassen_complexity(unsigned n, unsigned rec)
{
	double mult, add;

	double m = (1.0*n)/(pow(2.0, (double)rec));

	add = ((m*m)*(m+5)*(pow(7.0, (double)rec)) - 6.0*n*n);
	mult = (m*m*m)*(pow(7.0, (double)rec));
	
	//printf("%e adds %e mult\n", add, mult);

	return (add+mult);
}

/*
 * That program should compute C = A * B 
 * 
 *   A of size (z,y)
 *   B of size (x,z)
 *   C of size (x,y)
 */

void terminate(void *arg __attribute__ ((unused)))
{
	gettimeofday(&end, NULL);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	//uint64_t total_flop = flop_cublas + flop_atlas;
	double total_flop =  strassen_complexity(dim, reclevel);//4.0*pow((double)dim, 2.807);

	fprintf(stderr, "Computation took (ms):\n");
	printf("%2.2f\n", timing/1000);
	fprintf(stderr, "	GFlop : total (%2.2f)\n", (double)total_flop/1000000000.0f);
	fprintf(stderr, "	GFlop/s : %2.2f\n", (double)total_flop / (double)timing/1000);

	sem_post(&sem);
}

void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-size") == 0) {
			char *argptr;
			dim = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-rec") == 0) {
			char *argptr;
			reclevel = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-no-random") == 0) {
			norandom = 1;
		}
	}
}

void init_problem(void)
{
	unsigned i,j;

#ifdef STARPU_USE_FXT
	fxt_register_thread(0);
#endif

	A = malloc(dim*dim*sizeof(float));
	B = malloc(dim*dim*sizeof(float));
	C = malloc(dim*dim*sizeof(float));

	/* fill the A and B matrices */
	if (norandom) {
		for (i=0; i < dim; i++) {
			for (j=0; j < dim; j++) {
				A[i+j*dim] = (float)(i);
			}
		}
	
		for (i=0; i < dim; i++) {
			for (j=0; j < dim; j++) {
				B[i+j*dim] = (float)(j);
			}
		}
	} 
	else {
		srand(2008);
		for (j=0; j < dim; j++) {
			for (i=0; i < dim; i++) {
				A[i+j*dim] = (float)(starpu_drand48());
			}
		}
	
		for (j=0; j < dim; j++) {
			for (i=0; i < dim; i++) {
				B[i+j*dim] = (float)(starpu_drand48());
			}
		}
	}

	for (j=0; j < dim; j++) {
		for (i=0; i < dim; i++) {
			C[i+j*dim] = (float)(0);
		}
	}

	starpu_register_blas_data(&A_state, 0, (uintptr_t)A, 
		dim, dim, dim, sizeof(float));
	starpu_register_blas_data(&B_state, 0, (uintptr_t)B, 
		dim, dim, dim, sizeof(float));
	starpu_register_blas_data(&C_state, 0, (uintptr_t)C, 
		dim, dim, dim, sizeof(float));

	gettimeofday(&start, NULL);
	strassen(A_state, B_state, C_state, terminate, NULL, reclevel);
}

int main(__attribute__ ((unused)) int argc, 
	 __attribute__ ((unused)) char **argv)
{

	parse_args(argc, argv);

	/* start the runtime */
	starpu_init(NULL);

	starpu_helper_init_cublas();

	sem_init(&sem, 0, 0U);

	init_problem();
	sem_wait(&sem);
	sem_destroy(&sem);

	starpu_helper_shutdown_cublas();

	starpu_shutdown();

	return 0;
}
