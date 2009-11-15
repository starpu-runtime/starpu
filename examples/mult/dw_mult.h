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

#ifndef __MULT_H__
#define __MULT_H__

#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/time.h>
#include <pthread.h>
#include <signal.h>

#include <common/blas.h>
#include <common/blas_model.h>

#include <starpu.h>

#ifdef USE_CUDA
#include <cuda.h>
#include <cublas.h>
#endif

#define MAXSLICESX	64
#define MAXSLICESY	64
#define MAXSLICESZ	64

#define BLAS3_FLOP(n1,n2,n3)	\
	(2*((uint64_t)n1)*((uint64_t)n2)*((uint64_t)n3))

#define BLAS3_LS(n1,n2,n3)    \
	((2*(n1)*(n3) + (n1)*(n2) + (n2)*(n3))*sizeof(float))

struct block_conf {
	uint32_t m;
	uint32_t n;
	uint32_t k;
	uint32_t pad;
};

#define NITER	100

unsigned niter = NITER;
unsigned nslicesx = 4;
unsigned nslicesy = 4;
unsigned nslicesz = 4;
unsigned xdim = 256;
unsigned ydim = 256;
unsigned zdim = 64;
unsigned norandom = 0;
unsigned pin = 0;
unsigned use_common_model = 0;
unsigned check = 0;

/* to compute MFlop/s */
uint64_t flop_cublas = 0;
uint64_t flop_atlas = 0;
uint64_t flop_per_worker[STARPU_NMAXWORKERS] = {0};

/* to compute MB/s (load/store) */
uint64_t ls_cublas = 0;
uint64_t ls_atlas = 0;
uint64_t ls_per_worker[STARPU_NMAXWORKERS] = {0};


struct timeval start;
struct timeval end;

static int taskcounter __attribute__ ((unused));
static struct block_conf conf __attribute__ ((aligned (128)));

#define BLOCKSIZEX	(xdim / nslicesx)
#define BLOCKSIZEY	(ydim / nslicesy)
#define BLOCKSIZEZ	(zdim / nslicesz)

static void display_stats(double timing)
{
	unsigned worker;
	unsigned nworkers = starpu_get_worker_count();

	fprintf(stderr, "Computation took (ms):\n");
	printf("%2.2f\n", timing/1000);

	uint64_t flop_total = 0, ls_total = 0;
	
	for (worker = 0; worker < nworkers; worker++)
	{
		flop_total += flop_per_worker[worker];
		ls_total += ls_per_worker[worker];

		char name[32];
		starpu_get_worker_name(worker, name, 32);

		fprintf(stderr, "\t%s -> %2.2f GFlop\t%2.2f GFlop/s\n", name, (double)flop_per_worker[worker]/1000000000.0f, (double)flop_per_worker[worker]/(double)timing/1000);
	}

	fprintf(stderr, "Total: %2.2f GFlops\t%2.2f GFlop/s\n", (double)flop_total/1000000000.0f, (double)flop_total/(double)timing/1000);
}

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-nblocks") == 0) {
			char *argptr;
			nslicesx = strtol(argv[++i], &argptr, 10);
			nslicesy = nslicesx;
			nslicesz = nslicesx;
		}

		if (strcmp(argv[i], "-nblocksx") == 0) {
			char *argptr;
			nslicesx = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocksy") == 0) {
			char *argptr;
			nslicesy = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocksz") == 0) {
			char *argptr;
			nslicesz = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-x") == 0) {
			char *argptr;
			xdim = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-y") == 0) {
			char *argptr;
			ydim = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-z") == 0) {
			char *argptr;
			zdim = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-iter") == 0) {
			char *argptr;
			niter = strtol(argv[++i], &argptr, 10);
		}


		if (strcmp(argv[i], "-no-random") == 0) {
			norandom = 1;
		}

		if (strcmp(argv[i], "-pin") == 0) {
			pin = 1;
		}

		if (strcmp(argv[i], "-check") == 0) {
			check = 1;
		}

		if (strcmp(argv[i], "-common-model") == 0) {
			use_common_model = 1;
		}
	}

	assert(nslicesx <= MAXSLICESX); 
	assert(nslicesy <= MAXSLICESY); 
	assert(nslicesz <= MAXSLICESZ); 
}

static void display_memory_consumption(void)
{
	fprintf(stderr, "Total memory : %ld MB\n",
		(MAXSLICESY*MAXSLICESZ*sizeof(float *) 
		+ MAXSLICESZ*MAXSLICESX*sizeof(float *)
		+ MAXSLICESY*MAXSLICESX*sizeof(float *)
		+ MAXSLICESY*MAXSLICESZ*sizeof(starpu_data_handle)
		+ MAXSLICESZ*MAXSLICESX*sizeof(starpu_data_handle)
		+ MAXSLICESY*MAXSLICESX*sizeof(starpu_data_handle)
		+ ydim*zdim*sizeof(float)
		+ zdim*xdim*sizeof(float)
		+ ydim*xdim*sizeof(float))/(1024*1024) );
}

#ifdef USE_CUDA
void cublas_mult(starpu_data_interface_t *descr, __attribute__((unused)) void *arg);
#endif

void core_mult(starpu_data_interface_t *descr, __attribute__((unused))  void *arg);

#endif // __MULT_H__
