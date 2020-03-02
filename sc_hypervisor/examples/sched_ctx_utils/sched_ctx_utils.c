/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "sched_ctx_utils.h"
#include <starpu.h>
#include "sc_hypervisor.h"
#define NSAMPLES 3

unsigned size1;
unsigned size2;
unsigned nblocks1;
unsigned nblocks2;
unsigned cpu1;
unsigned cpu2;
unsigned gpu;
unsigned gpu1;
unsigned gpu2;

typedef struct
{
	unsigned id;
	unsigned ctx;
	int the_other_ctx;
	int *workers;
	int nworkers;
	void (*bench)(float*, unsigned, unsigned);
	unsigned size;
	unsigned nblocks;
	float *mat[NSAMPLES];
} params;

typedef struct
{
	double flops;
	double avg_timing;
} retvals;

int first = 1;
starpu_pthread_mutex_t mut;
retvals rv[2];
params p1, p2;
int it = 0;
int it2 = 0;

starpu_pthread_key_t key;

void init()
{
	size1 = 4*1024;
	size2 = 4*1024;
	nblocks1 = 16;
	nblocks2 = 16;
	cpu1 = 0;
	cpu2 = 0;
	gpu = 0;
	gpu1 = 0;
	gpu2 = 0;

	rv[0].flops = 0.0;
	rv[1].flops = 0.0;
	rv[1].avg_timing = 0.0;
	rv[1].avg_timing = 0.0;

	p1.ctx = 0;
	p2.ctx = 0;

	p1.id = 0;
	p2.id = 1;
	STARPU_PTHREAD_KEY_CREATE(&key, NULL);
}

void update_sched_ctx_timing_results(double flops, double avg_timing)
{
	unsigned *id = STARPU_PTHREAD_GETSPECIFIC(key);
	rv[*id].flops += flops;
	rv[*id].avg_timing += avg_timing;
}

void* start_bench(void *val)
{
	params *p = (params*)val;
	int i;

	STARPU_PTHREAD_SETSPECIFIC(key, &p->id);

	if(p->ctx != 0)
		starpu_sched_ctx_set_context(&p->ctx);

	for(i = 0; i < NSAMPLES; i++)
		p->bench(p->mat[i], p->size, p->nblocks);

	/* if(p->ctx != 0) */
	/* { */
	/* 	STARPU_PTHREAD_MUTEX_LOCK(&mut); */
	/* 	if(first){ */
	/* 		sc_hypervisor_unregiser_ctx(p->ctx); */
	/* 		starpu_sched_ctx_delete(p->ctx, p->the_other_ctx); */
	/* 	} */

	/* 	first = 0; */
	/* 	STARPU_PTHREAD_MUTEX_UNLOCK(&mut); */
	/* } */
	sc_hypervisor_stop_resize(p->the_other_ctx);
	rv[p->id].flops /= NSAMPLES;
	rv[p->id].avg_timing /= NSAMPLES;

	return NULL;
}

float* construct_matrix(unsigned size)
{
	float *mat;
	starpu_malloc((void **)&mat, (size_t)size*size*sizeof(float));

	unsigned i,j;
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			mat[j +i*size] = (1.0f/(1.0f+i+j)) + ((i == j)?1.0f*size:0.0f);
			/* mat[j +i*size] = ((i == j)?1.0f*size:0.0f); */
		}
	}
	return mat;
}
void start_2benchs(void (*bench)(float*, unsigned, unsigned))
{
	p1.bench = bench;
	p1.size = size1;
	p1.nblocks = nblocks1;

	p2.bench = bench;
	p2.size = size2;
	p2.nblocks = nblocks2;

	int i;
	for(i = 0; i < NSAMPLES; i++)
	{
		p1.mat[i] = construct_matrix(p1.size);
		p2.mat[i] = construct_matrix(p2.size);
	}

	starpu_pthread_t tid[2];
	STARPU_PTHREAD_MUTEX_INIT(&mut, NULL);

	struct timeval start;
	struct timeval end;

	gettimeofday(&start, NULL);

	STARPU_PTHREAD_CREATE(&tid[0], NULL, (void*)start_bench, (void*)&p1);
	STARPU_PTHREAD_CREATE(&tid[1], NULL, (void*)start_bench, (void*)&p2);

	STARPU_PTHREAD_JOIN(tid[0], NULL);
	STARPU_PTHREAD_JOIN(tid[1], NULL);

	gettimeofday(&end, NULL);

	STARPU_PTHREAD_MUTEX_DESTROY(&mut);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	timing /= 1000000;

	printf("%2.2f %2.2f ", rv[0].flops, rv[1].flops);
	printf("%2.2f %2.2f %2.2f\n", rv[0].avg_timing, rv[1].avg_timing, timing);

}

void start_1stbench(void (*bench)(float*, unsigned, unsigned))
{
	p1.bench = bench;
	p1.size = size1;
	p1.nblocks = nblocks1;

	int i;
	for(i = 0; i < NSAMPLES; i++)
	{
		p1.mat[i] = construct_matrix(p1.size);
	}

	struct timeval start;
	struct timeval end;

	gettimeofday(&start, NULL);

	start_bench((void*)&p1);

	gettimeofday(&end, NULL);

	STARPU_PTHREAD_MUTEX_DESTROY(&mut);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	timing /= 1000000;

	printf("%2.2f ", rv[0].flops);
	printf("%2.2f %2.2f\n", rv[0].avg_timing, timing);
}

void start_2ndbench(void (*bench)(float*, unsigned, unsigned))
{
	p2.bench = bench;
	p2.size = size2;
	p2.nblocks = nblocks2;
	int i;
	for(i = 0; i < NSAMPLES; i++)
	{
		p2.mat[i] = construct_matrix(p2.size);
	}

	struct timeval start;
	struct timeval end;

	gettimeofday(&start, NULL);

	start_bench((void*)&p2);

	gettimeofday(&end, NULL);

	STARPU_PTHREAD_MUTEX_DESTROY(&mut);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	timing /= 1000000;

	printf("%2.2f ", rv[1].flops);
	printf("%2.2f %2.2f\n", rv[1].avg_timing, timing);
}

void construct_contexts()
{
	struct sc_hypervisor_policy policy;
	policy.custom = 0;
	policy.name = "idle";
	void *perf_counters = sc_hypervisor_init(&policy);
	int nworkers1 = cpu1 + gpu + gpu1;
	int nworkers2 = cpu2 + gpu + gpu2;
	/* unsigned n_all_gpus = gpu + gpu1 + gpu2; */


	int i;
	/* int k = 0; */
	nworkers1 = 12;
	p1.workers = (int*)malloc(nworkers1*sizeof(int));

	/* for(i = 0; i < gpu; i++) */
	/* 	p1.workers[k++] = i; */

	/* for(i = gpu; i < gpu + gpu1; i++) */
	/* 	p1.workers[k++] = i; */


	/* for(i = n_all_gpus; i < n_all_gpus + cpu1; i++) */
	/* 	p1.workers[k++] = i; */


	for(i = 0; i < 12; i++)
		p1.workers[i] = i;

	p1.ctx = starpu_sched_ctx_create(p1.workers, nworkers1, "sched_ctx1", STARPU_SCHED_CTX_POLICY_NAME, "heft", 0);
	starpu_sched_ctx_set_perf_counters(p1.ctx, perf_counters);
	p2.the_other_ctx = (int)p1.ctx;
	p1.nworkers = nworkers1;
	sc_hypervisor_register_ctx(p1.ctx, 0.0);

	/* sc_hypervisor_ctl(p1.ctx, */
	/* 			   SC_HYPERVISOR_MAX_IDLE, p1.workers, p1.nworkers, 5000.0, */
	/* 			   SC_HYPERVISOR_MAX_IDLE, p1.workers, gpu+gpu1, 100000.0, */
	/* 			   SC_HYPERVISOR_EMPTY_CTX_MAX_IDLE, p1.workers, p1.nworkers, 500000.0, */
	/* 			   SC_HYPERVISOR_GRANULARITY, 2, */
	/* 			   SC_HYPERVISOR_MIN_TASKS, 1000, */
	/* 			   SC_HYPERVISOR_NEW_WORKERS_MAX_IDLE, 100000.0, */
	/* 			   SC_HYPERVISOR_MIN_WORKERS, 6, */
	/* 			   SC_HYPERVISOR_MAX_WORKERS, 12, */
	/* 			   NULL); */

	sc_hypervisor_ctl(p1.ctx,
				   SC_HYPERVISOR_GRANULARITY, 2,
				   SC_HYPERVISOR_MIN_TASKS, 1000,
				   SC_HYPERVISOR_MIN_WORKERS, 6,
				   SC_HYPERVISOR_MAX_WORKERS, 12,
				   NULL);

	/* k = 0; */
	p2.workers = (int*)malloc(nworkers2*sizeof(int));

	/* for(i = 0; i < gpu; i++) */
	/* 	p2.workers[k++] = i; */

	/* for(i = gpu + gpu1; i < gpu + gpu1 + gpu2; i++) */
	/* 	p2.workers[k++] = i; */

	/* for(i = n_all_gpus  + cpu1; i < n_all_gpus + cpu1 + cpu2; i++) */
	/* 	p2.workers[k++] = i; */

	p2.ctx = starpu_sched_ctx_create(p2.workers, 0, "sched_ctx2", STARPU_SCHED_CTX_POLICY_NAME, "heft", 0);
	starpu_sched_ctx_set_perf_counters(p2.ctx, perf_counters);
	p1.the_other_ctx = (int)p2.ctx;
	p2.nworkers = 0;
	sc_hypervisor_register_ctx(p2.ctx, 0.0);

	/* sc_hypervisor_ctl(p2.ctx, */
	/* 			   SC_HYPERVISOR_MAX_IDLE, p2.workers, p2.nworkers, 2000.0, */
	/* 			   SC_HYPERVISOR_MAX_IDLE, p2.workers, gpu+gpu2, 5000.0, */
	/* 			   SC_HYPERVISOR_EMPTY_CTX_MAX_IDLE, p1.workers, p1.nworkers, 500000.0, */
	/* 			   SC_HYPERVISOR_GRANULARITY, 2, */
	/* 			   SC_HYPERVISOR_MIN_TASKS, 500, */
	/* 			   SC_HYPERVISOR_NEW_WORKERS_MAX_IDLE, 1000.0, */
	/* 			   SC_HYPERVISOR_MIN_WORKERS, 4, */
	/* 			   SC_HYPERVISOR_MAX_WORKERS, 8, */
	/* 			   NULL); */

	sc_hypervisor_ctl(p2.ctx,
				   SC_HYPERVISOR_GRANULARITY, 2,
				   SC_HYPERVISOR_MIN_TASKS, 500,
				   SC_HYPERVISOR_MIN_WORKERS, 0,
				   SC_HYPERVISOR_MAX_WORKERS, 6,
				   NULL);

}

void set_hypervisor_conf(int event, int task_tag)
{
/* 	unsigned *id = STARPU_PTHREAD_GETSPECIFIC(key); */
/* 	if(*id == 0) */
/* 	{ */
/* 		if(event == END_BENCH) */
/* 		{ */
/* 			if(it < 2) */
/* 			{ */
/* 				sc_hypervisor_ctl(p2.ctx, */
/* 							   SC_HYPERVISOR_MIN_WORKERS, 2, */
/* 							   SC_HYPERVISOR_MAX_WORKERS, 4, */
/* 							   SC_HYPERVISOR_TIME_TO_APPLY, task_tag, */
/* 							   NULL); */

/* 				printf("%d: set max %d for tag %d\n", p2.ctx, 4, task_tag); */
/* 				sc_hypervisor_ctl(p1.ctx, */
/* 							   SC_HYPERVISOR_MIN_WORKERS, 6, */
/* 							   SC_HYPERVISOR_MAX_WORKERS, 8, */
/* 							   SC_HYPERVISOR_TIME_TO_APPLY, task_tag, */
/* 							   NULL); */
/* 				printf("%d: set max %d for tag %d\n", p1.ctx, 8, task_tag); */
/* 				sc_hypervisor_resize(p1.ctx, task_tag); */
/* 			} */
/* 			if(it == 2) */
/* 			{ */
/* 				sc_hypervisor_ctl(p2.ctx, */
/* 							   SC_HYPERVISOR_MIN_WORKERS, 12, */
/* 							   SC_HYPERVISOR_MAX_WORKERS, 12, */
/* 							   SC_HYPERVISOR_TIME_TO_APPLY, task_tag, */
/* 							   NULL); */
/* 				printf("%d: set max %d for tag %d\n", p2.ctx, 12, task_tag); */
/* 				sc_hypervisor_ctl(p1.ctx, */
/* 							   SC_HYPERVISOR_MIN_WORKERS, 0, */
/* 							   SC_HYPERVISOR_MAX_WORKERS, 0, */
/* 							   SC_HYPERVISOR_TIME_TO_APPLY, task_tag, */
/* 							   NULL); */
/* 				printf("%d: set max %d for tag %d\n", p1.ctx, 0, task_tag); */
/* 				sc_hypervisor_resize(p1.ctx, task_tag); */
/* 			} */
/* 			it++; */

/* 		} */
/* 	} */
/* 	else */
/* 	{ */
/* 		if(event == END_BENCH) */
/* 		{ */
/* 			if(it2 < 3) */
/* 			{ */
/* 				sc_hypervisor_ctl(p1.ctx, */
/* 							   SC_HYPERVISOR_MIN_WORKERS, 6, */
/* 							   SC_HYPERVISOR_MAX_WORKERS, 12, */
/* 							   SC_HYPERVISOR_TIME_TO_APPLY, task_tag, */
/* 							   NULL); */
/* 				printf("%d: set max %d for tag %d\n", p1.ctx, 12, task_tag); */
/* 				sc_hypervisor_ctl(p2.ctx, */
/* 							   SC_HYPERVISOR_MIN_WORKERS, 0, */
/* 							   SC_HYPERVISOR_MAX_WORKERS, 0, */
/* 							   SC_HYPERVISOR_TIME_TO_APPLY, task_tag, */
/* 							   NULL); */
/* 				printf("%d: set max %d for tag %d\n", p2.ctx, 0, task_tag); */
/* 				sc_hypervisor_resize(p2.ctx, task_tag); */
/* 			} */
/* 			it2++; */
/* 		} */
/* 	} */

	/* if(*id == 1) */
	/* { */
	/* 	if(event == START_BENCH) */
	/* 	{ */
	/* 		int workers[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; */
	/* 		sc_hypervisor_ctl(p1.ctx, */
	/* 					   SC_HYPERVISOR_MAX_IDLE, workers, 12, 800000.0, */
	/* 					   SC_HYPERVISOR_TIME_TO_APPLY, task_tag, */
	/* 					   NULL); */
	/* 	} */
	/* 	else */
	/* 	{ */
	/* 		if(it2 < 2) */
	/* 		{ */
	/* 			int workers[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; */
	/* 			sc_hypervisor_ctl(p2.ctx, */
	/* 						   SC_HYPERVISOR_MAX_IDLE, workers, 12, 500.0, */
	/* 						   SC_HYPERVISOR_MAX_IDLE, workers, 3, 200.0, */
	/* 						   SC_HYPERVISOR_TIME_TO_APPLY, task_tag, */
	/* 						   NULL); */
	/* 		} */
	/* 		if(it2 == 2) */
	/* 		{ */
	/* 			int workers[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; */
	/* 			sc_hypervisor_ctl(p2.ctx, */
	/* 						   SC_HYPERVISOR_MAX_IDLE, workers, 12, 1000.0, */
	/* 						   SC_HYPERVISOR_MAX_IDLE, workers, 3, 500.0, */
	/* 						   SC_HYPERVISOR_TIME_TO_APPLY, task_tag, */
	/* 						   SC_HYPERVISOR_MAX_WORKERS, 12, */
	/* 						   NULL); */
	/* 		} */
	/* 		it2++; */
	/* 	} */

	/* } else { */
	/* 	if(event == START_BENCH) */
	/* 	{ */
	/* 		int workers[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; */
	/* 		sc_hypervisor_ctl(p1.ctx, */
	/* 					   SC_HYPERVISOR_MAX_IDLE, workers, 12, 1500.0, */
	/* 					   SC_HYPERVISOR_MAX_IDLE, workers, 3, 4000.0, */
	/* 					   SC_HYPERVISOR_TIME_TO_APPLY, task_tag, */
	/* 					   NULL); */
	/* 	} */
	/* 	if(event == END_BENCH) */
	/* 	{ */
	/* 		if(it < 2) */
	/* 		{ */
	/* 			int workers[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; */
	/* 			sc_hypervisor_ctl(p1.ctx, */
	/* 						   SC_HYPERVISOR_MAX_IDLE, workers, 12, 100.0, */
	/* 						   SC_HYPERVISOR_MAX_IDLE, workers, 3, 5000.0, */
	/* 						   SC_HYPERVISOR_TIME_TO_APPLY, task_tag, */
	/* 						   NULL); */
	/* 		} */
	/* 		if(it == 2) */
	/* 		{ */
	/* 			int workers[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; */
	/* 			sc_hypervisor_ctl(p1.ctx, */
	/* 						   SC_HYPERVISOR_MAX_IDLE, workers, 12, 5000.0, */
	/* 						   SC_HYPERVISOR_MAX_IDLE, workers, 3, 10000.0, */
	/* 						   SC_HYPERVISOR_TIME_TO_APPLY, task_tag, */
	/* 						   NULL); */
	/* 		} */

	/* 		it++; */
	/* 	} */

	/* } */
}

void end_contexts()
{
	free(p1.workers);
	free(p2.workers);
	sc_hypervisor_shutdown();
}

void parse_args_ctx(int argc, char **argv)
{
	init();
	int i;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-size1") == 0) {
			char *argptr;
			size1 = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocks1") == 0) {
			char *argptr;
			nblocks1 = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-size2") == 0) {
			char *argptr;
			size2 = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocks2") == 0) {
			char *argptr;
			nblocks2 = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-cpu1") == 0) {
			char *argptr;
			cpu1 = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-cpu2") == 0) {
			char *argptr;
			cpu2 = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-gpu") == 0) {
			char *argptr;
			gpu = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-gpu1") == 0) {
			char *argptr;
			gpu1 = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-gpu2") == 0) {
			char *argptr;
			gpu2 = strtol(argv[++i], &argptr, 10);
		}
	}
}
