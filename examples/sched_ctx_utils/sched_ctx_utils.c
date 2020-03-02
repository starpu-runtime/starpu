/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

unsigned size1;
unsigned size2;
unsigned nblocks1;
unsigned nblocks2;
unsigned cpu1;
unsigned cpu2;
unsigned gpu;
unsigned gpu1;
unsigned gpu2;

struct params
{
	unsigned id;
	unsigned ctx;
	int the_other_ctx;
	int *procs;
	int nprocs;
	void (*bench)(unsigned, unsigned);
	unsigned size;
	unsigned nblocks;
};

struct retvals
{
	double flops;
	double avg_timing;
};

#define NSAMPLES 1
int first = 1;
starpu_pthread_mutex_t mut;
struct retvals rv[2];
struct params p1, p2;

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
	struct params *p = (struct params*)val;
	int i;

	STARPU_PTHREAD_SETSPECIFIC(key, &p->id);

	if(p->ctx != 0)
		starpu_sched_ctx_set_context(&p->ctx);

	for(i = 0; i < NSAMPLES; i++)
		p->bench(p->size, p->nblocks);

	if(p->ctx != 0)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&mut);
		if(first)
		{
			starpu_sched_ctx_delete(p->ctx);
		}

		first = 0;
		STARPU_PTHREAD_MUTEX_UNLOCK(&mut);
	}

	rv[p->id].flops /= NSAMPLES;
	rv[p->id].avg_timing /= NSAMPLES;

	return NULL;
}

void start_2benchs(void (*bench)(unsigned, unsigned))
{
	p1.bench = bench;
	p1.size = size1;
	printf("size %u\n", size1);
	p1.nblocks = nblocks1;

	p2.bench = bench;
	p2.size = size2;
	printf("size %u\n", size2);
	p2.nblocks = nblocks2;

	starpu_pthread_t tid[2];
	STARPU_PTHREAD_MUTEX_INIT(&mut, NULL);

	double start;
	double end;

	start = starpu_timing_now();

	STARPU_PTHREAD_CREATE(&tid[0], NULL, (void*)start_bench, (void*)&p1);
	STARPU_PTHREAD_CREATE(&tid[1], NULL, (void*)start_bench, (void*)&p2);

	STARPU_PTHREAD_JOIN(tid[0], NULL);
	STARPU_PTHREAD_JOIN(tid[1], NULL);

	end = starpu_timing_now();

	STARPU_PTHREAD_MUTEX_DESTROY(&mut);

	double timing = end - start;
	timing /= 1000000;

	printf("%2.2f %2.2f ", rv[0].flops, rv[1].flops);
	printf("%2.2f %2.2f %2.2f\n", rv[0].avg_timing, rv[1].avg_timing, timing);

}

void start_1stbench(void (*bench)(unsigned, unsigned))
{
	p1.bench = bench;
	p1.size = size1;
	p1.nblocks = nblocks1;

	double start;
	double end;

	start = starpu_timing_now();

	start_bench((void*)&p1);

	end = starpu_timing_now();

	STARPU_PTHREAD_MUTEX_DESTROY(&mut);

	double timing = end - start;
	timing /= 1000000;

	printf("%2.2f ", rv[0].flops);
	printf("%2.2f %2.2f\n", rv[0].avg_timing, timing);
}

void start_2ndbench(void (*bench)(unsigned, unsigned))
{
	p2.bench = bench;
	p2.size = size2;
	p2.nblocks = nblocks2;

	double start;
	double end;

	start = starpu_timing_now();

	start_bench((void*)&p2);

	end = starpu_timing_now();

	STARPU_PTHREAD_MUTEX_DESTROY(&mut);

	double timing = end - start;
	timing /= 1000000;

	printf("%2.2f ", rv[1].flops);
	printf("%2.2f %2.2f\n", rv[1].avg_timing, timing);
}

void construct_contexts()
{
	unsigned nprocs1 = cpu1 + gpu + gpu1;
	unsigned nprocs2 = cpu2 + gpu + gpu2;
	unsigned n_all_gpus = gpu + gpu1 + gpu2;
	int procs[nprocs1];
	unsigned i;
	int k = 0;

	for(i = 0; i < gpu; i++)
	{
		procs[k++] = i;
		printf("%u ", i);
	}

	for(i = gpu; i < gpu + gpu1; i++)
	{
		procs[k++] = i;
		printf("%u ", i);
	}


	for(i = n_all_gpus; i < n_all_gpus + cpu1; i++)
	{
		procs[k++] = i;
		printf("%u ", i);
	}
	printf("\n ");

	p1.ctx = starpu_sched_ctx_create(procs, nprocs1, "sched_ctx1", STARPU_SCHED_CTX_POLICY_NAME, "heft", 0);
	p2.the_other_ctx = (int)p1.ctx;
	p1.procs = procs;
	p1.nprocs = nprocs1;
	int procs2[nprocs2];

	k = 0;

	for(i = 0; i < gpu; i++)
	{
		procs2[k++] = i;
		printf("%u ", i);
	}

	for(i = gpu + gpu1; i < gpu + gpu1 + gpu2; i++)
	{
		procs2[k++] = i;
		printf("%u ", i);
	}

	for(i = n_all_gpus  + cpu1; i < n_all_gpus + cpu1 + cpu2; i++)
	{
		procs2[k++] = i;
		printf("%u ", i);
	}
	printf("\n");

	p2.ctx = starpu_sched_ctx_create(procs2, nprocs2, "sched_ctx2", STARPU_SCHED_CTX_POLICY_NAME, "heft", 0);
	p1.the_other_ctx = (int)p2.ctx;
	p2.procs = procs2;
	starpu_sched_ctx_set_inheritor(p1.ctx, p2.ctx);
	starpu_sched_ctx_set_inheritor(p2.ctx, p1.ctx);
	p2.nprocs = nprocs2;
}


void parse_args_ctx(int argc, char **argv)
{
	init();
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-size1") == 0)
		{
			char *argptr;
			size1 = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocks1") == 0)
		{
			char *argptr;
			nblocks1 = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-size2") == 0)
		{
			char *argptr;
			size2 = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-nblocks2") == 0)
		{
			char *argptr;
			nblocks2 = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-cpu1") == 0)
		{
			char *argptr;
			cpu1 = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-cpu2") == 0)
		{
			char *argptr;
			cpu2 = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-gpu") == 0)
		{
			char *argptr;
			gpu = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-gpu1") == 0)
		{
			char *argptr;
			gpu1 = strtol(argv[++i], &argptr, 10);
		}

		if (strcmp(argv[i], "-gpu2") == 0)
		{
			char *argptr;
			gpu2 = strtol(argv[++i], &argptr, 10);
		}
	}
}
